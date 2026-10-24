"""Candidate place-and-time slot ranker.

Genuinely compares (Venue, Forecast Hour) combinations, selecting the optimal
space-time recommendation across multiple candidate locations and upcoming hours.

Strict Invariant:
Maintains explicit distinction between:
- Relative ranking: best among the evaluated venue-hour combinations.
- Absolute recommendation status: recommended, caution, or not_recommended.
A top-ranked candidate is NEVER automatically described as safe or suitable.
When all slots are unfavorable, returns an explicit no-suitable-option result.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.common.coordinates import LatLon
from src.common.schemas import HourlyAQIObservation, HourlyWeatherObservation, PersonalHealthProfile, PlaceCandidate
from src.scoring.evaluator import RecommendationStatus, SuitabilityEvaluator, SuitabilityScoreBreakdown
from src.spatial_analysis.wind_dispersion import GeographicContextReport, analyze_geographic_context


class PlaceTimeSlotRecommendation(BaseModel):
    """A scored combination of a specific venue at a specific forecast hour."""
    venue_id: str
    venue_name: str
    place_type: str
    coordinate: LatLon
    distance_km: Optional[float] = None
    slot_time_ist: str
    suitability_score: float
    rating_category: str
    recommendation_status: str = Field(
        ...,
        description="'recommended', 'caution', or 'not_recommended'"
    )
    is_favorable: bool
    score_breakdown: SuitabilityScoreBreakdown
    geographic_context: GeographicContextReport
    weather_summary: Dict[str, Any]
    aqi_summary: Dict[str, Any]


class CandidateRankingResult(BaseModel):
    """Aggregate result from ranking multiple venues across multiple hours."""
    activity_id: str
    total_venues_evaluated: int
    total_slots_evaluated: int
    has_suitable_option: bool = Field(
        ...,
        description="True if at least one evaluated slot is recommended or acceptable caution; False if all are unfavorable/not_recommended."
    )
    no_suitable_option_reason: Optional[str] = None
    best_recommendation: Optional[PlaceTimeSlotRecommendation] = None
    top_candidates: List[PlaceTimeSlotRecommendation] = Field(default_factory=list)
    temporal_comparison_note: Optional[str] = None


class CandidateRanker:
    """Evaluates the cross-product of (Candidate Venues x Target Hours)."""

    def __init__(self, evaluator: Optional[SuitabilityEvaluator] = None):
        self.evaluator = evaluator or SuitabilityEvaluator()

    def rank_candidates(
        self,
        activity_id: str,
        candidates: List[PlaceCandidate],
        weather_forecasts: List[HourlyWeatherObservation],
        aqi_forecasts: List[HourlyAQIObservation],
        target_slots: List[datetime],
        health_profile: Optional[PersonalHealthProfile] = None,
        top_k: int = 5,
    ) -> CandidateRankingResult:
        """Rank all (Place, Hour) candidate combinations."""
        if not candidates or not target_slots or not weather_forecasts:
            return CandidateRankingResult(
                activity_id=activity_id,
                total_venues_evaluated=len(candidates),
                total_slots_evaluated=0,
                has_suitable_option=False,
                no_suitable_option_reason="Insufficient venue or weather forecast data to evaluate outdoor slots.",
                best_recommendation=None,
                top_candidates=[],
                temporal_comparison_note="Insufficient data.",
            )

        weather_map: Dict[str, HourlyWeatherObservation] = {
            w.time.strftime("%Y-%m-%d %H:00"): w for w in weather_forecasts
        }
        aqi_map: Dict[str, HourlyAQIObservation] = {
            a.time.strftime("%Y-%m-%d %H:00"): a for a in aqi_forecasts
        }

        all_slots: List[PlaceTimeSlotRecommendation] = []

        for candidate in candidates:
            for slot_dt in target_slots:
                key = slot_dt.strftime("%Y-%m-%d %H:00")
                weather_obs = weather_map.get(key)
                if not weather_obs:
                    continue

                aqi_obs = aqi_map.get(key)

                geo_report = analyze_geographic_context(
                    candidate=candidate,
                    wind_direction_deg=weather_obs.wind_direction_10m,
                    wind_speed_kmh=weather_obs.wind_speed_10m,
                )

                score_breakdown = self.evaluator.evaluate(
                    activity_id=activity_id,
                    weather=weather_obs,
                    aqi=aqi_obs,
                    geo_report=geo_report,
                    health_profile=health_profile,
                )

                slot_rec = PlaceTimeSlotRecommendation(
                    venue_id=candidate.id,
                    venue_name=candidate.name,
                    place_type=candidate.place_type,
                    coordinate=candidate.coordinate,
                    distance_km=candidate.distance_from_anchor_km,
                    slot_time_ist=slot_dt.strftime("%Y-%m-%d %H:%M IST"),
                    suitability_score=score_breakdown.overall_score,
                    rating_category=score_breakdown.rating_category,
                    recommendation_status=score_breakdown.recommendation_status,
                    is_favorable=score_breakdown.is_favorable,
                    score_breakdown=score_breakdown,
                    geographic_context=geo_report,
                    weather_summary={
                        "temperature_c": weather_obs.temperature_2m,
                        "humidity_percent": weather_obs.relative_humidity_2m,
                        "rain_probability_percent": weather_obs.precipitation_probability,
                        "precipitation_mm": weather_obs.precipitation,
                        "wind_speed_kmh": weather_obs.wind_speed_10m,
                        "wind_direction_deg": weather_obs.wind_direction_10m,
                        "uv_index": weather_obs.uv_index,
                    },
                    aqi_summary={
                        "pm2_5": aqi_obs.pm2_5 if aqi_obs else None,
                        "pm10": aqi_obs.pm10 if aqi_obs else None,
                        "us_aqi": aqi_obs.us_aqi if aqi_obs else None,
                    },
                )
                all_slots.append(slot_rec)

        # Sort all (Place, Hour) pairs by suitability score descending
        all_slots.sort(key=lambda s: s.suitability_score, reverse=True)

        best_slot = all_slots[0] if all_slots else None

        # Check absolute recommendation status across all evaluated slots
        favorable_count = sum(1 for s in all_slots if s.recommendation_status == RecommendationStatus.RECOMMENDED.value)
        caution_count = sum(1 for s in all_slots if s.recommendation_status == RecommendationStatus.CAUTION.value)
        not_rec_count = sum(1 for s in all_slots if s.recommendation_status == RecommendationStatus.NOT_RECOMMENDED.value)

        # Absolute distinction: If all evaluated slots are not_recommended, has_suitable_option is False!
        has_suitable = (favorable_count > 0 or caution_count > 0)
        no_suitable_reason = None

        if not has_suitable:
            reasons = set()
            for s in all_slots:
                reasons.update(s.score_breakdown.unfavorable_reasons)
            reason_str = "; ".join(reasons) if reasons else "environmental conditions exceed safety guidelines"
            no_suitable_reason = (
                f"No suitable outdoor window found for {activity_id} across all {len(all_slots)} evaluated venue-hour slots. "
                f"Primary factors: {reason_str}. Postponing outdoor activity or choosing an indoor alternative is advised."
            )

        # Build temporal comparison note
        temporal_note = None
        if best_slot and len(target_slots) > 1:
            best_venue_slots = [s for s in all_slots if s.venue_id == best_slot.venue_id]
            if len(best_venue_slots) > 1:
                lowest_slot = best_venue_slots[-1]
                temporal_note = (
                    f"At {best_slot.venue_name}, {best_slot.slot_time_ist} ranks highest ({best_slot.suitability_score}/100, status: {best_slot.recommendation_status}) "
                    f"compared to {lowest_slot.slot_time_ist} ({lowest_slot.suitability_score}/100, status: {lowest_slot.recommendation_status}) due to "
                    f"{', '.join(lowest_slot.score_breakdown.heuristic_notes or ['environmental differences'])}."
                )

        return CandidateRankingResult(
            activity_id=activity_id,
            total_venues_evaluated=len(candidates),
            total_slots_evaluated=len(all_slots),
            has_suitable_option=has_suitable,
            no_suitable_option_reason=no_suitable_reason,
            best_recommendation=best_slot,
            top_candidates=all_slots[:top_k],
            temporal_comparison_note=temporal_note,
        )
