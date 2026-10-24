
"""Candidate place-and-time suitability ranking.

Evaluates outdoor venue-hour combinations using environmental
forecasts and qualitative geographic context.

Guarantees:
- Forecasts and requested slots are matched by UTC hour.
- Absolute recommendation status takes priority over score.
- Destination-local time is used for daylight evaluation.
- Existing response fields remain available.
- Missing observations are not treated as measured hazards.
- Weather/AQI are anchor-level estimates unless supplied
  separately for individual venues.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from src.common.coordinates import LatLon

from src.common.schemas import (
    HourlyAQIObservation,
    HourlyWeatherObservation,
    PersonalHealthProfile,
    PlaceCandidate,
)

from src.common.temporal import IST

from src.scoring.evaluator import (
    RecommendationStatus,
    SuitabilityEvaluator,
    SuitabilityScoreBreakdown,
)

from src.spatial_analysis.wind_dispersion import (
    GeographicContextReport,
    analyze_geographic_context,
)


class PlaceTimeSlotRecommendation(BaseModel):
    """A scored venue at a specific forecast hour."""

    venue_id: str
    venue_name: str
    place_type: str
    coordinate: LatLon
    distance_km: Optional[float] = None

    # Legacy field retained for frontend compatibility.
    slot_time_ist: str

    # Additional unambiguous timestamps.
    slot_time_utc: Optional[str] = None
    slot_time_local: Optional[str] = None
    timezone_name: Optional[str] = None

    suitability_score: float
    rating_category: str

    recommendation_status: str = Field(
        ...,
        description=(
            "'recommended', 'caution', or 'not_recommended'"
        ),
    )

    is_favorable: bool

    score_breakdown: SuitabilityScoreBreakdown
    geographic_context: GeographicContextReport

    weather_summary: Dict[str, Any]
    aqi_summary: Dict[str, Any]


class CandidateRankingResult(BaseModel):
    """Aggregated ranking results for evaluated venue-hours."""

    activity_id: str
    total_venues_evaluated: int
    total_slots_evaluated: int

    has_suitable_option: bool

    no_suitable_option_reason: Optional[str] = None

    best_recommendation: Optional[
        PlaceTimeSlotRecommendation
    ] = None

    top_candidates: List[
        PlaceTimeSlotRecommendation
    ] = Field(default_factory=list)

    temporal_comparison_note: Optional[str] = None


class CandidateRanker:
    """Evaluate and rank venue-hour combinations."""

    STATUS_PRIORITY = {
        RecommendationStatus.RECOMMENDED.value: 2,
        RecommendationStatus.CAUTION.value: 1,
        RecommendationStatus.NOT_RECOMMENDED.value: 0,
    }

    def __init__(
        self,
        evaluator: Optional[SuitabilityEvaluator] = None,
    ):
        self.evaluator = evaluator or SuitabilityEvaluator()

    @staticmethod
    def _utc_hour_key(
        dt: datetime,
    ) -> datetime:
        """Normalize a timezone-aware timestamp to its UTC hour."""

        if (
            dt.tzinfo is None
            or dt.utcoffset() is None
        ):
            raise ValueError(
                "Forecast and slot timestamps must "
                "be timezone-aware."
            )

        return dt.astimezone(
            timezone.utc
        ).replace(
            minute=0,
            second=0,
            microsecond=0,
        )

    @staticmethod
    def _ist_label(
        dt: datetime,
    ) -> str:
        """Return an accurate IST timestamp for legacy clients."""

        return dt.astimezone(IST).strftime(
            "%Y-%m-%d %H:%M IST"
        )

    @staticmethod
    def _utc_label(
        dt: datetime,
    ) -> str:
        return dt.astimezone(
            timezone.utc
        ).isoformat()

    @staticmethod
    def _local_label(
        dt: datetime,
        local_timezone: Optional[str],
    ) -> Optional[str]:
        if not local_timezone:
            return None

        return dt.astimezone(
            ZoneInfo(local_timezone)
        ).isoformat()

    @staticmethod
    def _empty_result(
        activity_id: str,
        venue_count: int,
        reason: str,
        temporal_note: str,
    ) -> CandidateRankingResult:
        return CandidateRankingResult(
            activity_id=activity_id,
            total_venues_evaluated=venue_count,
            total_slots_evaluated=0,
            has_suitable_option=False,
            no_suitable_option_reason=reason,
            best_recommendation=None,
            top_candidates=[],
            temporal_comparison_note=temporal_note,
        )

    @classmethod
    def _ranking_key(
        cls,
        recommendation: PlaceTimeSlotRecommendation,
    ):
        """Higher status and score rank first; nearer venues win ties."""

        status_priority = cls.STATUS_PRIORITY.get(
            recommendation.recommendation_status,
            0,
        )

        distance = (
            recommendation.distance_km
            if recommendation.distance_km is not None
            else float("inf")
        )

        return (
            -status_priority,
            -recommendation.suitability_score,
            distance,
            recommendation.slot_time_utc or "",
            recommendation.venue_id,
        )

    def rank_candidates(
        self,
        activity_id: str,
        candidates: List[PlaceCandidate],
        weather_forecasts: List[HourlyWeatherObservation],
        aqi_forecasts: List[HourlyAQIObservation],
        target_slots: List[datetime],
        health_profile: Optional[
            PersonalHealthProfile
        ] = None,
        top_k: int = 5,
        local_timezone: Optional[str] = None,
    ) -> CandidateRankingResult:
        """Rank venue-hour combinations for an outdoor activity."""

        venue_count = len(candidates)

        if not candidates:
            return self._empty_result(
                activity_id,
                venue_count,
                "No candidate venues were available for evaluation.",
                "No venues available.",
            )

        if not target_slots:
            return self._empty_result(
                activity_id,
                venue_count,
                "No valid activity hours were available for evaluation.",
                "No requested hours available.",
            )

        if not weather_forecasts:
            return self._empty_result(
                activity_id,
                venue_count,
                "Weather forecast data is unavailable. "
                "Suitability cannot be determined.",
                "Weather data unavailable.",
            )

        # Validate timezone before processing any recommendations.
        if local_timezone is not None:
            try:
                ZoneInfo(local_timezone)
            except Exception as exc:
                raise ValueError(
                    f"Invalid destination timezone: "
                    f"{local_timezone}"
                ) from exc

        # Match observations by absolute UTC hour.
        weather_map: Dict[
            datetime,
            HourlyWeatherObservation,
        ] = {
            self._utc_hour_key(obs.time): obs
            for obs in weather_forecasts
        }

        aqi_map: Dict[
            datetime,
            HourlyAQIObservation,
        ] = {
            self._utc_hour_key(obs.time): obs
            for obs in aqi_forecasts
        }

        all_slots: List[
            PlaceTimeSlotRecommendation
        ] = []

        # Deduplicate target hours after UTC normalization.
        unique_slots: Dict[datetime, datetime] = {}

        for slot_dt in target_slots:
            key = self._utc_hour_key(slot_dt)
            unique_slots.setdefault(key, slot_dt)

        for candidate in candidates:
            for key, slot_dt in unique_slots.items():

                weather_obs = weather_map.get(key)

                if weather_obs is None:
                    continue

                aqi_obs = aqi_map.get(key)

                geo_report = analyze_geographic_context(
                    candidate=candidate,
                    wind_direction_deg=(
                        weather_obs.wind_direction_10m
                    ),
                    wind_speed_kmh=(
                        weather_obs.wind_speed_10m
                    ),
                )

                breakdown = self.evaluator.evaluate(
                    activity_id=activity_id,
                    weather=weather_obs,
                    aqi=aqi_obs,
                    geo_report=geo_report,
                    health_profile=health_profile,
                    local_timezone=local_timezone,
                )

                recommendation = PlaceTimeSlotRecommendation(
                    venue_id=candidate.id,
                    venue_name=candidate.name,
                    place_type=candidate.place_type,
                    coordinate=candidate.coordinate,
                    distance_km=(
                        candidate.distance_from_anchor_km
                    ),
                    slot_time_ist=self._ist_label(
                        slot_dt
                    ),
                    slot_time_utc=self._utc_label(
                        slot_dt
                    ),
                    slot_time_local=self._local_label(
                        slot_dt,
                        local_timezone,
                    ),
                    timezone_name=local_timezone,
                    suitability_score=(
                        breakdown.overall_score
                    ),
                    rating_category=(
                        breakdown.rating_category
                    ),
                    recommendation_status=(
                        breakdown.recommendation_status
                    ),
                    is_favorable=(
                        breakdown.is_favorable
                    ),
                    score_breakdown=breakdown,
                    geographic_context=geo_report,
                    weather_summary={
                        "temperature_c": (
                            weather_obs.temperature_2m
                        ),
                        "humidity_percent": (
                            weather_obs.relative_humidity_2m
                        ),
                        "rain_probability_percent": (
                            weather_obs.precipitation_probability
                        ),
                        "precipitation_mm": (
                            weather_obs.precipitation
                        ),
                        "wind_speed_kmh": (
                            weather_obs.wind_speed_10m
                        ),
                        "wind_direction_deg": (
                            weather_obs.wind_direction_10m
                        ),
                        "uv_index": (
                            weather_obs.uv_index
                        ),
                    },
                    aqi_summary={
                        "pm2_5": (
                            aqi_obs.pm2_5
                            if aqi_obs is not None
                            else None
                        ),
                        "pm10": (
                            aqi_obs.pm10
                            if aqi_obs is not None
                            else None
                        ),
                        "us_aqi": (
                            aqi_obs.us_aqi
                            if aqi_obs is not None
                            else None
                        ),
                    },
                )

                all_slots.append(
                    recommendation
                )

        if not all_slots:
            return self._empty_result(
                activity_id,
                venue_count,
                "No weather observations matched the "
                "requested evaluation hours. "
                "Suitability cannot be determined.",
                "No matching forecast hours.",
            )

        # Absolute status first, then score and proximity.
        all_slots.sort(
            key=self._ranking_key
        )

        best_slot = all_slots[0]

        has_suitable = any(
            slot.recommendation_status in {
                RecommendationStatus.RECOMMENDED.value,
                RecommendationStatus.CAUTION.value,
            }
            for slot in all_slots
        )

        no_suitable_reason = None

        if not has_suitable:
            reasons = set()

            for slot in all_slots:
                reasons.update(
                    slot.score_breakdown.unfavorable_reasons
                )

            reason_text = (
                "; ".join(sorted(reasons))
                if reasons
                else (
                    "environmental conditions exceed "
                    "the configured suitability thresholds"
                )
            )

            no_suitable_reason = (
                f"No suitable outdoor window found for "
                f"{activity_id} across {len(all_slots)} "
                f"evaluated venue-hour combinations. "
                f"Primary factors: {reason_text}. "
                "Consider postponing the activity or "
                "choosing an indoor alternative."
            )

        temporal_note = None

        if len(unique_slots) > 1:
            same_venue_slots = [
                slot
                for slot in all_slots
                if slot.venue_id == best_slot.venue_id
            ]

            if len(same_venue_slots) > 1:
                lowest_slot = min(
                    same_venue_slots,
                    key=lambda slot: (
                        self.STATUS_PRIORITY.get(
                            slot.recommendation_status,
                            0,
                        ),
                        slot.suitability_score,
                    ),
                )

                best_time = (
                    best_slot.slot_time_local
                    or best_slot.slot_time_ist
                )

                lowest_time = (
                    lowest_slot.slot_time_local
                    or lowest_slot.slot_time_ist
                )

                temporal_note = (
                    f"At {best_slot.venue_name}, "
                    f"{best_time} ranks highest "
                    f"({best_slot.suitability_score}/100, "
                    f"status: {best_slot.recommendation_status}) "
                    f"compared with {lowest_time} "
                    f"({lowest_slot.suitability_score}/100, "
                    f"status: {lowest_slot.recommendation_status})."
                )

        return CandidateRankingResult(
            activity_id=activity_id,
            total_venues_evaluated=venue_count,
            total_slots_evaluated=len(all_slots),
            has_suitable_option=has_suitable,
            no_suitable_option_reason=no_suitable_reason,
            best_recommendation=best_slot,
            top_candidates=all_slots[
                :max(0, top_k)
            ],
            temporal_comparison_note=temporal_note,
        )
