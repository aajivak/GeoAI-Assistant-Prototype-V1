"""Multi-criteria suitability scoring engine using normalized weighted averages and explicit penalties.

Strict Invariants:
1. Clear separation between relative ranking and absolute recommendation status (recommended, caution, not_recommended).
2. Scores remain strictly bounded between 0.0 and 100.0.
3. Conservative, activity-aware air quality guidance (strenuous cardio like running is penalized heavily under elevated PM2.5/AQI).
4. Missing live data fields remain Unknown (None). Weights are re-normalized over available metrics.
5. Explicit penalties are subtracted and documented.
6. Disclaimer included: Not a medically certified safety guarantee.
"""
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from config.activity_definitions import SUPPORTED_ACTIVITIES, ActivityDefinition
from src.common.schemas import HourlyAQIObservation, HourlyWeatherObservation, PersonalHealthProfile
from src.scoring.activity_rules import ACTIVITY_WEIGHTS, ActivityCriteriaWeights
from src.spatial_analysis.wind_dispersion import GeographicContextReport


DISCLAIMER_TEXT = (
    "Disclaimer: Suitability scores and recommendations are heuristic estimates based on open "
    "forecast data and geographic indicators. They do not constitute certified medical, health, "
    "or safety advice."
)


class RecommendationStatus(str, Enum):
    """Absolute safety/suitability status for an outdoor activity slot."""
    RECOMMENDED = "recommended"
    CAUTION = "caution"
    NOT_RECOMMENDED = "not_recommended"


class SuitabilityScoreBreakdown(BaseModel):
    """Detailed transparent breakdown of suitability calculation."""
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Final score [0-100]")
    rating_category: str = Field(..., description="e.g. 'Highly Suitable', 'Moderate', 'Poor'")
    recommendation_status: str = Field(
        ...,
        description="Absolute recommendation status: 'recommended', 'caution', or 'not_recommended'"
    )
    is_favorable: bool = Field(..., description="True if recommended, False if caution or not_recommended")
    base_weighted_score: float
    sub_scores: Dict[str, Optional[float]] = Field(
        ...,
        description="Sub-scores [0-100] for temperature, precipitation, wind, aqi, daylight. None indicates missing live data."
    )
    penalties_applied: Dict[str, float] = Field(default_factory=dict)
    total_penalty_deduction: float = 0.0
    uncertainty_warnings: List[str] = Field(default_factory=list)
    heuristic_notes: List[str] = Field(default_factory=list)
    unfavorable_reasons: List[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER_TEXT


class SuitabilityEvaluator:
    """Evaluates suitability and absolute safety status for an outdoor activity."""

    # Strenuous cardio activities with elevated respiratory intake
    HIGH_CARDIO_ACTIVITIES = {"running", "football", "cycling"}

    def evaluate(
        self,
        activity_id: str,
        weather: HourlyWeatherObservation,
        aqi: Optional[HourlyAQIObservation] = None,
        geo_report: Optional[GeographicContextReport] = None,
        health_profile: Optional[PersonalHealthProfile] = None,
    ) -> SuitabilityScoreBreakdown:
        """Calculate normalized weighted-average score and apply explainable penalties and absolute safety status."""
        activity_key = activity_id.lower()
        activity = SUPPORTED_ACTIVITIES.get(activity_key, SUPPORTED_ACTIVITIES["running"])
        weights: ActivityCriteriaWeights = ACTIVITY_WEIGHTS.get(activity_key, ACTIVITY_WEIGHTS["running"])

        sub_scores: Dict[str, Optional[float]] = {}
        uncertainty_warnings: List[str] = []
        heuristic_notes: List[str] = []
        unfavorable_reasons: List[str] = []

        # 1. Temperature Sub-Score
        if weather.temperature_2m is not None:
            sub_scores["temperature"] = self._score_temperature(
                weather.temperature_2m,
                activity.ideal_temp_min,
                activity.ideal_temp_max,
            )
        else:
            sub_scores["temperature"] = None
            uncertainty_warnings.append("Missing live temperature measurement; temperature criterion omitted.")

        # 2. Precipitation Sub-Score
        if weather.precipitation_probability is not None or weather.precipitation is not None:
            sub_scores["precipitation"] = self._score_precipitation(
                rain_prob=weather.precipitation_probability,
                rain_mm=weather.precipitation,
                rain_tolerant=activity.rain_tolerant,
            )
        else:
            sub_scores["precipitation"] = None
            uncertainty_warnings.append("Missing live precipitation measurement; precipitation criterion omitted.")

        # 3. Wind Sub-Score
        if weather.wind_speed_10m is not None:
            sub_scores["wind"] = self._score_wind(
                wind_speed_kmh=weather.wind_speed_10m,
                wind_sensitive=activity.wind_sensitive,
            )
        else:
            sub_scores["wind"] = None
            uncertainty_warnings.append("Missing live wind speed measurement; wind criterion omitted.")

        # 4. Air Quality Sub-Score (Conservative & Activity-Aware)
        pm25 = aqi.pm2_5 if (aqi and aqi.pm2_5 is not None) else None
        us_aqi = aqi.us_aqi if (aqi and aqi.us_aqi is not None) else None

        if pm25 is not None:
            sub_scores["aqi"] = self._score_pm25_activity_aware(pm25, is_high_cardio=(activity_key in self.HIGH_CARDIO_ACTIVITIES))
        elif us_aqi is not None:
            sub_scores["aqi"] = self._score_us_aqi(us_aqi)
        else:
            sub_scores["aqi"] = None
            uncertainty_warnings.append("Missing live air-quality (PM2.5) measurement; AQI criterion omitted.")

        # 5. Daylight Sub-Score
        hour = weather.time.hour
        is_daylight = 6 <= hour <= 18
        if activity.daylight_required:
            sub_scores["daylight"] = 100.0 if is_daylight else 20.0
            if not is_daylight:
                heuristic_notes.append("Activity typically requires natural daylight; night hours score lower.")
        else:
            sub_scores["daylight"] = 100.0

        # --- Compute Normalized Weighted Average over Available Metrics ---
        metric_weights = {
            "temperature": weights.weight_temperature,
            "precipitation": weights.weight_precipitation,
            "wind": weights.weight_wind,
            "aqi": weights.weight_aqi,
            "daylight": weights.weight_daylight,
        }

        active_weight_sum = 0.0
        weighted_score_accum = 0.0

        for metric, score_val in sub_scores.items():
            if score_val is not None:
                w = metric_weights[metric]
                active_weight_sum += w
                weighted_score_accum += w * score_val

        if active_weight_sum > 0.0:
            base_score = weighted_score_accum / active_weight_sum
        else:
            base_score = 0.0
            uncertainty_warnings.append("No environmental observations available; unable to compute base score.")

        # --- Explicit Penalties & Sensitivities ---
        penalties: Dict[str, float] = {}

        # Air Quality High-Exertion Penalties (Correction for running in unhealthy air)
        if activity_key in self.HIGH_CARDIO_ACTIVITIES and pm25 is not None:
            if pm25 > 55.0 or (us_aqi and us_aqi > 150):
                penalties["high_cardio_unhealthy_aqi"] = 35.0
                unfavorable_reasons.append(f"Air quality is unhealthy for intense cardio (PM2.5: {pm25:.1f} µg/m³).")
            elif pm25 > 35.0 or (us_aqi and us_aqi > 100):
                penalties["high_cardio_moderate_aqi"] = 18.0
                heuristic_notes.append(f"Moderate PM2.5 ({pm25:.1f} µg/m³); prolonged intense running may cause throat or lung irritation.")

        # Geographic Context Penalty
        if geo_report and geo_report.suggested_qualitative_penalty > 0.0:
            penalties["geographic_downwind_exposure"] = geo_report.suggested_qualitative_penalty
            heuristic_notes.append(f"Geographic exposure: {geo_report.summary_advisory}")

        # Personal Health Profile Penalties
        if health_profile:
            self._apply_health_penalties(
                health_profile=health_profile,
                weather=weather,
                aqi=aqi,
                geo_report=geo_report,
                penalties=penalties,
                heuristic_notes=heuristic_notes,
                unfavorable_reasons=unfavorable_reasons,
            )

        total_penalty = sum(penalties.values())
        final_score = max(0.0, min(100.0, base_score - total_penalty))

        # --- Absolute Safety/Recommendation Status Determination ---
        rec_status = self._determine_absolute_status(
            activity_key=activity_key,
            final_score=final_score,
            pm25=pm25,
            us_aqi=us_aqi,
            weather=weather,
            health_profile=health_profile,
            unfavorable_reasons=unfavorable_reasons,
        )

        is_favorable = (rec_status == RecommendationStatus.RECOMMENDED)

        # Rating categorization
        if final_score >= 80.0:
            category = "Highly Suitable"
        elif final_score >= 60.0:
            category = "Suitable with Minor Considerations"
        elif final_score >= 40.0:
            category = "Suboptimal / Marginal"
        else:
            category = "Not Recommended / Unfavorable"

        return SuitabilityScoreBreakdown(
            overall_score=round(final_score, 1),
            rating_category=category,
            recommendation_status=rec_status.value,
            is_favorable=is_favorable,
            base_weighted_score=round(base_score, 1),
            sub_scores={k: round(v, 1) if v is not None else None for k, v in sub_scores.items()},
            penalties_applied={k: round(v, 1) for k, v in penalties.items()},
            total_penalty_deduction=round(total_penalty, 1),
            uncertainty_warnings=uncertainty_warnings,
            heuristic_notes=heuristic_notes,
            unfavorable_reasons=unfavorable_reasons,
            disclaimer=DISCLAIMER_TEXT,
        )

    def _determine_absolute_status(
        self,
        activity_key: str,
        final_score: float,
        pm25: Optional[float],
        us_aqi: Optional[int],
        weather: HourlyWeatherObservation,
        health_profile: Optional[PersonalHealthProfile],
        unfavorable_reasons: List[str],
    ) -> RecommendationStatus:
        """Determine whether the slot is Recommended, Caution, or Not Recommended."""
        # 1. Hard exclusions for asthma
        if health_profile and health_profile.has_asthma:
            if (pm25 is not None and pm25 > 45.0) or (us_aqi is not None and us_aqi > 120):
                return RecommendationStatus.NOT_RECOMMENDED
            if (pm25 is not None and pm25 > 25.0) or (us_aqi is not None and us_aqi > 80):
                return RecommendationStatus.CAUTION

        # 2. Hard exclusions for high-cardio activities (running, football, cycling)
        if activity_key in self.HIGH_CARDIO_ACTIVITIES:
            if (pm25 is not None and pm25 > 55.0) or (us_aqi is not None and us_aqi > 150):
                return RecommendationStatus.NOT_RECOMMENDED
            if (pm25 is not None and pm25 > 35.0) or (us_aqi is not None and us_aqi > 100):
                return RecommendationStatus.CAUTION

        # 3. Weather hard exclusions
        if weather.precipitation is not None and weather.precipitation > 2.0:
            unfavorable_reasons.append("Heavy precipitation detected.")
            return RecommendationStatus.NOT_RECOMMENDED
        if weather.temperature_2m is not None and weather.temperature_2m > 36.0:
            unfavorable_reasons.append("Extreme ambient heat (>36°C).")
            return RecommendationStatus.NOT_RECOMMENDED

        # 4. Score-based thresholds
        if final_score < 45.0:
            return RecommendationStatus.NOT_RECOMMENDED
        elif final_score < 70.0:
            return RecommendationStatus.CAUTION
        else:
            return RecommendationStatus.RECOMMENDED

    def _score_temperature(self, temp: float, ideal_min: float, ideal_max: float) -> float:
        """Linear decay outside ideal temperature range."""
        if ideal_min <= temp <= ideal_max:
            return 100.0
        elif temp < ideal_min:
            delta = ideal_min - temp
            return max(0.0, 100.0 - (delta * 5.0))
        else:
            delta = temp - ideal_max
            return max(0.0, 100.0 - (delta * 6.0))

    def _score_precipitation(
        self,
        rain_prob: Optional[float],
        rain_mm: Optional[float],
        rain_tolerant: bool,
    ) -> float:
        """Score precipitation risk."""
        prob = rain_prob or 0.0
        mm = rain_mm or 0.0

        if mm > 2.0 or prob >= 80.0:
            return 20.0 if rain_tolerant else 0.0
        elif mm > 0.2 or prob >= 40.0:
            return 60.0 if rain_tolerant else 25.0
        elif prob > 15.0:
            return 85.0 if rain_tolerant else 65.0
        else:
            return 100.0

    def _score_wind(self, wind_speed_kmh: float, wind_sensitive: bool) -> float:
        """Score wind speed comfort."""
        if wind_speed_kmh <= 12.0:
            return 100.0
        elif wind_speed_kmh <= 20.0:
            return 85.0 if wind_sensitive else 95.0
        elif wind_speed_kmh <= 30.0:
            return 50.0 if wind_sensitive else 80.0
        else:
            return max(0.0, 30.0 if wind_sensitive else 55.0)

    def _score_pm25_activity_aware(self, pm25: float, is_high_cardio: bool) -> float:
        """Score particulate matter PM2.5 with stricter penalties for high-cardio activities."""
        if is_high_cardio:
            # Strenuous cardio: steep drop when air quality degrades
            if pm25 <= 15.0:
                return 100.0
            elif pm25 <= 35.0:
                return 100.0 - ((pm25 - 15.0) / 20.0) * 30.0  # 100 -> 70
            elif pm25 <= 55.0:
                return 70.0 - ((pm25 - 35.0) / 20.0) * 35.0   # 70 -> 35
            else:
                return max(0.0, 35.0 - ((pm25 - 55.0) / 30.0) * 35.0)
        else:
            if pm25 <= 15.0:
                return 100.0
            elif pm25 <= 35.0:
                return 100.0 - ((pm25 - 15.0) / 20.0) * 20.0  # 100 -> 80
            elif pm25 <= 55.0:
                return 80.0 - ((pm25 - 35.0) / 20.0) * 25.0   # 80 -> 55
            elif pm25 <= 75.0:
                return 55.0 - ((pm25 - 55.0) / 20.0) * 20.0   # 55 -> 35
            else:
                return max(5.0, 35.0 - ((pm25 - 75.0) / 40.0) * 25.0)

    def _score_us_aqi(self, aqi: int) -> float:
        """Fallback scoring using US AQI index (0-500)."""
        if aqi <= 50:
            return 100.0
        elif aqi <= 100:
            return 75.0
        elif aqi <= 150:
            return 45.0
        elif aqi <= 200:
            return 20.0
        else:
            return 5.0

    def _apply_health_penalties(
        self,
        health_profile: PersonalHealthProfile,
        weather: HourlyWeatherObservation,
        aqi: Optional[HourlyAQIObservation],
        geo_report: Optional[GeographicContextReport],
        penalties: Dict[str, float],
        heuristic_notes: List[str],
        unfavorable_reasons: List[str],
    ) -> None:
        """Apply explainable, conservative health profile sensitivity penalties."""
        # 1. Asthma Sensitivity
        if health_profile.has_asthma:
            pm25 = aqi.pm2_5 if (aqi and aqi.pm2_5 is not None) else None
            if pm25 is not None:
                if pm25 > 50.0:
                    penalties["asthma_high_pm25"] = 40.0
                    unfavorable_reasons.append(f"Elevated PM2.5 ({pm25:.1f} µg/m³) triggers asthma health exclusion.")
                elif pm25 > 30.0:
                    penalties["asthma_moderate_pm25"] = 25.0
                    heuristic_notes.append(f"Asthma advisory: PM2.5 is moderate ({pm25:.1f} µg/m³). Carry quick-relief inhaler.")
            if geo_report and (geo_report.is_downwind_of_road or geo_report.is_downwind_of_industry):
                penalties["asthma_downwind_plume"] = 15.0
                heuristic_notes.append("Asthma caution: Venue is downwind of traffic/industrial corridor.")

            if weather.temperature_2m is not None and weather.relative_humidity_2m is not None:
                if weather.temperature_2m < 15.0 and weather.relative_humidity_2m > 80.0:
                    penalties["asthma_cold_damp_air"] = 15.0
                    heuristic_notes.append("Asthma advisory: Cold, damp morning air may trigger bronchospasm.")

        # 2. Older Adults / Senior Sensitivity
        if health_profile.is_senior:
            if weather.temperature_2m is not None and weather.temperature_2m > 32.0:
                penalties["senior_heat_stress"] = 30.0
                unfavorable_reasons.append("High ambient heat (>32°C) presents heat strain risk for older adults.")
            if weather.precipitation_probability is not None and weather.precipitation_probability > 35.0:
                penalties["senior_slip_hazard"] = 20.0
                heuristic_notes.append("Senior caution: Rain probability increases risk of slick or slippery pathways.")

        # 3. Children Sensitivity
        if health_profile.is_child:
            if weather.uv_index is not None and weather.uv_index >= 6.0:
                penalties["child_high_uv"] = 25.0
                heuristic_notes.append(f"Child caution: Elevated UV Index ({weather.uv_index:.1f}). Sunscreen/hat essential.")
            if geo_report and geo_report.is_downwind_of_road:
                penalties["child_traffic_exposure"] = 15.0
                heuristic_notes.append("Child advisory: Proximity downwind of busy road.")

        # 4. Pregnancy Sensitivity
        if health_profile.is_pregnant:
            if weather.temperature_2m is not None and weather.temperature_2m > 28.0:
                penalties["pregnancy_thermal_stress"] = 25.0
                heuristic_notes.append("Pregnancy advisory: Warm conditions (>28°C); ensure shade and frequent hydration.")
            if aqi and aqi.pm2_5 is not None and aqi.pm2_5 > 45.0:
                penalties["pregnancy_air_quality"] = 20.0
                unfavorable_reasons.append("Air quality is suboptimal for pregnancy exertion.")
