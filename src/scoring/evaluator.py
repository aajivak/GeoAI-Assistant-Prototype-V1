
"""Multi-criteria outdoor activity suitability scoring engine.

Invariants:
1. Scores are bounded between 0 and 100.
2. Relative scores never override absolute exclusions.
3. Missing measurements remain unknown, never invented.
4. Severe environmental conditions are checked before caution rules.
5. Geographic proximity indicators are qualitative heuristics.
6. The system does not provide certified medical or safety advice.
"""

from datetime import timezone
from enum import Enum
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from config.activity_definitions import SUPPORTED_ACTIVITIES
from src.common.schemas import (
    HourlyAQIObservation,
    HourlyWeatherObservation,
    PersonalHealthProfile,
)
from src.scoring.activity_rules import ACTIVITY_WEIGHTS, ActivityCriteriaWeights
from src.spatial_analysis.wind_dispersion import GeographicContextReport


DISCLAIMER_TEXT = (
    "Disclaimer: Suitability scores and recommendations are heuristic "
    "estimates based on open forecast data and geographic indicators. "
    "They do not constitute certified medical, health, or safety advice."
)


class RecommendationStatus(str, Enum):
    RECOMMENDED = "recommended"
    CAUTION = "caution"
    NOT_RECOMMENDED = "not_recommended"


class SuitabilityScoreBreakdown(BaseModel):
    overall_score: float = Field(..., ge=0, le=100)
    rating_category: str
    recommendation_status: str
    is_favorable: bool
    base_weighted_score: float
    sub_scores: Dict[str, Optional[float]]
    penalties_applied: Dict[str, float] = Field(default_factory=dict)
    total_penalty_deduction: float = 0.0
    uncertainty_warnings: List[str] = Field(default_factory=list)
    heuristic_notes: List[str] = Field(default_factory=list)
    unfavorable_reasons: List[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER_TEXT


class SuitabilityEvaluator:
    """Evaluate environmental suitability for an outdoor activity."""

    HIGH_CARDIO_ACTIVITIES = {
        "running",
        "football",
        "cycling",
    }

    def evaluate(
        self,
        activity_id: str,
        weather: HourlyWeatherObservation,
        aqi: Optional[HourlyAQIObservation] = None,
        geo_report: Optional[GeographicContextReport] = None,
        health_profile: Optional[PersonalHealthProfile] = None,
        local_timezone: Optional[str] = None,
    ) -> SuitabilityScoreBreakdown:

        activity_key = activity_id.lower()

        activity = SUPPORTED_ACTIVITIES.get(
            activity_key,
            SUPPORTED_ACTIVITIES["running"],
        )

        weights: ActivityCriteriaWeights = ACTIVITY_WEIGHTS.get(
            activity_key,
            ACTIVITY_WEIGHTS["running"],
        )

        sub_scores: Dict[str, Optional[float]] = {}
        warnings: List[str] = []
        notes: List[str] = []
        unfavorable: List[str] = []
        penalties: Dict[str, float] = {}

        pm25 = aqi.pm2_5 if aqi else None
        us_aqi = aqi.us_aqi if aqi else None

        is_high_cardio = (
            activity_key in self.HIGH_CARDIO_ACTIVITIES
        )

        # -------------------------------------------
        # 1. Temperature
        # -------------------------------------------

        if weather.temperature_2m is not None:
            sub_scores["temperature"] = self._score_temperature(
                weather.temperature_2m,
                activity.ideal_temp_min,
                activity.ideal_temp_max,
            )
        else:
            sub_scores["temperature"] = None
            warnings.append("Temperature forecast unavailable.")

        # -------------------------------------------
        # 2. Precipitation
        # -------------------------------------------

        if (
            weather.precipitation_probability is not None
            or weather.precipitation is not None
        ):
            sub_scores["precipitation"] = self._score_precipitation(
                weather.precipitation_probability,
                weather.precipitation,
                activity.rain_tolerant,
            )
        else:
            sub_scores["precipitation"] = None
            warnings.append("Precipitation forecast unavailable.")

        # -------------------------------------------
        # 3. Wind
        # -------------------------------------------

        if weather.wind_speed_10m is not None:
            sub_scores["wind"] = self._score_wind(
                weather.wind_speed_10m,
                activity.wind_sensitive,
            )
        else:
            sub_scores["wind"] = None
            warnings.append("Wind speed forecast unavailable.")

        # -------------------------------------------
        # 4. Air quality
        # -------------------------------------------

        if pm25 is not None:
            sub_scores["aqi"] = self._score_pm25_activity_aware(
                pm25,
                is_high_cardio,
            )
        elif us_aqi is not None:
            sub_scores["aqi"] = self._score_us_aqi(
                us_aqi
            )
        else:
            sub_scores["aqi"] = None
            warnings.append(
                "Air-quality forecast unavailable; "
                "respiratory exposure cannot be assessed."
            )

        # -------------------------------------------
        # 5. Daylight
        # -------------------------------------------

        local_hour = None

        if local_timezone:
            try:
                local_time = weather.time.astimezone(
                    ZoneInfo(local_timezone)
                )
                local_hour = local_time.hour
            except Exception:
                warnings.append(
                    "Destination timezone could not be applied."
                )

        if local_hour is None:
            sub_scores["daylight"] = None
            warnings.append(
                "Daylight cannot be estimated without "
                "a destination timezone."
            )
        else:
            # Approximation only; not actual sunrise/sunset.
            is_daylight = 6 <= local_hour < 18

            if activity.daylight_required:
                sub_scores["daylight"] = (
                    100.0 if is_daylight else 20.0
                )
                if not is_daylight:
                    notes.append(
                        "Estimated nighttime conditions; "
                        "this activity usually requires daylight."
                    )
            else:
                sub_scores["daylight"] = 100.0

        # -------------------------------------------
        # 6. Weighted score
        # -------------------------------------------

        metric_weights = {
            "temperature": weights.weight_temperature,
            "precipitation": weights.weight_precipitation,
            "wind": weights.weight_wind,
            "aqi": weights.weight_aqi,
            "daylight": weights.weight_daylight,
        }

        weighted_total = 0.0
        active_weight = 0.0

        for metric, score in sub_scores.items():
            if score is not None:
                weight = metric_weights[metric]
                weighted_total += weight * score
                active_weight += weight

        base_score = (
            weighted_total / active_weight
            if active_weight > 0
            else 0.0
        )

        # -------------------------------------------
        # 7. High-cardio air-quality penalties
        # -------------------------------------------

        if is_high_cardio:
            severe_air = (
                (pm25 is not None and pm25 > 55)
                or (us_aqi is not None and us_aqi > 150)
            )

            moderate_air = (
                (pm25 is not None and pm25 > 35)
                or (us_aqi is not None and us_aqi > 100)
            )

            if severe_air:
                penalties["high_cardio_unhealthy_aqi"] = 35.0
                unfavorable.append(
                    "Air quality is unfavorable for intense cardio."
                )
            elif moderate_air:
                penalties["high_cardio_moderate_aqi"] = 18.0
                notes.append(
                    "Elevated air pollution may increase "
                    "respiratory exposure during intense activity."
                )

        # -------------------------------------------
        # 8. Geographic exposure
        # -------------------------------------------

        if (
            geo_report
            and geo_report.suggested_qualitative_penalty > 0
        ):
            penalties["geographic_downwind_exposure"] = (
                geo_report.suggested_qualitative_penalty
            )

            notes.append(
                f"Geographic exposure: "
                f"{geo_report.summary_advisory}"
            )

        # -------------------------------------------
        # 9. Personal health sensitivities
        # -------------------------------------------

        if health_profile:
            self._apply_health_penalties(
                health_profile,
                weather,
                aqi,
                geo_report,
                penalties,
                notes,
                unfavorable,
            )

        total_penalty = sum(penalties.values())

        final_score = max(
            0.0,
            min(100.0, base_score - total_penalty),
        )

        # -------------------------------------------
        # 10. Absolute status
        # -------------------------------------------

        status = self._determine_absolute_status(
            activity_key=activity_key,
            final_score=final_score,
            pm25=pm25,
            us_aqi=us_aqi,
            weather=weather,
            health_profile=health_profile,
            unfavorable_reasons=unfavorable,
        )

        # Missing essential observations should never
        # produce an unconditional recommendation.
        essential_missing = (
            sub_scores["temperature"] is None
            or sub_scores["precipitation"] is None
            or sub_scores["aqi"] is None
        )

        if (
            essential_missing
            and status == RecommendationStatus.RECOMMENDED
        ):
            status = RecommendationStatus.CAUTION
            warnings.append(
                "Recommendation downgraded because essential "
                "environmental observations are missing."
            )

        if (
            activity.daylight_required
            and sub_scores["daylight"] is None
            and status == RecommendationStatus.RECOMMENDED
        ):
            status = RecommendationStatus.CAUTION

        if status == RecommendationStatus.NOT_RECOMMENDED:
            category = "Not Recommended / Unfavorable"
        elif status == RecommendationStatus.CAUTION:
            category = "Suitable with Caution"
        elif final_score >= 80:
            category = "Highly Suitable"
        else:
            category = "Suitable"

        return SuitabilityScoreBreakdown(
            overall_score=round(final_score, 1),
            rating_category=category,
            recommendation_status=status.value,
            is_favorable=(
                status == RecommendationStatus.RECOMMENDED
            ),
            base_weighted_score=round(base_score, 1),
            sub_scores={
                key: round(value, 1)
                if value is not None else None
                for key, value in sub_scores.items()
            },
            penalties_applied={
                key: round(value, 1)
                for key, value in penalties.items()
            },
            total_penalty_deduction=round(
                total_penalty, 1
            ),
            uncertainty_warnings=warnings,
            heuristic_notes=notes,
            unfavorable_reasons=unfavorable,
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
        """Apply hard exclusions before caution conditions."""

        # Severe weather always takes precedence.
        if (
            weather.precipitation is not None
            and weather.precipitation > 2.0
        ):
            unfavorable_reasons.append(
                "Significant hourly precipitation detected."
            )
            return RecommendationStatus.NOT_RECOMMENDED

        if (
            weather.temperature_2m is not None
            and weather.temperature_2m > 36.0
        ):
            unfavorable_reasons.append(
                "Extreme ambient heat (>36°C)."
            )
            return RecommendationStatus.NOT_RECOMMENDED

        if health_profile and health_profile.has_asthma:
            if (
                (pm25 is not None and pm25 > 45)
                or (us_aqi is not None and us_aqi > 120)
            ):
                unfavorable_reasons.append(
                    "Air pollution exceeds the configured "
                    "asthma-sensitive threshold."
                )
                return RecommendationStatus.NOT_RECOMMENDED

        if activity_key in self.HIGH_CARDIO_ACTIVITIES:
            if (
                (pm25 is not None and pm25 > 55)
                or (us_aqi is not None and us_aqi > 150)
            ):
                unfavorable_reasons.append(
                    "Air pollution exceeds the configured "
                    "high-cardio threshold."
                )
                return RecommendationStatus.NOT_RECOMMENDED

        # Caution rules follow all hard exclusions.
        caution_required = False

        if health_profile and health_profile.has_asthma:
            if (
                (pm25 is not None and pm25 > 25)
                or (us_aqi is not None and us_aqi > 80)
            ):
                caution_required = True

        if activity_key in self.HIGH_CARDIO_ACTIVITIES:
            if (
                (pm25 is not None and pm25 > 35)
                or (us_aqi is not None and us_aqi > 100)
            ):
                caution_required = True

        if final_score < 45:
            return RecommendationStatus.NOT_RECOMMENDED

        if caution_required or final_score < 70:
            return RecommendationStatus.CAUTION

        return RecommendationStatus.RECOMMENDED

    def _score_temperature(
        self,
        temp: float,
        ideal_min: float,
        ideal_max: float,
    ) -> float:
        if ideal_min <= temp <= ideal_max:
            return 100.0
        if temp < ideal_min:
            return max(
                0.0,
                100.0 - (ideal_min - temp) * 5.0,
            )
        return max(
            0.0,
            100.0 - (temp - ideal_max) * 6.0,
        )

    def _score_precipitation(
        self,
        rain_prob: Optional[float],
        rain_mm: Optional[float],
        rain_tolerant: bool,
    ) -> float:
        prob = rain_prob if rain_prob is not None else 0.0
        mm = rain_mm if rain_mm is not None else 0.0

        if mm > 2.0 or prob >= 80:
            return 20.0 if rain_tolerant else 0.0
        if mm > 0.2 or prob >= 40:
            return 60.0 if rain_tolerant else 25.0
        if prob > 15:
            return 85.0 if rain_tolerant else 65.0
        return 100.0

    def _score_wind(
        self,
        wind_speed_kmh: float,
        wind_sensitive: bool,
    ) -> float:
        if wind_speed_kmh <= 12:
            return 100.0
        if wind_speed_kmh <= 20:
            return 85.0 if wind_sensitive else 95.0
        if wind_speed_kmh <= 30:
            return 50.0 if wind_sensitive else 80.0
        return 30.0 if wind_sensitive else 55.0

    def _score_pm25_activity_aware(
        self,
        pm25: float,
        is_high_cardio: bool,
    ) -> float:
        if is_high_cardio:
            if pm25 <= 15:
                return 100.0
            if pm25 <= 35:
                return 100 - ((pm25 - 15) / 20) * 30
            if pm25 <= 55:
                return 70 - ((pm25 - 35) / 20) * 35
            return max(
                0.0,
                35 - ((pm25 - 55) / 30) * 35,
            )

        if pm25 <= 15:
            return 100.0
        if pm25 <= 35:
            return 100 - ((pm25 - 15) / 20) * 20
        if pm25 <= 55:
            return 80 - ((pm25 - 35) / 20) * 25
        if pm25 <= 75:
            return 55 - ((pm25 - 55) / 20) * 20
        return max(
            5.0,
            35 - ((pm25 - 75) / 40) * 25,
        )

    def _score_us_aqi(self, aqi: int) -> float:
        if aqi <= 50:
            return 100.0
        if aqi <= 100:
            return 75.0
        if aqi <= 150:
            return 45.0
        if aqi <= 200:
            return 20.0
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
        """Apply explicit health-related score penalties."""

        pm25 = aqi.pm2_5 if aqi else None

        if health_profile.has_asthma:
            if pm25 is not None:
                if pm25 > 50:
                    penalties["asthma_high_pm25"] = 40.0
                    unfavorable_reasons.append(
                        "Elevated PM2.5 presents an "
                        "asthma-related exposure concern."
                    )
                elif pm25 > 30:
                    penalties["asthma_moderate_pm25"] = 25.0
                    heuristic_notes.append(
                        "Asthma sensitivity: elevated PM2.5."
                    )

            if geo_report and (
                geo_report.is_downwind_of_road
                or geo_report.is_downwind_of_industry
            ):
                penalties["asthma_downwind_plume"] = 15.0
                heuristic_notes.append(
                    "Potential traffic or industrial exposure."
                )

            if (
                weather.temperature_2m is not None
                and weather.relative_humidity_2m is not None
                and weather.temperature_2m < 15
                and weather.relative_humidity_2m > 80
            ):
                penalties["asthma_cold_damp_air"] = 15.0
                heuristic_notes.append(
                    "Cold, damp air may increase discomfort."
                )

        if health_profile.is_senior:
            if (
                weather.temperature_2m is not None
                and weather.temperature_2m > 32
            ):
                penalties["senior_heat_stress"] = 30.0
                unfavorable_reasons.append(
                    "Elevated heat exposure for older adults."
                )

            if (
                weather.precipitation_probability is not None
                and weather.precipitation_probability > 35
            ):
                penalties["senior_slip_hazard"] = 20.0
                heuristic_notes.append(
                    "Rain may increase slippery-surface risk."
                )

        if health_profile.is_child:
            if (
                weather.uv_index is not None
                and weather.uv_index >= 6
            ):
                penalties["child_high_uv"] = 25.0
                heuristic_notes.append(
                    "Elevated UV exposure for children."
                )

            if (
                geo_report
                and geo_report.is_downwind_of_road
            ):
                penalties["child_traffic_exposure"] = 15.0
                heuristic_notes.append(
                    "Potential traffic-related exposure."
                )

        if health_profile.is_pregnant:
            if (
                weather.temperature_2m is not None
                and weather.temperature_2m > 28
            ):
                penalties["pregnancy_thermal_stress"] = 25.0
                heuristic_notes.append(
                    "Warm conditions may increase thermal strain."
                )

            if pm25 is not None and pm25 > 45:
                penalties["pregnancy_air_quality"] = 20.0
                unfavorable_reasons.append(
                    "Elevated particulate exposure "
                    "during pregnancy."
                )
