"""Unit tests for activity suitability scoring, normalized weighted averaging, and safety status."""
from datetime import datetime
import pytest

from src.common.schemas import HourlyAQIObservation, HourlyWeatherObservation, PersonalHealthProfile
from src.common.temporal import IST
from src.scoring.evaluator import SuitabilityEvaluator


@pytest.fixture
def evaluator():
    return SuitabilityEvaluator()


def test_normalized_scoring_ideal_conditions(evaluator):
    """Verify that ideal conditions yield a high score and RECOMMENDED absolute status."""
    weather = HourlyWeatherObservation(
        time=datetime(2026, 10, 24, 7, 0, tzinfo=IST),
        temperature_2m=18.0,
        relative_humidity_2m=55.0,
        precipitation_probability=0.0,
        precipitation=0.0,
        wind_speed_10m=8.0,
        wind_direction_10m=180.0,
        uv_index=1.0,
        cloud_cover=20.0,
        weather_code=0,
    )
    aqi = HourlyAQIObservation(
        time=datetime(2026, 10, 24, 7, 0, tzinfo=IST),
        pm2_5=12.0,  # Good air
        pm10=25.0,
        us_aqi=35,
    )

    breakdown = evaluator.evaluate(activity_id="running", weather=weather, aqi=aqi)
    assert 85.0 <= breakdown.overall_score <= 100.0
    assert breakdown.recommendation_status == "recommended"
    assert breakdown.is_favorable is True
    assert "Disclaimer" in breakdown.disclaimer


def test_running_high_aqi_safety_correction(evaluator):
    """REGRESSION TEST: Verify that running in PM2.5 > 60 µg/m³ (US AQI > 150) receives NOT_RECOMMENDED status.

    Addresses the safety correction: running in unhealthy air must NOT be marked suitable.
    """
    weather = HourlyWeatherObservation(
        time=datetime(2026, 10, 24, 6, 0, tzinfo=IST),
        temperature_2m=25.0,
        relative_humidity_2m=65.0,
        precipitation_probability=0.0,
        precipitation=0.0,
        wind_speed_10m=5.0,
        wind_direction_10m=180.0,
        uv_index=1.0,
    )
    # PM2.5 61.9 µg/m³, US AQI 164 (unhealthy)
    aqi_unhealthy = HourlyAQIObservation(
        time=datetime(2026, 10, 24, 6, 0, tzinfo=IST),
        pm2_5=61.9,
        pm10=95.0,
        us_aqi=164,
    )

    breakdown = evaluator.evaluate("running", weather, aqi=aqi_unhealthy)
    # Must have absolute status NOT_RECOMMENDED!
    assert breakdown.recommendation_status == "not_recommended"
    assert breakdown.is_favorable is False
    assert any("cardio" in r.lower() or "unhealthy" in r.lower() for r in breakdown.unfavorable_reasons)
    # Penalty must be subtracted
    assert "high_cardio_unhealthy_aqi" in breakdown.penalties_applied


def test_missing_data_weight_renormalization(evaluator):
    """Verify that missing PM2.5 is preserved as None and remaining weights are re-normalized."""
    weather = HourlyWeatherObservation(
        time=datetime(2026, 10, 24, 8, 0, tzinfo=IST),
        temperature_2m=20.0,
        relative_humidity_2m=60.0,
        precipitation_probability=0.0,
        precipitation=0.0,
        wind_speed_10m=10.0,
        wind_direction_10m=180.0,
        uv_index=2.0,
        cloud_cover=10.0,
        weather_code=0,
    )
    aqi = HourlyAQIObservation(
        time=datetime(2026, 10, 24, 8, 0, tzinfo=IST),
        pm2_5=None,
        pm10=None,
        us_aqi=None,
    )

    breakdown = evaluator.evaluate(activity_id="running", weather=weather, aqi=aqi)
    assert breakdown.sub_scores["aqi"] is None
    assert any("air-quality" in warn.lower() for warn in breakdown.uncertainty_warnings)
    assert 0.0 <= breakdown.overall_score <= 100.0


def test_asthma_high_pollution_penalty_and_status(evaluator):
    """Verify that elevated PM2.5 sets absolute status to NOT_RECOMMENDED for asthmatics."""
    weather = HourlyWeatherObservation(
        time=datetime(2026, 10, 24, 8, 0, tzinfo=IST),
        temperature_2m=22.0,
        relative_humidity_2m=60.0,
        precipitation_probability=0.0,
        precipitation=0.0,
        wind_speed_10m=8.0,
        wind_direction_10m=180.0,
    )
    aqi_high = HourlyAQIObservation(
        time=datetime(2026, 10, 24, 8, 0, tzinfo=IST),
        pm2_5=65.0,  # Above asthma exclusion threshold
        pm10=110.0,
        us_aqi=160,
    )

    asthma_profile = PersonalHealthProfile(has_asthma=True)
    res_asthma = evaluator.evaluate("walking", weather, aqi=aqi_high, health_profile=asthma_profile)

    assert res_asthma.recommendation_status == "not_recommended"
    assert res_asthma.is_favorable is False
    assert "asthma_high_pm25" in res_asthma.penalties_applied


def test_senior_heat_and_child_uv_penalties(evaluator):
    """Verify senior heat penalty and child UV penalty."""
    hot_sunny_weather = HourlyWeatherObservation(
        time=datetime(2026, 10, 24, 13, 0, tzinfo=IST),
        temperature_2m=34.5,
        relative_humidity_2m=40.0,
        precipitation_probability=0.0,
        precipitation=0.0,
        wind_speed_10m=12.0,
        wind_direction_10m=180.0,
        uv_index=7.5,
    )

    senior_res = evaluator.evaluate("walking", hot_sunny_weather, health_profile=PersonalHealthProfile(is_senior=True))
    assert "senior_heat_stress" in senior_res.penalties_applied

    child_res = evaluator.evaluate("cricket", hot_sunny_weather, health_profile=PersonalHealthProfile(is_child=True))
    assert "child_high_uv" in child_res.penalties_applied
