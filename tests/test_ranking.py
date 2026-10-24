"""Unit tests for joint place-and-time candidate ranking across multiple venues and forecast hours."""
from datetime import datetime, timedelta
import pytest

from src.common.coordinates import LatLon
from src.common.schemas import HourlyAQIObservation, HourlyWeatherObservation, PlaceCandidate
from src.common.temporal import IST
from src.scoring.candidate_ranker import CandidateRanker


def test_candidate_ranking_place_and_time():
    """Verify that ranking compares place-and-time combinations and picks the optimal slot."""
    t1 = datetime(2026, 10, 24, 6, 0, tzinfo=IST)
    t2 = datetime(2026, 10, 24, 13, 0, tzinfo=IST)

    candidates = [
        PlaceCandidate(
            id="node/1",
            name="Hilltop Nature Park",
            place_type="park",
            coordinate=LatLon(latitude=18.52, longitude=73.85),
            distance_from_anchor_km=2.0,
        ),
        PlaceCandidate(
            id="node/2",
            name="City Square Ground",
            place_type="pitch",
            coordinate=LatLon(latitude=18.53, longitude=73.86),
            distance_from_anchor_km=3.5,
        ),
    ]

    weather_forecasts = [
        HourlyWeatherObservation(
            time=t1,
            temperature_2m=19.0,
            relative_humidity_2m=60.0,
            precipitation_probability=0.0,
            precipitation=0.0,
            wind_speed_10m=6.0,
            wind_direction_10m=90.0,
            uv_index=0.0,
        ),
        HourlyWeatherObservation(
            time=t2,
            temperature_2m=33.0,
            relative_humidity_2m=35.0,
            precipitation_probability=0.0,
            precipitation=0.0,
            wind_speed_10m=18.0,
            wind_direction_10m=90.0,
            uv_index=8.0,
        ),
    ]

    aqi_forecasts = [
        HourlyAQIObservation(time=t1, pm2_5=15.0, us_aqi=40),
        HourlyAQIObservation(time=t2, pm2_5=35.0, us_aqi=75),
    ]

    ranker = CandidateRanker()
    result = ranker.rank_candidates(
        activity_id="running",
        candidates=candidates,
        weather_forecasts=weather_forecasts,
        aqi_forecasts=aqi_forecasts,
        target_slots=[t1, t2],
        top_k=4,
    )

    assert result.total_venues_evaluated == 2
    assert result.total_slots_evaluated == 4
    assert result.best_recommendation is not None
    assert result.has_suitable_option is True

    # 6 AM slot must beat 1 PM slot
    assert "06:00" in result.best_recommendation.slot_time_ist
    assert result.best_recommendation.recommendation_status == "recommended"

    scores = [c.suitability_score for c in result.top_candidates]
    assert scores == sorted(scores, reverse=True)
    assert result.temporal_comparison_note is not None


def test_no_suitable_option_scenario():
    """REGRESSION TEST: Verify that when all slots are unfavorable, has_suitable_option is False."""
    t1 = datetime(2026, 10, 24, 6, 0, tzinfo=IST)
    candidates = [
        PlaceCandidate(
            id="node/1",
            name="Expressway Park",
            place_type="park",
            coordinate=LatLon(latitude=18.52, longitude=73.85),
        )
    ]
    # Heavy pollution across the entire forecast: PM2.5 = 75 µg/m³
    weather = [
        HourlyWeatherObservation(
            time=t1,
            temperature_2m=26.0,
            precipitation_probability=0.0,
            precipitation=0.0,
            wind_speed_10m=5.0,
        )
    ]
    aqi = [
        HourlyAQIObservation(
            time=t1,
            pm2_5=75.0,
            us_aqi=165,
        )
    ]

    ranker = CandidateRanker()
    result = ranker.rank_candidates("running", candidates, weather, aqi, [t1])

    # Must be NOT suitable despite being rank 1
    assert result.has_suitable_option is False
    assert result.best_recommendation is not None
    assert result.best_recommendation.recommendation_status == "not_recommended"
    assert result.no_suitable_option_reason is not None
    assert "no suitable outdoor window" in result.no_suitable_option_reason.lower()
