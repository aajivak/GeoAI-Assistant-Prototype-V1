"""Unit tests for SLM prompt construction, post-validation guards, and deterministic fallback generation."""
import pytest
from datetime import datetime

from src.common.coordinates import LatLon
from src.common.schemas import HourlyAQIObservation, HourlyWeatherObservation, ParsedQuery, PlaceCandidate
from src.common.temporal import IST
from src.scoring.candidate_ranker import CandidateRankingResult, PlaceTimeSlotRecommendation
from src.scoring.evaluator import SuitabilityEvaluator
from src.slm.deterministic_generator import generate_deterministic_explanation
from src.slm.post_validator import SLMPostValidator
from src.slm.prompt_builder import build_grounded_slm_prompt
from src.spatial_analysis.wind_dispersion import GeographicContextReport


@pytest.fixture
def sample_query():
    return ParsedQuery(
        raw_query="Where can I go running tomorrow morning near Powai?",
        activity="running",
        spatial_anchor="Powai",
        search_radius_km=8.0,
    )


@pytest.fixture
def sample_ranking():
    evaluator = SuitabilityEvaluator()
    w = HourlyWeatherObservation(
        time=datetime(2026, 10, 25, 6, 0, tzinfo=IST),
        temperature_2m=21.0,
        precipitation_probability=0.0,
        wind_speed_10m=5.0,
    )
    a = HourlyAQIObservation(
        time=datetime(2026, 10, 25, 6, 0, tzinfo=IST),
        pm2_5=22.0,
        us_aqi=72,
    )
    cand = PlaceCandidate(
        id="way/101",
        name="Powai Vihar Park",
        place_type="park",
        coordinate=LatLon(latitude=19.117, longitude=72.902),
        distance_from_anchor_km=0.49,
    )
    geo = GeographicContextReport(
        venue_id=cand.id,
        venue_name=cand.name,
        has_green_canopy=True,
        summary_advisory="Beneficial green canopy buffer present.",
    )
    score_breakdown = evaluator.evaluate("running", w, a, geo)

    slot = PlaceTimeSlotRecommendation(
        venue_id=cand.id,
        venue_name=cand.name,
        place_type=cand.place_type,
        coordinate=cand.coordinate,
        distance_km=cand.distance_from_anchor_km,
        slot_time_ist="2026-10-25 06:00 IST",
        suitability_score=score_breakdown.overall_score,
        rating_category=score_breakdown.rating_category,
        recommendation_status=score_breakdown.recommendation_status,
        is_favorable=score_breakdown.is_favorable,
        score_breakdown=score_breakdown,
        geographic_context=geo,
        weather_summary={"temperature_c": 21.0, "humidity_percent": 60.0, "wind_speed_kmh": 5.0},
        aqi_summary={"pm2_5": 22.0, "us_aqi": 72},
    )

    return CandidateRankingResult(
        activity_id="running",
        total_venues_evaluated=1,
        total_slots_evaluated=1,
        has_suitable_option=True,
        best_recommendation=slot,
        top_candidates=[slot],
        temporal_comparison_note="Optimal morning hour.",
    )


def test_build_prompt_evidence_table(sample_query, sample_ranking):
    """Verify that the constructed prompt injects only verified deterministic facts."""
    prompt = build_grounded_slm_prompt(sample_query, sample_ranking)
    assert "Powai Vihar Park" in prompt
    assert "21.0°C" in prompt
    assert "PM2.5: 22.0 µg/m³" in prompt
    assert "Has Suitable Option: True" in prompt
    assert "Never invent places" in prompt


def test_post_validator_rejects_unsafe_claim_when_not_suitable(sample_query, sample_ranking):
    """Verify that validator flags any positive safety claim when conditions are unfavorable."""
    sample_ranking.has_suitable_option = False
    sample_ranking.no_suitable_option_reason = "Unhealthy PM2.5."

    hallucinated_text = "It is highly recommended and great conditions to go running at Powai Vihar Park tomorrow!"
    is_valid, errors = SLMPostValidator.validate(hallucinated_text, sample_query, sample_ranking)

    assert is_valid is False
    assert any("safety assertion" in err for err in errors)


def test_post_validator_accepts_grounded_caution_output(sample_query, sample_ranking):
    """Verify that validator passes an accurate, grounded cautionary explanation."""
    sample_ranking.has_suitable_option = False
    sample_ranking.no_suitable_option_reason = "PM2.5 is high."

    valid_caution_text = (
        "Outdoor running near Powai is not recommended tomorrow morning due to elevated particulate matter (PM2.5). "
        "Conditions at Powai Vihar Park are unfavorable. You should postpone outdoor exertion or exercise indoors."
    )
    is_valid, errors = SLMPostValidator.validate(valid_caution_text, sample_query, sample_ranking)
    assert is_valid is True
    assert len(errors) == 0


def test_deterministic_generator_content(sample_query, sample_ranking):
    """Verify that deterministic generator covers venue, timing, weather, and advisories."""
    text = generate_deterministic_explanation(sample_query, sample_ranking)
    assert "Powai Vihar Park" in text
    assert "06:00 IST" in text
    assert "21.0°C" in text
    assert "PM2.5 22.0" in text


def test_hf_runner_interface():
    """Verify HF runner interface and configuration defaults."""
    from src.slm.hf_runner import HFTransformersSLMRunner
    runner = HFTransformersSLMRunner()
    assert runner.is_loaded() is False
    assert runner.device == "cpu"
    assert runner.dtype_str == "bfloat16"
