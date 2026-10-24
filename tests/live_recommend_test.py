"""Live end-to-end recommendation tests executing against live external APIs.

Scenarios:
1. Running tomorrow morning near Powai (CSRE IIT Bombay anchor).
2. Morning walk near Powai with asthma (testing conservative air-quality sensitivity under real data).
"""
import asyncio
import pytest

from src.api.routes import RecommendRequest, get_recommendation


async def run_live_recommendation_tests():
    """Execute live end-to-end recommendation queries against live open APIs."""
    print("=" * 65)
    print("STARTING LIVE END-TO-END RECOMMENDATION TEST (POWAI / IIT BOMBAY)")
    print("=" * 65)

    # --- Scenario 1: Running near Powai ---
    query_1 = "Where can I go running tomorrow morning near Powai?"
    print(f"\n[Scenario 1] Query: '{query_1}'")
    req_1 = RecommendRequest(query=query_1)
    res_1 = await get_recommendation(req_1)

    print(f"  Status: {res_1.status}")
    print(f"  Parsed Activity: {res_1.parsed_query.activity}")
    print(f"  Resolved Location: {res_1.parsed_query.spatial_anchor}")
    print(f"  Total Venues Evaluated: {res_1.total_venues_evaluated}")
    print(f"  Total Space-Time Slots Evaluated: {res_1.total_space_time_slots_evaluated}")

    assert res_1.status == "success"
    assert res_1.best_recommendation is not None
    best = res_1.best_recommendation
    print(f"\n  >>> BEST RECOMMENDATION <<<")
    print(f"  Venue: {best.venue_name} ({best.place_type})")
    print(f"  Distance: {best.distance_km:.2f} km from Powai")
    print(f"  Optimal Time: {best.slot_time_ist}")
    print(f"  Suitability Score: {best.suitability_score}/100 ({best.rating_category})")
    print(f"  Weather Summary: Temp {best.weather_summary['temperature_c']}°C, Rain Prob {best.weather_summary['rain_probability_percent']}%, Wind {best.weather_summary['wind_speed_kmh']} km/h")
    print(f"  AQI Summary: PM2.5 {best.aqi_summary['pm2_5']} µg/m³, US AQI {best.aqi_summary['us_aqi']}")
    print(f"  Geographic Advisory: {best.geographic_context.summary_advisory}")
    if res_1.temporal_comparison:
        print(f"  Temporal Comparison: {res_1.temporal_comparison}")

    # --- Scenario 2: High Pollution / Asthma Scenario near Powai ---
    query_2 = "I have mild asthma, where's good for a morning walk near Powai in the next two days?"
    print(f"\n[Scenario 2] Asthma Query: '{query_2}'")
    req_2 = RecommendRequest(query=query_2)
    res_2 = await get_recommendation(req_2)

    print(f"  Status: {res_2.status}")
    print(f"  Parsed Asthma Flag: {res_2.parsed_query.health_profile.has_asthma}")
    assert res_2.status == "success"
    assert res_2.parsed_query.health_profile.has_asthma is True
    assert res_2.best_recommendation is not None

    best_asthma = res_2.best_recommendation
    print(f"\n  >>> BEST ASTHMA RECOMMENDATION <<<")
    print(f"  Venue: {best_asthma.venue_name}")
    print(f"  Optimal Time: {best_asthma.slot_time_ist}")
    print(f"  Score: {best_asthma.suitability_score}/100 ({best_asthma.rating_category})")
    print(f"  Penalties Applied: {best_asthma.score_breakdown.penalties_applied}")
    print(f"  Heuristic Notes: {best_asthma.score_breakdown.heuristic_notes}")

    print("\n" + "=" * 65)
    print("LIVE RECOMMENDATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)
    return True


@pytest.mark.asyncio
async def test_live_recommendation_scenarios():
    """Pytest wrapper for live end-to-end recommendation tests."""
    success = await run_live_recommendation_tests()
    assert success is True


if __name__ == "__main__":
    asyncio.run(run_live_recommendation_tests())
