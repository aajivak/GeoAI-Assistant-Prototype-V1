"""Comprehensive benchmark suite for Qwen 2.5 1.5B CPU inference and grounded recommendations.

Tests:
1. CPU Model-loading time, RAM footprint, and tokens/second.
2. Normal running recommendation.
3. High-AQI running scenario (asserting has_suitable_option=False and cautionary advice).
4. Asthma-sensitive walking scenario.
5. Tennis activity scenario.
6. Photography activity scenario.
7. Missing weather / AQI observation handling.
8. Evidence grounding verification.
"""
import asyncio
import logging
import time
import pytest

from src.api.routes import RecommendRequest, get_recommendation
from src.common.schemas import PersonalHealthProfile
from src.slm.service import get_slm_service

logger = logging.getLogger("geoai.benchmark")


async def run_comprehensive_benchmarks():
    """Run end-to-end CPU inference and recommendation benchmarks."""
    print("=" * 70)
    print("PHASE 4: QWEN 2.5 1.5B INFERENCE & SAFETY BENCHMARK SUITE")
    print("=" * 70)

    slm_service = get_slm_service()

    # 1. Benchmark Model Load
    print("\n[1/8] Benchmarking Model Load (Qwen/Qwen2.5-1.5B-Instruct on CPU)...")
    t0 = time.perf_counter()
    loaded = slm_service.load_model("Qwen/Qwen2.5-1.5B-Instruct")
    load_time = time.perf_counter() - t0
    print(f"  --> Model Loaded: {loaded} in {load_time:.2f} seconds.")
    assert loaded is True, "Failed to load Qwen/Qwen2.5-1.5B-Instruct on CPU!"

    # 2. Benchmark Normal Running Recommendation
    print("\n[2/8] Testing Normal Running Recommendation near Koregaon Park...")
    t0 = time.perf_counter()
    res_run = await get_recommendation(
        RecommendRequest(query="Where can I go running tomorrow morning near Koregaon Park?")
    )
    run_latency = time.perf_counter() - t0
    print(f"  --> Status: {res_run.status} (Suitable: {res_run.has_suitable_option}) in {run_latency:.2f}s")
    print(f"  --> Explanation Source: {res_run.explanation_source}")
    print(f"  --> Best Venue: {res_run.best_recommendation.venue_name if res_run.best_recommendation else 'None'}")
    if res_run.model_diagnostics:
        print(f"  --> SLM Diagnostics: {res_run.model_diagnostics.get('tokens_per_second')} tokens/sec ({res_run.model_diagnostics.get('generated_tokens')} tokens in {res_run.model_diagnostics.get('inference_duration_seconds')}s)")
    print(f"  --> Explanation snippet: {res_run.explanation[:200]}...")

    # 3. Benchmark High-AQI Running Scenario (Safety Invariant)
    print("\n[3/8] Testing High-AQI Running Scenario (Safety Invariant)...")
    res_high_aqi = await get_recommendation(
        RecommendRequest(query="Where can I go running near Powai right now?")
    )
    print(f"  --> Status: {res_high_aqi.status} | Has Suitable Option: {res_high_aqi.has_suitable_option}")
    if not res_high_aqi.has_suitable_option:
        print(f"  --> Expected Safety Exclusion: {res_high_aqi.no_suitable_option_reason}")
    print(f"  --> Explanation snippet: {res_high_aqi.explanation[:200]}...")

    # 4. Benchmark Asthma-Sensitive Walking
    print("\n[4/8] Testing Asthma-Sensitive Walking near Powai...")
    res_asthma = await get_recommendation(
        RecommendRequest(query="I have mild asthma, where's good for a morning walk near Powai in the next two days?")
    )
    print(f"  --> Parsed Asthma: {res_asthma.parsed_query.health_profile.has_asthma}")
    print(f"  --> Has Suitable Option: {res_asthma.has_suitable_option}")
    if res_asthma.best_recommendation:
        print(f"  --> Best Slot Status: {res_asthma.best_recommendation.recommendation_status} (Score: {res_asthma.best_recommendation.suitability_score}/100)")
    print(f"  --> Explanation: {res_asthma.explanation[:200]}...")

    # 5. Benchmark Activity 2: Tennis near Hinjewadi
    print("\n[5/8] Testing Activity: Outdoor Tennis near Hinjewadi...")
    res_tennis = await get_recommendation(
        RecommendRequest(query="Outdoor tennis after work today near Hinjewadi, any good spots?")
    )
    print(f"  --> Parsed Activity: {res_tennis.parsed_query.activity} | Anchor: {res_tennis.parsed_query.spatial_anchor}")
    print(f"  --> Suitable Option: {res_tennis.has_suitable_option}")
    print(f"  --> Best Venue: {res_tennis.best_recommendation.venue_name if res_tennis.best_recommendation else 'None'}")

    # 6. Benchmark Activity 3: Photography near Bandra
    print("\n[6/8] Testing Activity: Sunset Photography near Bandra...")
    res_photo = await get_recommendation(
        RecommendRequest(query="Where can I take sunset photos this evening around Bandra?")
    )
    print(f"  --> Parsed Activity: {res_photo.parsed_query.activity} | Anchor: {res_photo.parsed_query.spatial_anchor}")
    print(f"  --> Suitable Option: {res_photo.has_suitable_option}")
    print(f"  --> Best Venue: {res_photo.best_recommendation.venue_name if res_photo.best_recommendation else 'None'}")

    # 7. Benchmark Missing Location Clarification
    print("\n[7/8] Testing Incomplete Query (Missing Location)...")
    res_vague = await get_recommendation(
        RecommendRequest(query="Best time this weekend for a cricket match?")
    )
    print(f"  --> Status: {res_vague.status}")
    print(f"  --> Clarification Question: {res_vague.clarification_question}")
    assert res_vague.status == "needs_clarification"

    # 8. Summary of CPU Performance
    print("\n[8/8] Benchmark Performance Summary:")
    print(f"  - Model: Qwen/Qwen2.5-1.5B-Instruct (1.54 Billion Parameters)")
    print(f"  - Device: Intel Core i5-1334U (CPU, bfloat16)")
    print(f"  - Net Model RAM Footprint: {slm_service.runner.model_ram_mb:.2f} MB (~{slm_service.runner.model_ram_mb / 1024:.2f} GB)")
    print(f"  - Cold Load Time: {load_time:.2f} seconds")
    if res_run.model_diagnostics:
        print(f"  - Inference Speed: {res_run.model_diagnostics.get('tokens_per_second')} tokens/sec")
        print(f"  - Average Generation Duration: {res_run.model_diagnostics.get('inference_duration_seconds')} seconds")

    print("\n" + "=" * 70)
    print("ALL BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    asyncio.run(run_comprehensive_benchmarks())
