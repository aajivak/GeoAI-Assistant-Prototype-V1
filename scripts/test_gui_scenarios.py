"""Script to verify Nagpur cycling queries with use_slm: false and use_slm: true."""
import sys
import time
import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://127.0.0.1:8000"

def run_tests():
    print("=" * 70)
    print("VERIFYING GUI & API SCENARIOS (NAGPUR CYCLING)")
    print("=" * 70)

    with httpx.Client(base_url=BASE_URL, timeout=240.0) as client:
        # 1. Health check
        print("\n[1/4] Verifying GET /health...")
        r_health = client.get("/health")
        assert r_health.status_code == 200, f"Health check failed: {r_health.status_code}"
        h_data = r_health.json()
        print(f"  --> Status: {h_data.get('status')}")
        print(f"  --> Supported Activities: {h_data.get('supported_activities_count')}")

        # 2. Activities check
        print("\n[2/4] Verifying GET /activities...")
        r_act = client.get("/activities")
        assert r_act.status_code == 200, f"Activities check failed: {r_act.status_code}"
        a_data = r_act.json()
        print(f"  --> Total Activities: {a_data.get('total_count')}")

        # 3. Test Query with use_slm: False
        query_nagpur = "Find a place for cycling tomorrow morning near Nagpur"
        payload_fast = {
            "query": query_nagpur,
            "search_radius_km": 5.0,
            "user_profile": {
                "has_asthma": False,
                "is_senior": False,
                "is_child": False,
                "is_pregnant": False
            },
            "use_slm": False
        }
        print(f"\n[3/4] Testing POST /recommend (use_slm: False)...")
        print(f"  Query: '{query_nagpur}' | Radius: 5.0 km")
        t0 = time.perf_counter()
        r_fast = client.post("/recommend", json=payload_fast)
        duration_fast = time.perf_counter() - t0

        assert r_fast.status_code == 200, f"Recommendation failed: {r_fast.status_code}"
        d_fast = r_fast.json()
        print(f"  --> Request completed in {duration_fast:.2f}s")
        print(f"  --> Status: {d_fast.get('status')}")
        print(f"  --> Has Suitable Option: {d_fast.get('has_suitable_option')}")
        print(f"  --> Venues Evaluated: {d_fast.get('total_venues_evaluated')}")
        print(f"  --> Explanation Source: {d_fast.get('explanation_source')}")
        best_fast = d_fast.get("best_recommendation")
        if best_fast:
            print(f"  --> Best Venue: {best_fast.get('venue_name')} ({best_fast.get('place_type')})")
            print(f"  --> Distance: {best_fast.get('distance_km'):.2f} km")
            print(f"  --> Optimal Time: {best_fast.get('slot_time_ist')}")
            print(f"  --> Score: {best_fast.get('suitability_score')}/100 ({best_fast.get('recommendation_status')})")
            w = best_fast.get('weather_summary', {})
            a = best_fast.get('aqi_summary', {})
            print(f"  --> Weather: Temp {w.get('temperature_c')}°C, Rain {w.get('rain_probability_percent')}%, Wind {w.get('wind_speed_kmh')} km/h")
            print(f"  --> AQI: PM2.5 {a.get('pm2_5')} µg/m³, US AQI {a.get('us_aqi')}")
            print(f"  --> Geographic Advisory: {best_fast.get('geographic_context', {}).get('summary_advisory')}")
        print(f"  --> Explanation Snippet:\n{d_fast.get('explanation')[:200]}...")

        # 4. Test Query with use_slm: True
        payload_slm = {
            "query": query_nagpur,
            "search_radius_km": 5.0,
            "user_profile": {
                "has_asthma": False,
                "is_senior": False,
                "is_child": False,
                "is_pregnant": False
            },
            "use_slm": True
        }
        print(f"\n[4/4] Testing POST /recommend (use_slm: True - Qwen SLM on CPU)...")
        t0 = time.perf_counter()
        r_slm = client.post("/recommend", json=payload_slm)
        duration_slm = time.perf_counter() - t0

        assert r_slm.status_code == 200, f"SLM recommendation failed: {r_slm.status_code}"
        d_slm = r_slm.json()
        print(f"  --> Request completed in {duration_slm:.2f}s")
        print(f"  --> Status: {d_slm.get('status')}")
        print(f"  --> Has Suitable Option: {d_slm.get('has_suitable_option')}")
        print(f"  --> Explanation Source: {d_slm.get('explanation_source')}")
        if d_slm.get("model_diagnostics"):
            diag = d_slm.get("model_diagnostics")
            print(f"  --> Diagnostics: {diag.get('tokens_per_second')} tok/sec | {diag.get('generated_tokens')} tokens in {diag.get('inference_duration_seconds')}s")
            print(f"  --> Validation Passed: {diag.get('validation_passed')}")
        print(f"  --> SLM Grounded Explanation:\n{d_slm.get('explanation')}")

    print("\n" + "=" * 70)
    print("ALL VERIFICATION CHECKS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
