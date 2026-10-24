"""Verify the public Cloudflare tunnel endpoint end-to-end."""
import sys
import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PUBLIC_URL = "https://indicated-origin-boutique-treo.trycloudflare.com"

def test_public():
    print(f"Testing Public Cloudflare Tunnel: {PUBLIC_URL}")

    with httpx.Client(timeout=45.0) as client:
        # 1. Test GET /health
        print("\n[1/3] Testing GET /health...")
        r_health = client.get(f"{PUBLIC_URL}/health")
        print(f"  --> Status: {r_health.status_code}")
        health_data = r_health.json()
        print(f"  --> Service: {health_data.get('service')}")
        print(f"  --> Supported Activities: {health_data.get('supported_activities_count')}")
        assert r_health.status_code == 200

        # 2. Test GET /activities
        print("\n[2/3] Testing GET /activities...")
        r_act = client.get(f"{PUBLIC_URL}/activities")
        print(f"  --> Status: {r_act.status_code}")
        act_data = r_act.json()
        print(f"  --> Total Activities: {act_data.get('total_count')}")
        assert r_act.status_code == 200

        # 3. Test POST /recommend
        print("\n[3/3] Testing POST /recommend via Public Internet...")
        payload = {
            "query": "Where can I go running tomorrow morning near Powai?",
            "use_slm": False
        }
        r_rec = client.post(f"{PUBLIC_URL}/recommend", json=payload)
        print(f"  --> Status: {r_rec.status_code}")
        rec_data = r_rec.json()
        print(f"  --> Query Status: {rec_data.get('status')}")
        print(f"  --> Has Suitable Option: {rec_data.get('has_suitable_option')}")
        print(f"  --> Venues Evaluated: {rec_data.get('total_venues_evaluated')}")
        best = rec_data.get("best_recommendation")
        if best:
            print(f"  --> Best Venue: {best.get('venue_name')}")
            print(f"  --> Best Slot Time: {best.get('slot_time_ist')}")
            print(f"  --> Recommendation Status: {best.get('recommendation_status')}")
        print(f"  --> Explanation:\n{rec_data.get('explanation')[:300]}...")
        assert r_rec.status_code == 200
        assert rec_data.get("status") == "success"

    print("\n" + "=" * 65)
    print("ALL PUBLIC ENDPOINTS VERIFIED AND ACCESSIBLE!")
    print("=" * 65)

if __name__ == "__main__":
    test_public()
