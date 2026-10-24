import httpx

try:
    r = httpx.get("https://overpass-api.de/api/interpreter", params={"data": '[out:json];node(19.11,72.89,19.13,72.92);out 1;'}, headers={"User-Agent": "GeoAI-IITB/1.0"}, timeout=6.0)
    print("STATUS:", r.status_code)
    print("BODY:", r.text[:300])
except Exception as e:
    print("FAILED:", e)
