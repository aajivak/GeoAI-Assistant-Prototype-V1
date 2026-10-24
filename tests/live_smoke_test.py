"""Live integration smoke test for external APIs (Nominatim, Overpass, Open-Meteo).

Exercises live endpoints to verify connectivity, payload schemas, and rate-limit compliance.
Target anchor: Powai, Mumbai (CSRE, IIT Bombay neighborhood).
"""
import asyncio
import sys
import pytest

from src.common.coordinates import LatLon
from src.data_ingestion.geocoding import NominatimClient
from src.data_ingestion.osm_places import OverpassClient
from src.data_ingestion.weather_client import OpenMeteoWeatherClient
from src.data_ingestion.aqi_client import OpenMeteoAQIClient


async def run_live_smoke_test():
    """Execute live smoke test across all four open APIs."""
    print("=" * 60)
    print("STARTING LIVE EXTERNAL API SMOKE TEST (CSRE IIT Bombay Focus)")
    print("=" * 60)

    # 1. Nominatim Geocoding
    print("\n[1/4] Testing Nominatim Geocoding for 'Powai, Mumbai'...")
    nom_client = NominatimClient()
    try:
        geo_result = await nom_client.resolve_primary_location("Powai, Mumbai")
        print(f"  --> SUCCESS: Resolved to: {geo_result.display_name}")
        print(f"  --> Coordinates: ({geo_result.coordinate.latitude}, {geo_result.coordinate.longitude})")
        target_coord = geo_result.coordinate
    except Exception as e:
        print(f"  --> FAILED: Nominatim geocoding error: {e}")
        raise

    # 2. Open-Meteo Weather
    print("\n[2/4] Testing Open-Meteo Live Weather Forecast...")
    weather_client = OpenMeteoWeatherClient()
    try:
        weather_obs = await weather_client.fetch_hourly_forecast(target_coord, forecast_days=2)
        print(f"  --> SUCCESS: Received {len(weather_obs)} hourly weather forecast slots.")
        if weather_obs:
            first_slot = weather_obs[0]
            print(f"  --> Sample slot ({first_slot.time.isoformat()}):")
            print(f"      Temp: {first_slot.temperature_2m}°C | Humidity: {first_slot.relative_humidity_2m}% | Rain Prob: {first_slot.precipitation_probability}% | Wind: {first_slot.wind_speed_10m} km/h")
    except Exception as e:
        print(f"  --> FAILED: Open-Meteo Weather error: {e}")
        raise

    # 3. Open-Meteo Air Quality
    print("\n[3/4] Testing Open-Meteo Live Air Quality Forecast...")
    aqi_client = OpenMeteoAQIClient()
    try:
        aqi_obs = await aqi_client.fetch_hourly_aqi(target_coord, forecast_days=2)
        print(f"  --> SUCCESS: Received {len(aqi_obs)} hourly AQI forecast slots.")
        if aqi_obs:
            first_aqi = aqi_obs[0]
            print(f"  --> Sample slot ({first_aqi.time.isoformat()}):")
            print(f"      PM2.5: {first_aqi.pm2_5} µg/m³ | PM10: {first_aqi.pm10} µg/m³ | US AQI: {first_aqi.us_aqi}")
    except Exception as e:
        print(f"  --> FAILED: Open-Meteo AQI error: {e}")
        raise

    # 4. OpenStreetMap Overpass Venue Discovery & Context Enrichment
    print("\n[4/4] Testing OSM Overpass API Venue Discovery (Running near Powai)...")
    overpass_client = OverpassClient()
    try:
        candidates = await overpass_client.discover_venues(
            center=target_coord,
            activity_id="running",
            radius_km=4.0,
            limit=5,
        )
        print(f"  --> SUCCESS: Discovered {len(candidates)} venues within 4 km of Powai.")
        for idx, cand in enumerate(candidates, 1):
            print(f"      {idx}. {cand.name} ({cand.place_type}) - {cand.distance_from_anchor_km:.2f} km away")

        print("  --> Enriching venues with qualitative geographic context (roads/industry/water)...")
        enriched = await overpass_client.enrich_with_geographic_context(candidates, center=target_coord, radius_km=4.0)
        for cand in enriched:
            if cand.nearby_context:
                print(f"      * {cand.name} has {len(cand.nearby_context)} nearby geographic features:")
                for feat in cand.nearby_context[:2]:
                    print(f"        - {feat.feature_type} ({feat.name or 'unnamed'}): {feat.distance_meters:.0f}m away, bearing {feat.bearing_degrees:.1f}°")
    except Exception as e:
        print(f"  --> FAILED: Overpass venue discovery error: {e}")
        raise

    print("\n" + "=" * 60)
    print("ALL LIVE SMOKE TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    return True


@pytest.mark.asyncio
async def test_live_apis_smoke():
    """Pytest wrapper for live smoke test."""
    success = await run_live_smoke_test()
    assert success is True


if __name__ == "__main__":
    asyncio.run(run_live_smoke_test())
