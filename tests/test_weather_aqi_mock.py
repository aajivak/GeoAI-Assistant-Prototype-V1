"""Unit tests for Open-Meteo Weather and AQI clients.

Strictly verifies the invariant:
Missing API observations must remain None and never be substituted with invented defaults.
"""
from datetime import datetime
import pytest
from unittest.mock import AsyncMock

from src.common.coordinates import LatLon
from src.common.http_client import AsyncHttpClient
from src.data_ingestion.weather_client import OpenMeteoWeatherClient
from src.data_ingestion.aqi_client import OpenMeteoAQIClient


MOCK_WEATHER_RESPONSE = {
    "latitude": 18.52,
    "longitude": 73.85,
    "generationtime_ms": 0.12,
    "utc_offset_seconds": 19800,
    "timezone": "Asia/Kolkata",
    "hourly": {
        "time": ["2026-10-24T06:00", "2026-10-24T07:00"],
        "temperature_2m": [21.4, None],  # One valid, one missing observation!
        "relative_humidity_2m": [65.0, 70.0],
        "precipitation_probability": [0.0, None],
        "precipitation": [0.0, 0.0],
        "wind_speed_10m": [8.5, 9.2],
        "wind_direction_10m": [180.0, 195.0],
        "uv_index": [0.0, 1.2],
        "cloud_cover": [20.0, 35.0],
        "weather_code": [1, None],
    },
}

MOCK_AQI_RESPONSE = {
    "latitude": 18.52,
    "longitude": 73.85,
    "hourly": {
        "time": ["2026-10-24T06:00", "2026-10-24T07:00"],
        "pm2_5": [28.5, None],  # One valid, one missing measurement
        "pm10": [52.1, 48.0],
        "us_aqi": [85, None],
        "european_aqi": [40, 38],
    },
}


@pytest.mark.asyncio
async def test_weather_parsing_and_missing_values_preserved():
    """Verify that parsed weather preserves None for missing measurements without inventing values."""
    mock_http = AsyncMock(spec=AsyncHttpClient)
    mock_http.get_json.return_value = MOCK_WEATHER_RESPONSE

    client = OpenMeteoWeatherClient(http_client=mock_http)
    coord = LatLon(latitude=18.52, longitude=73.85)

    observations = await client.fetch_hourly_forecast(coord)
    assert len(observations) == 2

    obs_0 = observations[0]
    assert obs_0.temperature_2m == 21.4
    assert obs_0.relative_humidity_2m == 65.0
    assert obs_0.precipitation_probability == 0.0
    assert obs_0.weather_code == 1

    obs_1 = observations[1]
    # INVARIANT CHECK: Missing values must remain None!
    assert obs_1.temperature_2m is None
    assert obs_1.precipitation_probability is None
    assert obs_1.weather_code is None
    # Present values must remain accurate
    assert obs_1.relative_humidity_2m == 70.0
    assert obs_1.wind_speed_10m == 9.2


@pytest.mark.asyncio
async def test_aqi_parsing_and_missing_values_preserved():
    """Verify that parsed air quality preserves None for missing measurements without inventing values."""
    mock_http = AsyncMock(spec=AsyncHttpClient)
    mock_http.get_json.return_value = MOCK_AQI_RESPONSE

    client = OpenMeteoAQIClient(http_client=mock_http)
    coord = LatLon(latitude=18.52, longitude=73.85)

    observations = await client.fetch_hourly_aqi(coord)
    assert len(observations) == 2

    obs_0 = observations[0]
    assert obs_0.pm2_5 == 28.5
    assert obs_0.us_aqi == 85

    obs_1 = observations[1]
    # INVARIANT CHECK: Missing values must remain None!
    assert obs_1.pm2_5 is None
    assert obs_1.us_aqi is None
    assert obs_1.pm10 == 48.0
    assert obs_1.european_aqi == 38
