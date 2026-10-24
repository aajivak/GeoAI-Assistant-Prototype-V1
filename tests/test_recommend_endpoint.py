"""Integration tests for POST /recommend endpoint using mocked external APIs."""
from datetime import datetime
from fastapi.testclient import TestClient
import pytest
from unittest.mock import AsyncMock, patch

from src.api.app import app
from src.common.coordinates import LatLon
from src.common.schemas import GeocodingResult, HourlyAQIObservation, HourlyWeatherObservation, PlaceCandidate
from src.common.temporal import IST

client = TestClient(app)

MOCK_GEO_RESULT = GeocodingResult(
    query="Koregaon Park, Pune",
    display_name="Koregaon Park, Pune, Maharashtra, India",
    coordinate=LatLon(latitude=18.5362, longitude=73.8940),
)

MOCK_VENUES = [
    PlaceCandidate(
        id="way/101",
        name="Osho Teerth Park",
        place_type="park",
        coordinate=LatLon(latitude=18.5380, longitude=73.8960),
        distance_from_anchor_km=0.35,
    ),
    PlaceCandidate(
        id="node/102",
        name="Koregaon Park Jogging Track",
        place_type="track",
        coordinate=LatLon(latitude=18.5310, longitude=73.8910),
        distance_from_anchor_km=0.60,
    ),
]

MOCK_WEATHER = [
    HourlyWeatherObservation(
        time=datetime(2026, 10, 25, h, 0, tzinfo=IST),
        temperature_2m=19.0 + (h - 6) * 1.5,
        relative_humidity_2m=65.0,
        precipitation_probability=0.0,
        precipitation=0.0,
        wind_speed_10m=8.0,
        wind_direction_10m=120.0,
        uv_index=1.0,
    )
    for h in range(6, 12)
]

# Good air quality for standard test: PM2.5 = 18 µg/m³
MOCK_AQI_GOOD = [
    HourlyAQIObservation(
        time=datetime(2026, 10, 25, h, 0, tzinfo=IST),
        pm2_5=18.0,
        pm10=35.0,
        us_aqi=45,
    )
    for h in range(6, 12)
]


@patch("src.api.routes.NominatimClient.resolve_primary_location", new_callable=AsyncMock)
@patch("src.api.routes.OverpassClient.discover_venues", new_callable=AsyncMock)
@patch("src.api.routes.OverpassClient.enrich_with_geographic_context", new_callable=AsyncMock)
@patch("src.api.routes.OpenMeteoWeatherClient.fetch_hourly_forecast", new_callable=AsyncMock)
@patch("src.api.routes.OpenMeteoAQIClient.fetch_hourly_aqi", new_callable=AsyncMock)
def test_recommend_endpoint_success(mock_aqi, mock_weather, mock_enrich, mock_venues, mock_geo):
    """Test successful recommendation workflow using mocked APIs."""
    mock_geo.return_value = MOCK_GEO_RESULT
    mock_venues.return_value = MOCK_VENUES
    mock_enrich.return_value = MOCK_VENUES
    mock_weather.return_value = MOCK_WEATHER
    mock_aqi.return_value = MOCK_AQI_GOOD

    payload = {"query": "Where can I go running tomorrow morning near Koregaon Park?"}
    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["has_suitable_option"] is True
    assert data["explanation"] is not None
    assert "explanation_source" in data
    assert data["best_recommendation"] is not None
    assert data["best_recommendation"]["recommendation_status"] == "recommended"
    assert len(data["data_sources"]) >= 4
    assert "Disclaimer" in data["disclaimer"]


def test_recommend_endpoint_missing_location():
    """Test that missing location triggers explicit clarification question."""
    payload = {"query": "Where can I go running tomorrow morning?"}
    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "needs_clarification"
    assert data["has_suitable_option"] is False
    assert data["clarification_question"] is not None
    assert "location" in data["clarification_question"].lower()
    assert data["best_recommendation"] is None


@patch("src.api.routes.NominatimClient.resolve_primary_location", new_callable=AsyncMock)
@patch("src.api.routes.OverpassClient.discover_venues", new_callable=AsyncMock)
@patch("src.api.routes.OverpassClient.enrich_with_geographic_context", new_callable=AsyncMock)
@patch("src.api.routes.OpenMeteoWeatherClient.fetch_hourly_forecast", new_callable=AsyncMock)
@patch("src.api.routes.OpenMeteoAQIClient.fetch_hourly_aqi", new_callable=AsyncMock)
def test_recommend_endpoint_asthma_high_pollution(mock_aqi, mock_weather, mock_enrich, mock_venues, mock_geo):
    """Test that an asthmatic user receives conservative scores and has_suitable_option=False under high PM2.5."""
    mock_geo.return_value = MOCK_GEO_RESULT
    mock_venues.return_value = MOCK_VENUES
    mock_enrich.return_value = MOCK_VENUES
    mock_weather.return_value = MOCK_WEATHER
    # High PM2.5: 85 µg/m³
    mock_aqi.return_value = [
        HourlyAQIObservation(
            time=datetime(2026, 10, 25, h, 0, tzinfo=IST),
            pm2_5=85.0,
            pm10=130.0,
            us_aqi=185,
        )
        for h in range(6, 12)
    ]

    payload = {"query": "I have mild asthma, where's good for a morning walk near Koregaon Park tomorrow?"}
    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["has_suitable_option"] is False
    assert data["no_suitable_option_reason"] is not None
    assert data["best_recommendation"]["recommendation_status"] == "not_recommended"
    assert "asthma_high_pm25" in data["best_recommendation"]["score_breakdown"]["penalties_applied"]
    assert "NO SUITABLE OPTION" in data["explanation"]


@patch("src.api.routes.NominatimClient.resolve_primary_location", new_callable=AsyncMock)
@patch("src.api.routes.OverpassClient.discover_venues", new_callable=AsyncMock)
@patch("src.api.routes.OverpassClient.enrich_with_geographic_context", new_callable=AsyncMock)
@patch("src.api.routes.OpenMeteoWeatherClient.fetch_hourly_forecast", new_callable=AsyncMock)
@patch("src.api.routes.OpenMeteoAQIClient.fetch_hourly_aqi", new_callable=AsyncMock)
def test_recommend_endpoint_slm_flag_handling(mock_aqi, mock_weather, mock_enrich, mock_venues, mock_geo):
    """Test that explicit use_slm=False guarantees deterministic synthesis."""
    mock_geo.return_value = MOCK_GEO_RESULT
    mock_venues.return_value = MOCK_VENUES
    mock_enrich.return_value = MOCK_VENUES
    mock_weather.return_value = MOCK_WEATHER
    mock_aqi.return_value = MOCK_AQI_GOOD

    payload = {
        "query": "Where can I go running tomorrow morning near Koregaon Park?",
        "use_slm": False,
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["explanation_source"] == "deterministic_engine"
