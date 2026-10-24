"""Unit tests for Nominatim geocoding using mocked HTTP responses."""
import pytest
from unittest.mock import AsyncMock, patch

from src.common.coordinates import LatLon
from src.common.http_client import AsyncHttpClient, GeocodingError
from src.data_ingestion.geocoding import NominatimClient


MOCK_NOMINATIM_SUCCESS = [
    {
        "place_id": 123456,
        "licence": "Data © OpenStreetMap contributors, ODbL 1.0. http://osm.org/copyright",
        "osm_type": "node",
        "osm_id": 987654321,
        "lat": "18.5362084",
        "lon": "73.8939748",
        "category": "place",
        "type": "suburb",
        "place_rank": 19,
        "importance": 0.45,
        "addresstype": "suburb",
        "name": "Koregaon Park",
        "display_name": "Koregaon Park, Pune, Haveli, Pune District, Maharashtra, 411001, India",
        "boundingbox": ["18.525", "18.545", "73.880", "73.910"],
    }
]


@pytest.mark.asyncio
async def test_geocode_success():
    """Test parsing of successful Nominatim geocoding response."""
    mock_http = AsyncMock(spec=AsyncHttpClient)
    mock_http.get_json.return_value = MOCK_NOMINATIM_SUCCESS

    client = NominatimClient(http_client=mock_http)
    results = await client.geocode("Koregaon Park, Pune")

    assert len(results) == 1
    res = results[0]
    assert res.query == "Koregaon Park, Pune"
    assert res.coordinate.latitude == pytest.approx(18.536208, rel=1e-5)
    assert res.coordinate.longitude == pytest.approx(73.893974, rel=1e-5)
    assert res.bounding_box is not None
    assert res.bounding_box.min_latitude == 18.525
    assert res.osm_id == 987654321


@pytest.mark.asyncio
async def test_geocode_empty_query():
    """Test validation error on empty query string."""
    client = NominatimClient()
    with pytest.raises(ValueError):
        await client.geocode("   ")


@pytest.mark.asyncio
async def test_resolve_primary_location_not_found():
    """Test GeocodingError when location is unknown."""
    mock_http = AsyncMock(spec=AsyncHttpClient)
    mock_http.get_json.return_value = []

    client = NominatimClient(http_client=mock_http)
    with pytest.raises(GeocodingError):
        await client.resolve_primary_location("NonExistentUnknownPlace999")
