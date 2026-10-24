"""Unit tests for Overpass venue discovery and context enrichment using mocked HTTP responses."""
import pytest
from unittest.mock import AsyncMock

from src.common.coordinates import LatLon
from src.common.http_client import AsyncHttpClient
from src.data_ingestion.osm_places import OverpassClient


MOCK_OVERPASS_VENUES = {
    "version": 0.6,
    "generator": "Overpass API",
    "elements": [
        {
            "type": "way",
            "id": 111111,
            "center": {"lat": 18.5380, "lon": 73.8960},
            "tags": {
                "name": "Osho Teerth Park",
                "leisure": "park",
            },
        },
        {
            "type": "node",
            "id": 222222,
            "lat": 18.5310,
            "lon": 73.8910,
            "tags": {
                "name": "Koregaon Park Jogging Track",
                "leisure": "track",
                "surface": "tartan",
            },
        },
        {
            "type": "node",
            "id": 333333,
            "lat": 19.9999,  # Far away coordinate to test distance filtering
            "lon": 75.0000,
            "tags": {
                "name": "Distant Park",
                "leisure": "park",
            },
        },
    ],
}

MOCK_OVERPASS_CONTEXT = {
    "version": 0.6,
    "elements": [
        {
            "type": "way",
            "id": 888888,
            "center": {"lat": 18.5390, "lon": 73.8970},
            "tags": {
                "name": "Koregaon Park Main Road",
                "highway": "primary",
            },
        },
        {
            "type": "relation",
            "id": 999999,
            "center": {"lat": 18.5400, "lon": 73.8950},
            "tags": {
                "name": "Mula Mutha River",
                "waterway": "river",
            },
        },
    ],
}


@pytest.mark.asyncio
async def test_discover_venues_filtering():
    """Test venue discovery, name extraction, and radius filtering."""
    mock_http = AsyncMock(spec=AsyncHttpClient)
    mock_http.post_overpass.return_value = MOCK_OVERPASS_VENUES

    client = OverpassClient(http_client=mock_http)
    center = LatLon(latitude=18.5362, longitude=73.8940)

    candidates = await client.discover_venues(center=center, activity_id="running", radius_km=5.0)

    # Distant park (19.99, 75.0) must be filtered out!
    assert len(candidates) == 2
    names = [c.name for c in candidates]
    assert "Osho Teerth Park" in names
    assert "Koregaon Park Jogging Track" in names
    assert "Distant Park" not in names


@pytest.mark.asyncio
async def test_enrich_with_geographic_context():
    """Test that qualitative context features (roads, rivers) are attached to candidates."""
    mock_http = AsyncMock(spec=AsyncHttpClient)
    mock_http.post_overpass.side_effect = [MOCK_OVERPASS_VENUES, MOCK_OVERPASS_CONTEXT]

    client = OverpassClient(http_client=mock_http)
    center = LatLon(latitude=18.5362, longitude=73.8940)

    candidates = await client.discover_venues(center=center, activity_id="running", radius_km=5.0)
    enriched = await client.enrich_with_geographic_context(candidates, center=center, radius_km=5.0)

    assert len(enriched) == 2
    for cand in enriched:
        assert len(cand.nearby_context) > 0
        feature_types = [f.feature_type for f in cand.nearby_context]
        assert any("road" in ft or "water" in ft for ft in feature_types)
        for feature in cand.nearby_context:
            assert feature.distance_meters >= 0.0
            assert 0.0 <= feature.bearing_degrees <= 360.0
