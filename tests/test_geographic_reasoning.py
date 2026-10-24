"""Unit tests for geographic context analysis and wind dispersion reasoning."""
import pytest

from src.common.coordinates import LatLon
from src.common.schemas import NearbyGeographicFeature, PlaceCandidate
from src.spatial_analysis.wind_dispersion import analyze_geographic_context


def test_green_canopy_and_water_body():
    """Verify detection of park canopy and nearby water bodies."""
    cand = PlaceCandidate(
        id="way/123",
        name="Lakeside Botanical Park",
        place_type="park",
        coordinate=LatLon(latitude=18.53, longitude=73.89),
        nearby_context=[
            NearbyGeographicFeature(
                feature_type="water_body",
                name="Powai Lake",
                distance_meters=350.0,
                bearing_degrees=120.0,
            )
        ],
    )

    report = analyze_geographic_context(cand, wind_direction_deg=180.0, wind_speed_kmh=10.0)
    assert report.has_green_canopy is True
    assert report.is_near_water_body is True
    indicator_types = [ind.indicator_type for ind in report.indicators]
    assert "green_canopy" in indicator_types
    assert "water_buffer" in indicator_types


def test_wind_shift_downwind_hazard():
    """Verify that a venue downwind of an expressway is flagged, and when wind shifts, the hazard clears.

    Road is directly South of venue (bearing from road to venue = 0° North).
    Distance = 500 meters.
    """
    cand = PlaceCandidate(
        id="way/456",
        name="Sports Ground",
        place_type="pitch",
        coordinate=LatLon(latitude=18.54, longitude=73.89),
        nearby_context=[
            NearbyGeographicFeature(
                feature_type="road_motorway",
                name="Eastern Express Highway",
                distance_meters=500.0,
                bearing_degrees=0.0,  # Road lies South, bearing from Road to Venue is 0° (North)
            )
        ],
    )

    # 1. Morning Wind from South (180°):
    # Wind blows from 180° towards 0° (North) -> directly toward the venue!
    morning_report = analyze_geographic_context(cand, wind_direction_deg=180.0, wind_speed_kmh=12.0)
    assert morning_report.is_downwind_of_road is True
    assert morning_report.suggested_qualitative_penalty > 0.0
    assert any("downwind_road" in ind.indicator_type for ind in morning_report.indicators)

    # 2. Evening Wind shifts from North (0°):
    # Wind blows from 0° towards 180° (South) -> blowing AWAY from venue toward the highway!
    evening_report = analyze_geographic_context(cand, wind_direction_deg=0.0, wind_speed_kmh=12.0)
    assert evening_report.is_downwind_of_road is False
    assert evening_report.suggested_qualitative_penalty == 0.0


def test_missing_wind_direction_preserved():
    """Verify that missing wind direction is reported as unknown and no downwind hazard is invented."""
    cand = PlaceCandidate(
        id="way/789",
        name="Community Park",
        place_type="park",
        coordinate=LatLon(latitude=18.50, longitude=73.80),
        nearby_context=[
            NearbyGeographicFeature(
                feature_type="road_primary",
                name="Main Arterial",
                distance_meters=400.0,
                bearing_degrees=90.0,
            )
        ],
    )

    report = analyze_geographic_context(cand, wind_direction_deg=None, wind_speed_kmh=None)
    assert report.is_downwind_of_road is False
    assert any(ind.indicator_type == "wind_direction_missing" for ind in report.indicators)
