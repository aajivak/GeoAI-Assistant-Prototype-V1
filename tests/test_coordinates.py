"""Unit tests for geographic coordinate handling, Haversine distances, and bearings."""
import pytest
from pydantic import ValidationError

from src.common.coordinates import (
    BoundingBox,
    LatLon,
    calculate_bearing,
    haversine_distance,
)


def test_latlon_valid():
    """Test valid coordinate creation."""
    coord = LatLon(latitude=18.5362, longitude=73.8940)
    assert coord.latitude == 18.5362
    assert coord.longitude == 73.8940
    assert coord.as_tuple() == (18.5362, 73.8940)


def test_latlon_bounds():
    """Test out-of-bounds coordinate validation."""
    with pytest.raises(ValidationError):
        LatLon(latitude=95.0, longitude=73.0)

    with pytest.raises(ValidationError):
        LatLon(latitude=18.0, longitude=-195.0)


def test_haversine_known_distance():
    """Test Haversine distance against known reference (Mumbai Gateway of India to Pune Shaniwar Wada ~120 km)."""
    mumbai_gateway = LatLon(latitude=18.9220, longitude=72.8347)
    pune_shaniwar_wada = LatLon(latitude=18.5196, longitude=73.8553)

    dist = haversine_distance(mumbai_gateway, pune_shaniwar_wada)
    # Geodesic distance is approximately 117-122 km
    assert 115.0 < dist < 125.0


def test_haversine_zero_distance():
    """Distance between a point and itself should be zero."""
    pt = LatLon(latitude=19.0760, longitude=72.8777)
    assert haversine_distance(pt, pt) == 0.0


def test_calculate_bearing_cardinal():
    """Test compass bearings for cardinal directions."""
    origin = LatLon(latitude=18.0, longitude=73.0)

    # Directly North
    north_pt = LatLon(latitude=19.0, longitude=73.0)
    bearing_north = calculate_bearing(origin, north_pt)
    assert abs(bearing_north - 0.0) < 0.1 or abs(bearing_north - 360.0) < 0.1

    # Directly East
    east_pt = LatLon(latitude=18.0, longitude=74.0)
    bearing_east = calculate_bearing(origin, east_pt)
    assert 88.0 < bearing_east < 92.0

    # Directly South
    south_pt = LatLon(latitude=17.0, longitude=73.0)
    bearing_south = calculate_bearing(origin, south_pt)
    assert 178.0 < bearing_south < 182.0

    # Directly West
    west_pt = LatLon(latitude=18.0, longitude=72.0)
    bearing_west = calculate_bearing(origin, west_pt)
    assert 268.0 < bearing_west < 272.0


def test_bounding_box_expansion():
    """Test bounding box expansion from center and radius."""
    center = LatLon(latitude=18.5204, longitude=73.8567)  # Pune
    bbox = BoundingBox.from_center_radius(center, radius_km=10.0)

    assert bbox.min_latitude < center.latitude < bbox.max_latitude
    assert bbox.min_longitude < center.longitude < bbox.max_longitude

    # Check that overpass bbox format is correct: (min_lat, min_lon, max_lat, max_lon)
    obbox = bbox.to_overpass_bbox()
    parts = [float(p) for p in obbox.split(",")]
    assert len(parts) == 4
    assert parts[0] == pytest.approx(bbox.min_latitude, rel=1e-4)
    assert parts[1] == pytest.approx(bbox.min_longitude, rel=1e-4)
