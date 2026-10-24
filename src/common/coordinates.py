"""Geospatial coordinate representations, Haversine distance, and bearing calculations.

Adheres to Avani/TerraEngine GIS principles (WGS-84 coordinate system).
"""
import math
from typing import Tuple
from pydantic import BaseModel, Field, field_validator


class LatLon(BaseModel):
    """WGS-84 Geographic Coordinate."""
    latitude: float = Field(..., description="Latitude in decimal degrees [-90.0, 90.0]")
    longitude: float = Field(..., description="Longitude in decimal degrees [-180.0, 180.0]")

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        if not (-90.0 <= v <= 90.0):
            raise ValueError(f"Latitude must be between -90 and 90, got {v}")
        return round(v, 6)

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        if not (-180.0 <= v <= 180.0):
            raise ValueError(f"Longitude must be between -180 and 180, got {v}")
        return round(v, 6)

    def as_tuple(self) -> Tuple[float, float]:
        return (self.latitude, self.longitude)


class BoundingBox(BaseModel):
    """Geographic bounding box for spatial search queries."""
    min_latitude: float
    min_longitude: float
    max_latitude: float
    max_longitude: float

    @classmethod
    def from_center_radius(cls, center: LatLon, radius_km: float) -> "BoundingBox":
        """Compute bounding box around a center coordinate given a radius in kilometers."""
        # 1 deg latitude ~ 111.0 km
        lat_delta = radius_km / 111.0
        # 1 deg longitude ~ 111.0 * cos(lat) km
        cos_lat = math.cos(math.radians(center.latitude))
        lon_delta = radius_km / (111.0 * max(0.01, abs(cos_lat)))

        return cls(
            min_latitude=max(-90.0, center.latitude - lat_delta),
            min_longitude=max(-180.0, center.longitude - lon_delta),
            max_latitude=min(90.0, center.latitude + lat_delta),
            max_longitude=min(180.0, center.longitude + lon_delta),
        )

    def to_overpass_bbox(self) -> str:
        """Overpass QL bbox format: (min_lat, min_lon, max_lat, max_lon)."""
        return f"{self.min_latitude:.6f},{self.min_longitude:.6f},{self.max_latitude:.6f},{self.max_longitude:.6f}"


def haversine_distance(p1: LatLon, p2: LatLon) -> float:
    """Calculate the great-circle distance between two points in kilometers.

    Uses spherical Earth radius R = 6371.0088 km.
    """
    R = 6371.0088  # Mean Earth radius in km

    phi1 = math.radians(p1.latitude)
    phi2 = math.radians(p2.latitude)
    delta_phi = math.radians(p2.latitude - p1.latitude)
    delta_lambda = math.radians(p2.longitude - p1.longitude)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return round(R * c, 3)


def calculate_bearing(origin: LatLon, destination: LatLon) -> float:
    """Calculate the initial compass bearing from origin to destination in degrees [0, 360).

    0° = North, 90° = East, 180° = South, 270° = West.
    """
    phi1 = math.radians(origin.latitude)
    phi2 = math.radians(destination.latitude)
    delta_lambda = math.radians(destination.longitude - origin.longitude)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = (math.cos(phi1) * math.sin(phi2) -
         math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda))

    initial_bearing = math.atan2(y, x)
    # Convert from radians [-pi, pi] to degrees [0, 360)
    compass_bearing = (math.degrees(initial_bearing) + 360.0) % 360.0

    return round(compass_bearing, 2)
