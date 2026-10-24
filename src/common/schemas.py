"""Data schemas for geospatial entities, live weather, air quality, and user profiles.

Strict invariants:
1. Missing live observations remain None. They are NEVER imputed or defaulted.
2. Proximity indicators to roads, industry, water, and green cover are qualitative tags/features.
3. Coordinates use WGS-84 decimal degrees.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.common.coordinates import LatLon, BoundingBox


class PersonalHealthProfile(BaseModel):
    """Personal sensitivities that tighten environmental thresholds."""
    has_asthma: bool = False
    is_senior: bool = False
    is_child: bool = False
    is_pregnant: bool = False
    notes: Optional[str] = None


class GeocodingResult(BaseModel):
    """Geocoding outcome from Nominatim."""
    query: str
    display_name: str
    coordinate: LatLon
    bounding_box: Optional[BoundingBox] = None
    osm_type: Optional[str] = None
    osm_id: Optional[int] = None
    place_rank: Optional[int] = None
    category: Optional[str] = None


class NearbyGeographicFeature(BaseModel):
    """Qualitative geographic context near a candidate venue (road, water body, industry, green cover)."""
    feature_type: str = Field(..., description="e.g. 'motorway', 'industrial', 'water_body', 'park_canopy'")
    name: Optional[str] = None
    distance_meters: float
    bearing_degrees: float = Field(..., description="Bearing from feature to venue in degrees [0, 360)")
    tags: Dict[str, Any] = Field(default_factory=dict)


class PlaceCandidate(BaseModel):
    """A candidate outdoor location discovered via OpenStreetMap."""
    id: str
    name: str
    place_type: str = Field(..., description="e.g. 'park', 'pitch', 'track', 'golf_course', 'nature_reserve'")
    coordinate: LatLon
    distance_from_anchor_km: Optional[float] = None
    osm_tags: Dict[str, Any] = Field(default_factory=dict)
    nearby_context: List[NearbyGeographicFeature] = Field(
        default_factory=list,
        description="Qualitative proximity indicators (highways, industrial, water bodies, parks)"
    )


class HourlyWeatherObservation(BaseModel):
    """Live weather observation or forecast for a specific hour.

    INVARIANT: Missing API measurements remain None. Never replace with invented defaults.
    """
    time: datetime
    temperature_2m: Optional[float] = Field(None, description="Air temperature at 2m (°C)")
    relative_humidity_2m: Optional[float] = Field(None, description="Relative humidity (%)")
    precipitation_probability: Optional[float] = Field(None, description="Precipitation probability (0-100%)")
    precipitation: Optional[float] = Field(None, description="Total precipitation (mm)")
    wind_speed_10m: Optional[float] = Field(None, description="Wind speed at 10m (km/h)")
    wind_direction_10m: Optional[float] = Field(None, description="Wind direction (degrees from which wind is blowing)")
    uv_index: Optional[float] = Field(None, description="UV Index")
    cloud_cover: Optional[float] = Field(None, description="Total cloud cover (%)")
    weather_code: Optional[int] = Field(None, description="WMO weather interpretation code")


class HourlyAQIObservation(BaseModel):
    """Live air-quality observation or forecast for a specific hour.

    INVARIANT: Missing API measurements remain None. Never replace with invented defaults.
    """
    time: datetime
    pm2_5: Optional[float] = Field(None, description="Particulate Matter PM2.5 (µg/m³)")
    pm10: Optional[float] = Field(None, description="Particulate Matter PM10 (µg/m³)")
    us_aqi: Optional[int] = Field(None, description="United States Air Quality Index (0-500)")
    european_aqi: Optional[int] = Field(None, description="European Air Quality Index")


class EnvironmentalProfile(BaseModel):
    """Combined environmental data for a single location across an hourly window."""
    coordinate: LatLon
    weather: List[HourlyWeatherObservation] = Field(default_factory=list)
    air_quality: List[HourlyAQIObservation] = Field(default_factory=list)


class ParsedQuery(BaseModel):
    """Structured representation of a natural language user query."""
    raw_query: str
    activity: Optional[str] = None
    spatial_anchor: Optional[str] = None
    search_radius_km: float = 10.0
    time_window_description: Optional[str] = None
    resolved_time_window: Optional[Any] = None
    health_profile: PersonalHealthProfile = Field(default_factory=PersonalHealthProfile)
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
