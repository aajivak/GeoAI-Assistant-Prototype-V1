
"""Data schemas for GeoAI geospatial and environmental entities.

Invariants:
1. Missing environmental observations remain None.
2. Geographic proximity features are qualitative indicators.
3. Coordinates use WGS-84 decimal degrees.
4. Environmental timestamps must be timezone-aware.
5. Time windows are validated and timezone-aware.
6. Existing public field names remain backward-compatible.
"""

from datetime import datetime
from math import isfinite
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from src.common.coordinates import (
    LatLon,
    BoundingBox,
)

from src.common.temporal import TimeWindow


class PersonalHealthProfile(BaseModel):
    """Personal sensitivities affecting environmental thresholds."""

    has_asthma: bool = False
    is_senior: bool = False
    is_child: bool = False
    is_pregnant: bool = False
    notes: Optional[str] = None


class GeocodingResult(BaseModel):
    """Geographic result returned by Nominatim."""

    query: str
    display_name: str
    coordinate: LatLon
    bounding_box: Optional[BoundingBox] = None
    osm_type: Optional[str] = None
    osm_id: Optional[int] = None
    place_rank: Optional[int] = Field(
        default=None,
        ge=0,
    )
    category: Optional[str] = None


class NearbyGeographicFeature(BaseModel):
    """Qualitative feature near an outdoor candidate."""

    feature_type: str = Field(
        ...,
        min_length=1,
        description=(
            "Examples: motorway, industrial, "
            "water_body, park_canopy."
        ),
    )

    name: Optional[str] = None

    distance_meters: float = Field(
        ...,
        ge=0,
        description="Distance from feature to candidate in meters.",
    )

    bearing_degrees: float = Field(
        ...,
        ge=0,
        lt=360,
        description=(
            "Bearing from feature to venue in degrees."
        ),
    )

    tags: Dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator(
        "distance_meters",
        "bearing_degrees",
    )
    @classmethod
    def validate_finite_geographic_values(
        cls,
        value: float,
    ) -> float:
        if not isfinite(value):
            raise ValueError(
                "Geographic measurements must be finite."
            )

        return value


class PlaceCandidate(BaseModel):
    """Outdoor venue discovered through geographic services."""

    id: str = Field(
        ...,
        min_length=1,
    )

    name: str = Field(
        ...,
        min_length=1,
    )

    place_type: str = Field(
        ...,
        min_length=1,
        description=(
            "Examples: park, pitch, track, "
            "golf_course, nature_reserve."
        ),
    )

    coordinate: LatLon

    distance_from_anchor_km: Optional[float] = Field(
        default=None,
        ge=0,
    )

    osm_tags: Dict[str, Any] = Field(
        default_factory=dict,
    )

    nearby_context: List[
        NearbyGeographicFeature
    ] = Field(
        default_factory=list,
        description=(
            "Qualitative proximity indicators "
            "for roads, industry, water and green cover."
        ),
    )

    @field_validator("distance_from_anchor_km")
    @classmethod
    def validate_candidate_distance(
        cls,
        value: Optional[float],
    ) -> Optional[float]:
        if value is not None and not isfinite(value):
            raise ValueError(
                "Candidate distance must be finite."
            )

        return value


class HourlyWeatherObservation(BaseModel):
    """Weather observation or forecast for a specific hour.

    Missing API measurements remain None.
    """

    time: datetime

    temperature_2m: Optional[float] = Field(
        default=None,
        description="Air temperature at 2 meters (°C).",
    )

    relative_humidity_2m: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Relative humidity (%).",
    )

    precipitation_probability: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Precipitation probability (%).",
    )

    precipitation: Optional[float] = Field(
        default=None,
        ge=0,
        description="Precipitation amount (mm).",
    )

    wind_speed_10m: Optional[float] = Field(
        default=None,
        ge=0,
        description="Wind speed at 10 meters (km/h).",
    )

    wind_direction_10m: Optional[float] = Field(
        default=None,
        ge=0,
        lt=360,
        description=(
            "Meteorological wind direction in degrees."
        ),
    )

    uv_index: Optional[float] = Field(
        default=None,
        description="UV Index.",
    )

    cloud_cover: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Total cloud cover (%).",
    )

    weather_code: Optional[int] = Field(
        default=None,
        description="WMO weather interpretation code.",
    )

    @field_validator("time")
    @classmethod
    def validate_aware_time(
        cls,
        value: datetime,
    ) -> datetime:
        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "Weather timestamps must be timezone-aware."
            )

        return value

    @field_validator(
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation_probability",
        "precipitation",
        "wind_speed_10m",
        "wind_direction_10m",
        "uv_index",
        "cloud_cover",
    )
    @classmethod
    def validate_finite_weather(
        cls,
        value: Optional[float],
    ) -> Optional[float]:
        if value is not None and not isfinite(value):
            raise ValueError(
                "Weather measurements must be finite."
            )

        return value


class HourlyAQIObservation(BaseModel):
    """Air-quality observation or forecast for a specific hour.

    Missing API measurements remain None.
    """

    time: datetime

    pm2_5: Optional[float] = Field(
        default=None,
        ge=0,
        description="PM2.5 concentration (µg/m³).",
    )

    pm10: Optional[float] = Field(
        default=None,
        ge=0,
        description="PM10 concentration (µg/m³).",
    )

    us_aqi: Optional[int] = Field(
        default=None,
        ge=0,
        description="United States Air Quality Index.",
    )

    european_aqi: Optional[int] = Field(
        default=None,
        ge=0,
        description="European Air Quality Index.",
    )

    @field_validator("time")
    @classmethod
    def validate_aware_time(
        cls,
        value: datetime,
    ) -> datetime:
        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "AQI timestamps must be timezone-aware."
            )

        return value

    @field_validator(
        "pm2_5",
        "pm10",
    )
    @classmethod
    def validate_finite_aqi(
        cls,
        value: Optional[float],
    ) -> Optional[float]:
        if value is not None and not isfinite(value):
            raise ValueError(
                "AQI measurements must be finite."
            )

        return value


class EnvironmentalProfile(BaseModel):
    """Weather and air quality for a geographic coordinate."""

    coordinate: LatLon

    weather: List[
        HourlyWeatherObservation
    ] = Field(default_factory=list)

    air_quality: List[
        HourlyAQIObservation
    ] = Field(default_factory=list)


class ParsedQuery(BaseModel):
    """Structured representation of a natural-language query."""

    raw_query: str = Field(
        ...,
        min_length=1,
    )

    activity: Optional[str] = None

    spatial_anchor: Optional[str] = None

    search_radius_km: float = Field(
        default=10.0,
        gt=0,
        le=100,
    )

    time_window_description: Optional[str] = None

    resolved_time_window: Optional[
        TimeWindow
    ] = None

    health_profile: PersonalHealthProfile = Field(
        default_factory=PersonalHealthProfile,
    )

    needs_clarification: bool = False

    clarification_question: Optional[str] = None

    @field_validator("search_radius_km")
    @classmethod
    def validate_finite_radius(
        cls,
        value: float,
    ) -> float:
        if not isfinite(value):
            raise ValueError(
                "Search radius must be finite."
            )

        return value
