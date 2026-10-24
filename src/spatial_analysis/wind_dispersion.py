
"""Qualitative geographic exposure analysis for outdoor venues.

Analyzes:
- Road and industrial proximity
- Potential downwind exposure
- Water-body proximity
- Green-space indicators
- Uncertainty in wind and OSM geometry

Important:
This is a qualitative geographic heuristic, NOT a
physical atmospheric dispersion simulation.

No numerical pollution concentrations or temperature
offsets are invented.
"""

from math import isfinite
from typing import List, Optional

from pydantic import BaseModel, Field

from src.common.schemas import PlaceCandidate


class GeographicIndicator(BaseModel):
    """A qualitative geographic indicator associated with a venue."""

    indicator_type: str = Field(
        ...,
        description=(
            "Examples: downwind_road, downwind_industry, "
            "water_buffer, green_canopy."
        ),
    )

    description: str

    severity_or_benefit: str = Field(
        ...,
        description="'advisory', 'warning', or 'benefit'.",
    )

    feature_name: Optional[str] = None

    distance_meters: Optional[float] = Field(
        default=None,
        ge=0,
    )


class GeographicContextReport(BaseModel):
    """Qualitative geographic analysis for one venue and hour."""

    venue_id: str
    venue_name: str

    has_green_canopy: bool = False
    is_near_water_body: bool = False

    is_downwind_of_road: bool = False
    is_downwind_of_industry: bool = False

    indicators: List[GeographicIndicator] = Field(
        default_factory=list
    )

    summary_advisory: str

    suggested_qualitative_penalty: float = Field(
        default=0.0,
        ge=0.0,
        le=30.0,
        description=(
            "Heuristic exposure penalty between 0 and 30 points."
        ),
    )


def _angular_difference(
    first: float,
    second: float,
) -> float:
    """Return the smallest difference between compass bearings."""

    return abs(
        (first - second + 180.0) % 360.0 - 180.0
    )


def _is_valid_wind(
    wind_direction_deg: Optional[float],
    wind_speed_kmh: Optional[float],
) -> bool:
    """Check whether wind data is usable for directional analysis."""

    if wind_direction_deg is None:
        return False

    if not isfinite(wind_direction_deg):
        return False

    if wind_speed_kmh is None:
        return False

    if not isfinite(wind_speed_kmh):
        return False

    return wind_speed_kmh >= 2.0


def analyze_geographic_context(
    candidate: PlaceCandidate,
    wind_direction_deg: Optional[float] = None,
    wind_speed_kmh: Optional[float] = None,
) -> GeographicContextReport:
    """Evaluate qualitative geographic context under forecast wind.

    Wind directions follow meteorological convention:
    - 0°: wind from north
    - 90°: wind from east
    - 180°: wind from south
    - 270°: wind from west

    Potential transport direction is wind direction + 180°.

    OSM feature distances are representative-point estimates,
    not exact shortest distances to road or polygon boundaries.
    """

    indicators: List[GeographicIndicator] = []
    summary_parts: List[str] = []

    has_green_canopy = False
    is_near_water = False

    is_downwind_road = False
    is_downwind_industry = False

    road_penalty = 0.0
    industry_penalty = 0.0

    # --------------------------------------------------
    # 1. Green-space evidence
    # --------------------------------------------------

    green_place_types = {
        "park",
        "garden",
        "nature_reserve",
        "forest",
        "wood",
    }

    if candidate.place_type in green_place_types:
        has_green_canopy = True

        indicators.append(
            GeographicIndicator(
                indicator_type="green_canopy",
                description=(
                    f"{candidate.name} is mapped as "
                    f"{candidate.place_type}. Vegetation may "
                    "provide shade and local comfort, but actual "
                    "tree-canopy coverage is unverified."
                ),
                severity_or_benefit="benefit",
                feature_name=candidate.name,
                distance_meters=0.0,
            )
        )

    # Nearby mapped green spaces are also relevant.
    if not has_green_canopy:
        for feature in candidate.nearby_context:
            if (
                feature.feature_type == "park_canopy"
                and feature.distance_meters <= 300.0
            ):
                has_green_canopy = True

                indicators.append(
                    GeographicIndicator(
                        indicator_type="green_canopy",
                        description=(
                            "Mapped green space is nearby. "
                            "Actual vegetation density and shade "
                            "have not been measured."
                        ),
                        severity_or_benefit="benefit",
                        feature_name=feature.name,
                        distance_meters=feature.distance_meters,
                    )
                )

                break

    # --------------------------------------------------
    # 2. Water-body proximity
    # --------------------------------------------------

    for feature in candidate.nearby_context:
        if (
            feature.feature_type == "water_body"
            and feature.distance_meters <= 600.0
        ):
            is_near_water = True

            indicators.append(
                GeographicIndicator(
                    indicator_type="water_buffer",
                    description=(
                        f"A mapped water body is approximately "
                        f"{feature.distance_meters:.0f} m away. "
                        "Any local cooling or breeze benefit is "
                        "possible but not quantified."
                    ),
                    severity_or_benefit="benefit",
                    feature_name=feature.name,
                    distance_meters=feature.distance_meters,
                )
            )

            break

    # --------------------------------------------------
    # 3. Wind availability and calm conditions
    # --------------------------------------------------

    wind_valid = _is_valid_wind(
        wind_direction_deg,
        wind_speed_kmh,
    )

    if wind_direction_deg is None or wind_speed_kmh is None:
        indicators.append(
            GeographicIndicator(
                indicator_type="wind_data_missing",
                description=(
                    "Wind direction or speed is unavailable. "
                    "Directional exposure cannot be assessed."
                ),
                severity_or_benefit="advisory",
            )
        )

    elif not wind_valid:
        indicators.append(
            GeographicIndicator(
                indicator_type="wind_direction_uncertain",
                description=(
                    "Wind is calm, invalid, or too weak for "
                    "reliable directional exposure assessment. "
                    "Nearby pollution sources may still matter."
                ),
                severity_or_benefit="advisory",
            )
        )

    # --------------------------------------------------
    # 4. Potential downwind exposure
    # --------------------------------------------------

    if wind_valid:
        plume_direction = (
            wind_direction_deg + 180.0
        ) % 360.0

        for feature in candidate.nearby_context:
            distance = feature.distance_meters

            if distance > 1500.0:
                continue

            angle_difference = _angular_difference(
                feature.bearing_degrees,
                plume_direction,
            )

            # Heuristic directional sector, not a plume model.
            if angle_difference > 40.0:
                continue

            is_road = (
                feature.feature_type.startswith("road_")
                or feature.feature_type == "road"
            )

            is_industry = (
                feature.feature_type == "industrial_zone"
            )

            if is_road:
                is_downwind_road = True

                candidate_penalty = (
                    15.0 if distance < 700.0 else 8.0
                )

                road_penalty = max(
                    road_penalty,
                    candidate_penalty,
                )

                indicators.append(
                    GeographicIndicator(
                        indicator_type="downwind_road",
                        description=(
                            f"Forecast wind is aligned with the "
                            f"direction from mapped road "
                            f"'{feature.name or 'unnamed'}' toward "
                            f"the venue (angular difference "
                            f"{angle_difference:.0f}°, approximate "
                            f"distance {distance:.0f} m). "
                            "This suggests possible traffic-related "
                            "exposure, not a confirmed pollution plume."
                        ),
                        severity_or_benefit=(
                            "warning"
                            if distance < 700.0
                            else "advisory"
                        ),
                        feature_name=feature.name,
                        distance_meters=distance,
                    )
                )

            elif is_industry:
                is_downwind_industry = True

                candidate_penalty = (
                    20.0 if distance < 800.0 else 10.0
                )

                industry_penalty = max(
                    industry_penalty,
                    candidate_penalty,
                )

                indicators.append(
                    GeographicIndicator(
                        indicator_type="downwind_industry",
                        description=(
                            f"Forecast wind is aligned with the "
                            f"direction from mapped industrial area "
                            f"'{feature.name or 'unnamed'}' toward "
                            f"the venue (angular difference "
                            f"{angle_difference:.0f}°, approximate "
                            f"distance {distance:.0f} m). "
                            "Industrial exposure is possible but "
                            "has not been directly measured."
                        ),
                        severity_or_benefit="warning",
                        feature_name=feature.name,
                        distance_meters=distance,
                    )
                )

    # --------------------------------------------------
    # 5. Combined qualitative penalty
    # --------------------------------------------------

    # Avoid stacking every road/industrial source.
    # Combine the strongest road and industrial indicators
    # while respecting the documented 30-point ceiling.
    penalty = min(
        30.0,
        road_penalty + industry_penalty,
    )

    # --------------------------------------------------
    # 6. Explain the context
    # --------------------------------------------------

    if is_downwind_road:
        summary_parts.append(
            "Potential downwind traffic exposure."
        )

    if is_downwind_industry:
        summary_parts.append(
            "Potential downwind industrial exposure."
        )

    if has_green_canopy:
        summary_parts.append(
            "Mapped green-space indicator present; "
            "actual tree cover unverified."
        )

    if is_near_water:
        summary_parts.append(
            "Mapped water body nearby."
        )

    if not wind_valid:
        summary_parts.append(
            "Directional exposure uncertain."
        )

    if not summary_parts:
        summary_parts.append(
            "No notable geographic exposure indicators "
            "identified from available mapped features."
        )

    return GeographicContextReport(
        venue_id=candidate.id,
        venue_name=candidate.name,
        has_green_canopy=has_green_canopy,
        is_near_water_body=is_near_water,
        is_downwind_of_road=is_downwind_road,
        is_downwind_of_industry=is_downwind_industry,
        indicators=indicators,
        summary_advisory=" ".join(summary_parts),
        suggested_qualitative_penalty=penalty,
    )
