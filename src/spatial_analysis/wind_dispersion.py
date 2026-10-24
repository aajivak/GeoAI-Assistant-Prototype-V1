"""Geographic context analysis: road proximity, industrial sources, wind dispersion, water bodies, and green canopy.

INVARIANT:
All proximity and microclimate indicators are qualitative geographic factors.
Never invent precise numerical pollution (µg/m³) or temperature (°C) offsets.
"""
from typing import List, Optional
from pydantic import BaseModel, Field

from src.common.schemas import NearbyGeographicFeature, PlaceCandidate


class GeographicIndicator(BaseModel):
    """A qualitative geographic indicator associated with a venue."""
    indicator_type: str = Field(..., description="e.g. 'downwind_road', 'downwind_industry', 'water_buffer', 'green_canopy'")
    description: str
    severity_or_benefit: str = Field(..., description="'advisory', 'warning', 'benefit'")
    feature_name: Optional[str] = None
    distance_meters: Optional[float] = None


class GeographicContextReport(BaseModel):
    """Qualitative geographic and microclimatic analysis for a candidate venue at a given hour."""
    venue_id: str
    venue_name: str
    has_green_canopy: bool = False
    is_near_water_body: bool = False
    is_downwind_of_road: bool = False
    is_downwind_of_industry: bool = False
    indicators: List[GeographicIndicator] = Field(default_factory=list)
    summary_advisory: str
    suggested_qualitative_penalty: float = Field(
        0.0,
        description="Heuristic score reduction [0-30] for unfavorable geographic exposure (e.g. downwind expressways)."
    )


def analyze_geographic_context(
    candidate: PlaceCandidate,
    wind_direction_deg: Optional[float] = None,
    wind_speed_kmh: Optional[float] = None,
) -> GeographicContextReport:
    """Evaluate qualitative geographic indicators for a venue under specified wind conditions.

    Does NOT fabricate numerical adjustments. Produces qualitative indicators and explainable warnings.
    """
    indicators: List[GeographicIndicator] = []
    penalty = 0.0

    # 1. Green Canopy Check (Venue itself is a designated park, garden, nature reserve)
    has_green_canopy = candidate.place_type in ("park", "garden", "nature_reserve", "pitch")
    if has_green_canopy:
        indicators.append(
            GeographicIndicator(
                indicator_type="green_canopy",
                description=f"{candidate.name} is situated in a designated green space ({candidate.place_type}), providing natural vegetation buffering from urban heat and immediate traffic.",
                severity_or_benefit="benefit",
                feature_name=candidate.name,
                distance_meters=0.0,
            )
        )

    # 2. Water Body Proximity Check
    is_near_water = False
    for feat in candidate.nearby_context:
        if feat.feature_type == "water_body" and feat.distance_meters <= 600.0:
            is_near_water = True
            indicators.append(
                GeographicIndicator(
                    indicator_type="water_buffer",
                    description=f"Located {feat.distance_meters:.0f}m from {feat.name or 'a water body'}; open water may provide localized breeze and evaporative comfort.",
                    severity_or_benefit="benefit",
                    feature_name=feat.name,
                    distance_meters=feat.distance_meters,
                )
            )
            break

    # 3. Downwind Wind Dispersion Analysis relative to Roads and Industrial Zones
    is_downwind_road = False
    is_downwind_industry = False

    if wind_direction_deg is not None:
        # Meteorological wind direction: direction wind is blowing FROM
        # Direction wind is blowing TOWARDS is (wind_direction_deg + 180) % 360
        plume_direction = (wind_direction_deg + 180.0) % 360.0

        for feat in candidate.nearby_context:
            # Check angular offset between bearing (Feature -> Candidate) and plume direction
            # If plume direction aligns with the vector from Feature to Candidate, venue is downwind!
            bearing_from_source = feat.bearing_degrees
            angle_diff = abs(bearing_from_source - plume_direction)
            if angle_diff > 180.0:
                angle_diff = 360.0 - angle_diff

            # If within 40 degrees cone of dispersion and within 1500 meters
            if angle_diff <= 40.0 and feat.distance_meters <= 1500.0:
                if "road" in feat.feature_type:
                    is_downwind_road = True
                    severity = "warning" if feat.distance_meters < 700.0 else "advisory"
                    p_deduction = 15.0 if feat.distance_meters < 700.0 else 8.0
                    penalty = max(penalty, p_deduction)
                    indicators.append(
                        GeographicIndicator(
                            indicator_type="downwind_road",
                            description=(
                                f"Lies directly downwind ({angle_diff:.0f}° alignment) of {feat.name or 'arterial road'} "
                                f"({feat.distance_meters:.0f}m away). Wind from {wind_direction_deg:.0f}° carries vehicular emissions toward the venue."
                            ),
                            severity_or_benefit=severity,
                            feature_name=feat.name or feat.feature_type,
                            distance_meters=feat.distance_meters,
                        )
                    )
                elif feat.feature_type == "industrial_zone":
                    is_downwind_industry = True
                    p_deduction = 20.0 if feat.distance_meters < 800.0 else 10.0
                    penalty = max(penalty, p_deduction)
                    indicators.append(
                        GeographicIndicator(
                            indicator_type="downwind_industry",
                            description=(
                                f"Lies downwind ({angle_diff:.0f}° alignment) of industrial zone '{feat.name or 'unnamed'}' "
                                f"({feat.distance_meters:.0f}m away). Possible localized dust or industrial plume."
                            ),
                            severity_or_benefit="warning",
                            feature_name=feat.name or "Industrial Area",
                            distance_meters=feat.distance_meters,
                        )
                    )
    else:
        # Wind direction missing: report unknown without inventing
        indicators.append(
            GeographicIndicator(
                indicator_type="wind_direction_missing",
                description="Live wind direction measurement is unavailable; downwind plume exposure cannot be verified.",
                severity_or_benefit="advisory",
            )
        )

    # Compile qualitative summary
    summary_parts = []
    if is_downwind_road or is_downwind_industry:
        summary_parts.append("Downwind exposure to nearby transport/industrial sources detected.")
    if has_green_canopy:
        summary_parts.append("Beneficial green canopy buffer present.")
    if is_near_water:
        summary_parts.append("Water body nearby.")
    if not summary_parts:
        summary_parts.append("Neutral surrounding geographic context.")

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
