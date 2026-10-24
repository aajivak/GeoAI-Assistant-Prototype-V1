"""Activity-specific criteria weights, scoring curves, and personal profile sensitivity rules.

DISCLAIMER:
All scoring thresholds, weights, and sensitivity rules are configurable engineering heuristics.
They are NOT medically certified safety guarantees or official health advice.
"""
from typing import Dict
from pydantic import BaseModel


class ActivityCriteriaWeights(BaseModel):
    """Weights for the five evaluation dimensions (sum normalized to 1.0)."""
    weight_temperature: float = 0.25
    weight_precipitation: float = 0.25
    weight_wind: float = 0.20
    weight_aqi: float = 0.20
    weight_daylight: float = 0.10


# Tailored weights for all 10 activities
ACTIVITY_WEIGHTS: Dict[str, ActivityCriteriaWeights] = {
    "running": ActivityCriteriaWeights(
        weight_temperature=0.30,
        weight_precipitation=0.25,
        weight_wind=0.10,
        weight_aqi=0.30,
        weight_daylight=0.05,
    ),
    "cycling": ActivityCriteriaWeights(
        weight_temperature=0.25,
        weight_precipitation=0.25,
        weight_wind=0.25,  # High drag & gust sensitivity
        weight_aqi=0.20,
        weight_daylight=0.05,
    ),
    "golf": ActivityCriteriaWeights(
        weight_temperature=0.20,
        weight_precipitation=0.30,  # Rain stops play
        weight_wind=0.25,           # Ball flight affected
        weight_aqi=0.10,
        weight_daylight=0.15,
    ),
    "cricket": ActivityCriteriaWeights(
        weight_temperature=0.20,
        weight_precipitation=0.35,  # Pitch unplayable if wet
        weight_wind=0.15,
        weight_aqi=0.15,
        weight_daylight=0.15,
    ),
    "football": ActivityCriteriaWeights(
        weight_temperature=0.25,
        weight_precipitation=0.25,
        weight_wind=0.15,
        weight_aqi=0.25,
        weight_daylight=0.10,
    ),
    "tennis": ActivityCriteriaWeights(
        weight_temperature=0.20,
        weight_precipitation=0.35,  # Slick court hazard
        weight_wind=0.25,           # Wind ruins ball trajectory
        weight_aqi=0.10,
        weight_daylight=0.10,
    ),
    "hiking": ActivityCriteriaWeights(
        weight_temperature=0.25,
        weight_precipitation=0.30,  # Slip hazards on trail
        weight_wind=0.15,
        weight_aqi=0.15,
        weight_daylight=0.15,
    ),
    "picnics": ActivityCriteriaWeights(
        weight_temperature=0.30,
        weight_precipitation=0.30,
        weight_wind=0.20,  # Blankets/food blow away
        weight_aqi=0.10,
        weight_daylight=0.10,
    ),
    "outdoor_gatherings": ActivityCriteriaWeights(
        weight_temperature=0.30,
        weight_precipitation=0.30,
        weight_wind=0.20,  # Canopies/tents vulnerable
        weight_aqi=0.10,
        weight_daylight=0.10,
    ),
    "photography": ActivityCriteriaWeights(
        weight_temperature=0.15,
        weight_precipitation=0.15,
        weight_wind=0.10,
        weight_aqi=0.30,  # Haze / PM2.5 severely affects contrast & horizon visibility
        weight_daylight=0.30,  # Daylight / golden hour critical
    ),
}
