"""Configuration package for GeoAI Outdoor Activity Recommendation Assistant."""
from config.settings import settings
from config.activity_definitions import SUPPORTED_ACTIVITIES, ActivityDefinition

__all__ = ["settings", "SUPPORTED_ACTIVITIES", "ActivityDefinition"]
