"""Domain definitions for all supported outdoor activities and their venue tags.

The system supports at least ten outdoor activities:
1. Running or jogging
2. Cycling (leisure or commute)
3. Golf
4. Cricket (recreational match or practice)
5. Outdoor football or five-a-side
6. Outdoor tennis
7. Hiking or a nature walk
8. Picnic or park outing
9. Outdoor event or gathering (wedding, fair, etc.)
10. Outdoor photography session
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ActivityDefinition(BaseModel):
    """Specification of an outdoor activity and its geospatial context requirements."""
    id: str
    display_name: str
    category: str
    description: str
    osm_leisure_tags: List[str] = Field(default_factory=list)
    osm_sport_tags: List[str] = Field(default_factory=list)
    osm_highway_tags: List[str] = Field(default_factory=list)
    osm_tourism_tags: List[str] = Field(default_factory=list)
    ideal_temp_min: float
    ideal_temp_max: float
    rain_tolerant: bool = False
    wind_sensitive: bool = True
    daylight_required: bool = True


SUPPORTED_ACTIVITIES: Dict[str, ActivityDefinition] = {
    "running": ActivityDefinition(
        id="running",
        display_name="Running or Jogging",
        category="fitness",
        description="Continuous running, trail running, or jogging along paths, tracks, or parks.",
        osm_leisure_tags=["park", "track", "nature_reserve"],
        osm_highway_tags=["footway", "pedestrian", "path"],
        ideal_temp_min=12.0,
        ideal_temp_max=22.0,
        rain_tolerant=False,
        wind_sensitive=False,
        daylight_required=False,
    ),
    "cycling": ActivityDefinition(
        id="cycling",
        display_name="Cycling (Leisure or Commute)",
        category="fitness",
        description="Road cycling, commuting, or recreational trail riding.",
        osm_leisure_tags=["park"],
        osm_highway_tags=["cycleway", "path", "residential", "secondary"],
        ideal_temp_min=15.0,
        ideal_temp_max=25.0,
        rain_tolerant=False,
        wind_sensitive=True,
        daylight_required=True,
    ),
    "golf": ActivityDefinition(
        id="golf",
        display_name="Golf",
        category="sports",
        description="Playing 9 or 18 holes on a golf course or driving range practice.",
        osm_leisure_tags=["golf_course"],
        ideal_temp_min=18.0,
        ideal_temp_max=26.0,
        rain_tolerant=False,
        wind_sensitive=True,
        daylight_required=True,
    ),
    "cricket": ActivityDefinition(
        id="cricket",
        display_name="Cricket (Recreational Match or Practice)",
        category="sports",
        description="Cricket net practice or recreational open-ground match.",
        osm_leisure_tags=["pitch", "sports_centre"],
        osm_sport_tags=["cricket"],
        ideal_temp_min=18.0,
        ideal_temp_max=28.0,
        rain_tolerant=False,
        wind_sensitive=False,
        daylight_required=True,
    ),
    "football": ActivityDefinition(
        id="football",
        display_name="Outdoor Football or Five-a-side",
        category="sports",
        description="Recreational soccer, turf football, or five-a-side matches.",
        osm_leisure_tags=["pitch", "sports_centre"],
        osm_sport_tags=["soccer"],
        ideal_temp_min=14.0,
        ideal_temp_max=26.0,
        rain_tolerant=True,  # Turf can handle light drizzle
        wind_sensitive=False,
        daylight_required=False,  # Many turfs have floodlights
    ),
    "tennis": ActivityDefinition(
        id="tennis",
        display_name="Outdoor Tennis",
        category="sports",
        description="Tennis singles or doubles on open-air clay, hard, or grass courts.",
        osm_leisure_tags=["pitch", "sports_centre"],
        osm_sport_tags=["tennis"],
        ideal_temp_min=16.0,
        ideal_temp_max=26.0,
        rain_tolerant=False,
        wind_sensitive=True,  # Ball trajectory affected by wind
        daylight_required=False,
    ),
    "hiking": ActivityDefinition(
        id="hiking",
        display_name="Hiking or Nature Walk",
        category="recreation",
        description="Walking on natural trails, hill climbs, forest reserves, or ridge walks.",
        osm_leisure_tags=["nature_reserve", "park"],
        osm_highway_tags=["path", "track"],
        osm_tourism_tags=["viewpoint"],
        ideal_temp_min=14.0,
        ideal_temp_max=23.0,
        rain_tolerant=False,  # Wet trails are hazardous
        wind_sensitive=False,
        daylight_required=True,
    ),
    "picnics": ActivityDefinition(
        id="picnics",
        display_name="Picnic or Park Outing",
        category="leisure",
        description="Relaxed outdoor leisure, picnic blanket sitting, family park gatherings.",
        osm_leisure_tags=["park", "garden"],
        osm_tourism_tags=["picnic_site"],
        ideal_temp_min=20.0,
        ideal_temp_max=27.0,
        rain_tolerant=False,
        wind_sensitive=True,  # Food, blankets, tents disturbed by high wind
        daylight_required=True,
    ),
    "outdoor_gatherings": ActivityDefinition(
        id="outdoor_gatherings",
        display_name="Outdoor Event or Gathering",
        category="events",
        description="Outdoor birthday celebrations, fairs, ceremonies, or open-air lawn events.",
        osm_leisure_tags=["park", "garden"],
        ideal_temp_min=20.0,
        ideal_temp_max=27.0,
        rain_tolerant=False,
        wind_sensitive=True,
        daylight_required=False,
    ),
    "photography": ActivityDefinition(
        id="photography",
        display_name="Outdoor Photography Session",
        category="arts",
        description="Landscape, portrait, street, or architectural photography in scenic locations.",
        osm_leisure_tags=["park", "garden", "nature_reserve"],
        osm_tourism_tags=["viewpoint", "attraction"],
        ideal_temp_min=15.0,
        ideal_temp_max=28.0,
        rain_tolerant=True,  # Dramatic weather can be photogenic
        wind_sensitive=False,
        daylight_required=True,  # Golden hour / daylight critical
    ),
}
