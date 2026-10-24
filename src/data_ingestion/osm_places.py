"""Venue and geographic feature discovery using OpenStreetMap Overpass API.

Discovers activity-specific venues (parks, pitches, trails, courts, golf courses)
and nearby qualitative geographic context features (major roads, industrial zones, water bodies, green canopy).
"""
import logging
from typing import Any, Dict, List, Optional

from config.activity_definitions import SUPPORTED_ACTIVITIES
from config.settings import settings
from src.common.coordinates import (
    BoundingBox,
    LatLon,
    calculate_bearing,
    haversine_distance,
)
from src.common.http_client import AsyncHttpClient
from src.common.schemas import NearbyGeographicFeature, PlaceCandidate

logger = logging.getLogger("geoai.osm_places")


class OverpassClient:
    """Overpass API client for outdoor venue discovery and context feature queries."""

    def __init__(self, http_client: Optional[AsyncHttpClient] = None):
        self.http_client = http_client or AsyncHttpClient()
        self.primary_url = settings.OVERPASS_BASE_URL
        self.fallback_url = settings.OVERPASS_FALLBACK_URL

    def build_venue_query(self, activity_id: str, bbox: BoundingBox, timeout_sec: int = 15) -> str:
        """Construct fast, optimized Overpass QL query tailored to the activity within a bounding box.

        Uses node and way queries (avoiding heavy recursive relation traversal to prevent 504 timeouts).
        """
        bbox_str = bbox.to_overpass_bbox()
        activity = SUPPORTED_ACTIVITIES.get(activity_id.lower())

        subqueries = []
        if activity:
            for leisure in activity.osm_leisure_tags:
                subqueries.append(f'node["leisure"="{leisure}"]({bbox_str});')
                subqueries.append(f'way["leisure"="{leisure}"]({bbox_str});')
            for sport in activity.osm_sport_tags:
                subqueries.append(f'node["sport"="{sport}"]({bbox_str});')
                subqueries.append(f'way["sport"="{sport}"]({bbox_str});')
            for hwy in activity.osm_highway_tags:
                subqueries.append(f'way["highway"="{hwy}"]({bbox_str});')
            for tour in activity.osm_tourism_tags:
                subqueries.append(f'node["tourism"="{tour}"]({bbox_str});')
                subqueries.append(f'way["tourism"="{tour}"]({bbox_str});')

        # Fallback to general recreational parks if no tags match
        if not subqueries:
            subqueries.append(f'node["leisure"="park"]({bbox_str});')
            subqueries.append(f'way["leisure"="park"]({bbox_str});')

        query_body = "\n  ".join(subqueries)
        return f"""[out:json][timeout:{timeout_sec}];
(
  {query_body}
);
out center tags 25;"""

    def build_context_query(self, bbox: BoundingBox, timeout_sec: int = 15) -> str:
        """Query qualitative geographic context features: expressways, industrial zones, and water bodies."""
        bbox_str = bbox.to_overpass_bbox()
        return f"""[out:json][timeout:{timeout_sec}];
(
  way["highway"~"^(motorway|trunk)$"]({bbox_str});
  way["landuse"="industrial"]({bbox_str});
  way["natural"="water"]({bbox_str});
  way["waterway"~"^(river|canal)$"]({bbox_str});
);
out center tags 20;"""

    async def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute query against primary Overpass endpoint with automatic fallback."""
        try:
            return await self.http_client.post_overpass(
                url=self.primary_url,
                query=query,
                cache_ttl_seconds=settings.OSM_PLACES_CACHE_TTL_SECONDS,
            )
        except Exception as e:
            logger.warning("Primary Overpass failed (%s). Attempting fallback: %s", e, self.fallback_url)
            return await self.http_client.post_overpass(
                url=self.fallback_url,
                query=query,
                cache_ttl_seconds=settings.OSM_PLACES_CACHE_TTL_SECONDS,
            )

    async def discover_venues(
        self,
        center: LatLon,
        activity_id: str,
        radius_km: float = 5.0,
        limit: int = 10,
    ) -> List[PlaceCandidate]:
        """Discover candidate venues for an activity within radius of center."""
        # Cap query radius to 5.0 km for live Overpass stability in dense cities
        effective_radius = min(radius_km, 5.0)
        bbox = BoundingBox.from_center_radius(center, effective_radius)
        venue_query = self.build_venue_query(activity_id, bbox)

        try:
            data = await self.execute_query(venue_query)
            elements = data.get("elements", [])
        except Exception as e:
            logger.warning("Overpass venue discovery failed (%s). Returning empty candidate list.", e)
            return []

        candidates: List[PlaceCandidate] = []

        for elem in elements:
            tags = elem.get("tags", {})
            name = tags.get("name")
            if not name:
                leisure = tags.get("leisure") or tags.get("sport") or tags.get("highway")
                if not leisure:
                    continue
                name = f"Unnamed {leisure.title()} Area"

            lat = elem.get("lat") or (elem.get("center", {}).get("lat"))
            lon = elem.get("lon") or (elem.get("center", {}).get("lon"))

            if lat is None or lon is None:
                continue

            coord = LatLon(latitude=lat, longitude=lon)
            distance = haversine_distance(center, coord)

            if distance > effective_radius:
                continue

            place_type = (
                tags.get("leisure")
                or tags.get("sport")
                or tags.get("tourism")
                or tags.get("highway")
                or "recreation_area"
            )

            elem_id = f"{elem.get('type', 'node')}/{elem.get('id')}"

            candidates.append(
                PlaceCandidate(
                    id=elem_id,
                    name=name,
                    place_type=place_type,
                    coordinate=coord,
                    distance_from_anchor_km=distance,
                    osm_tags=tags,
                )
            )

        candidates.sort(key=lambda c: c.distance_from_anchor_km or 999.0)
        return candidates[:limit]

    async def enrich_with_geographic_context(
        self,
        candidates: List[PlaceCandidate],
        center: LatLon,
        radius_km: float = 5.0,
    ) -> List[PlaceCandidate]:
        """Attach qualitative proximity context (roads, industry, water) to candidate venues."""
        if not candidates:
            return []

        effective_radius = min(radius_km, 5.0)
        bbox = BoundingBox.from_center_radius(center, effective_radius)
        context_query = self.build_context_query(bbox)

        try:
            data = await self.execute_query(context_query)
            elements = data.get("elements", [])
        except Exception as e:
            logger.warning("Could not fetch contextual features (%s). Proceeding with base candidates.", e)
            return candidates

        context_features: List[tuple[str, Optional[str], LatLon, Dict[str, Any]]] = []
        for elem in elements:
            tags = elem.get("tags", {})
            lat = elem.get("lat") or (elem.get("center", {}).get("lat"))
            lon = elem.get("lon") or (elem.get("center", {}).get("lon"))
            if lat is None or lon is None:
                continue

            feature_type = "unknown"
            if "highway" in tags:
                feature_type = f"road_{tags['highway']}"
            elif tags.get("landuse") == "industrial":
                feature_type = "industrial_zone"
            elif "natural" in tags or "waterway" in tags:
                feature_type = "water_body"

            name = tags.get("name")
            context_features.append((feature_type, name, LatLon(latitude=lat, longitude=lon), tags))

        # Associate nearby features within 2.0 km of each candidate
        for candidate in candidates:
            nearby: List[NearbyGeographicFeature] = []
            for f_type, f_name, f_coord, f_tags in context_features:
                dist_km = haversine_distance(f_coord, candidate.coordinate)
                dist_meters = dist_km * 1000.0
                if dist_meters <= 2000.0:
                    bearing = calculate_bearing(f_coord, candidate.coordinate)
                    nearby.append(
                        NearbyGeographicFeature(
                            feature_type=f_type,
                            name=f_name,
                            distance_meters=round(dist_meters, 1),
                            bearing_degrees=bearing,
                            tags=f_tags,
                        )
                    )
            nearby.sort(key=lambda x: x.distance_meters)
            candidate.nearby_context = nearby[:5]

        return candidates
