"""Outdoor venue discovery and geographic context using OpenStreetMap.

Overpass is the primary provider. A bounded OpenStreetMap core API XML
fallback provides real, locally mapped features when Overpass is unavailable.
The XML fallback has partial coverage; mapped tags do not prove legal access.
"""

import logging
import math
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import httpx

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
    """OSM venue and context discovery with bounded fallback."""

    MAX_RADIUS_KM = 100.0
    MAX_RESULTS = 100
    CONTEXT_DISTANCE_METERS = 2000.0
    OSM_MAP_URL = "https://api.openstreetmap.org/api/0.6/map"
    # The core OSM API is not a substitute for regional Overpass queries.
    # Keep this fallback strictly local and use only one small request.
    FALLBACK_HALF_HEIGHT_DEG = 0.002
    FALLBACK_HALF_WIDTH_DEG = 0.002
    FALLBACK_MAX_RESPONSE_BYTES = 5_000_000
    FALLBACK_TIMEOUT_SECONDS = 15.0

    def __init__(self, http_client: Optional[AsyncHttpClient] = None):
        self.http_client = http_client or AsyncHttpClient()
        self.primary_url = settings.OVERPASS_BASE_URL
        self.fallback_url = settings.OVERPASS_FALLBACK_URL
        self._overpass_unavailable_until = 0.0

    @staticmethod
    def _validate_radius(radius_km: float) -> float:
        radius = float(radius_km)
        if not 0 < radius <= OverpassClient.MAX_RADIUS_KM:
            raise ValueError("Search radius must be greater than 0 and no more than 100 km.")
        return radius

    @staticmethod
    def _extract_coordinate(element: Dict[str, Any]) -> Optional[LatLon]:
        lat = element.get("lat")
        lon = element.get("lon")
        center = element.get("center") or {}
        if lat is None:
            lat = center.get("lat")
        if lon is None:
            lon = center.get("lon")
        if lat is None or lon is None:
            return None
        try:
            return LatLon(latitude=float(lat), longitude=float(lon))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_accessible(tags: Dict[str, Any]) -> bool:
        access = str(tags.get("access", "")).lower()
        if access in {"private", "no", "customers"}:
            return False
        if str(tags.get("disused", "")).lower() == "yes":
            return False
        if str(tags.get("abandoned", "")).lower() == "yes":
            return False
        return True

    @staticmethod
    def _safe_tag_value(value: str) -> str:
        return str(value).replace("\\", "\\\\").replace('"', '\\"')

    def build_venue_query(
        self, activity_id: str, bbox: BoundingBox, timeout_sec: int = 15
    ) -> str:
        bbox_str = bbox.to_overpass_bbox()
        activity = SUPPORTED_ACTIVITIES.get(activity_id.lower())
        subqueries: List[str] = []
        if activity:
            for leisure in activity.osm_leisure_tags:
                tag = self._safe_tag_value(leisure)
                subqueries.extend([
                    f'node["leisure"="{tag}"]({bbox_str});',
                    f'way["leisure"="{tag}"]({bbox_str});',
                ])
            for sport in activity.osm_sport_tags:
                tag = self._safe_tag_value(sport)
                subqueries.extend([
                    f'node["sport"="{tag}"]({bbox_str});',
                    f'way["sport"="{tag}"]({bbox_str});',
                ])
            for highway in activity.osm_highway_tags:
                tag = self._safe_tag_value(highway)
                subqueries.append(f'way["highway"="{tag}"]({bbox_str});')
            for tourism in activity.osm_tourism_tags:
                tag = self._safe_tag_value(tourism)
                subqueries.extend([
                    f'node["tourism"="{tag}"]({bbox_str});',
                    f'way["tourism"="{tag}"]({bbox_str});',
                ])
        if not subqueries:
            subqueries = [
                f'node["leisure"="park"]({bbox_str});',
                f'way["leisure"="park"]({bbox_str});',
            ]
        query_body = "\n  ".join(dict.fromkeys(subqueries))
        return (
            f"[out:json][timeout:{timeout_sec}];\n"
            "(\n"
            f"  {query_body}\n"
            ");\n"
            "out center tags;"
        )

    def build_context_query(self, bbox: BoundingBox, timeout_sec: int = 15) -> str:
        bbox_str = bbox.to_overpass_bbox()
        return (
            f"[out:json][timeout:{timeout_sec}];\n"
            "(\n"
            f'  way["highway"~"^(motorway|trunk|primary)$"]({bbox_str});\n'
            f'  way["landuse"="industrial"]({bbox_str});\n'
            f'  way["natural"="water"]({bbox_str});\n'
            f'  way["waterway"~"^(river|canal)$"]({bbox_str});\n'
            f'  way["leisure"="park"]({bbox_str});\n'
            f'  way["landuse"="forest"]({bbox_str});\n'
            f'  way["natural"="wood"]({bbox_str});\n'
            ");\n"
            "out center tags;"
        )

    async def execute_query(self, query: str) -> Dict[str, Any]:
        """Try both Overpass endpoints; remember temporary outage locally."""
        if time.monotonic() < self._overpass_unavailable_until:
            raise RuntimeError("Overpass temporarily unavailable; skipping repeat calls")
        errors = []
        for url in dict.fromkeys([self.primary_url, self.fallback_url]):
            if not url:
                continue
            try:
                return await self.http_client.post_overpass(
                    url=url,
                    query=query,
                    cache_ttl_seconds=settings.OSM_PLACES_CACHE_TTL_SECONDS,
                )
            except Exception as exc:
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
                logger.warning("Overpass endpoint failed: %s (%s)", url, type(exc).__name__)
        self._overpass_unavailable_until = time.monotonic() + 300.0
        raise RuntimeError("Both Overpass endpoints unavailable: " + "; ".join(errors))

    @staticmethod
    def _activity_matches(tags: Dict[str, Any], activity_id: str) -> bool:
        activity = SUPPORTED_ACTIVITIES.get(activity_id.lower())
        if activity is None:
            return tags.get("leisure") == "park"
        return (
            tags.get("leisure") in activity.osm_leisure_tags
            or tags.get("sport") in activity.osm_sport_tags
            or tags.get("highway") in activity.osm_highway_tags
            or tags.get("tourism") in activity.osm_tourism_tags
        )

    @staticmethod
    def _xml_tags(element: ET.Element) -> Dict[str, str]:
        return {
            tag.attrib["k"]: tag.attrib.get("v", "")
            for tag in element.findall("tag")
            if "k" in tag.attrib
        }

    async def _fetch_osm_map_elements(self, center: LatLon) -> List[Dict[str, Any]]:
        """Fetch one tiny core-API map tile and normalize nodes/ways."""
        lat = center.latitude
        lon = center.longitude
        half_lat = self.FALLBACK_HALF_HEIGHT_DEG
        half_lon = min(
            0.006,
            self.FALLBACK_HALF_WIDTH_DEG / max(0.35, math.cos(math.radians(lat))),
        )
        south = max(-89.999, lat - half_lat)
        north = min(89.999, lat + half_lat)
        west = max(-179.999, lon - half_lon)
        east = min(179.999, lon + half_lon)
        if west >= east or south >= north:
            return []
        params = {"bbox": f"{west:.6f},{south:.6f},{east:.6f},{north:.6f}"}
        headers = {"User-Agent": settings.USER_AGENT, "Accept": "application/xml"}
        timeout = httpx.Timeout(self.FALLBACK_TIMEOUT_SECONDS, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(self.OSM_MAP_URL, params=params, headers=headers)
            response.raise_for_status()
            if len(response.content) > self.FALLBACK_MAX_RESPONSE_BYTES:
                raise ValueError("OSM fallback response exceeded safe size")
            root = ET.fromstring(response.content)

        nodes: Dict[str, Tuple[float, float]] = {}
        elements: List[Dict[str, Any]] = []
        for node in root.findall("node"):
            node_id = node.get("id")
            try:
                nlat = float(node.get("lat", ""))
                nlon = float(node.get("lon", ""))
            except ValueError:
                continue
            if node_id:
                nodes[node_id] = (nlat, nlon)
            tags = self._xml_tags(node)
            if tags:
                elements.append({
                    "type": "node", "id": node_id,
                    "lat": nlat, "lon": nlon, "tags": tags,
                })

        for way in root.findall("way"):
            tags = self._xml_tags(way)
            if not tags:
                continue
            coords = [
                nodes[nd.get("ref")]
                for nd in way.findall("nd")
                if nd.get("ref") in nodes
            ]
            if not coords:
                continue
            # A way may extend well beyond the requested tile. Prefer its
            # mapped vertices inside the local tile, avoiding a remote centroid.
            local_coords = [
                (y, x) for y, x in coords
                if south <= y <= north and west <= x <= east
            ]
            representative = local_coords or coords
            avg_lat = sum(y for y, _ in representative) / len(representative)
            avg_lon = sum(x for _, x in representative) / len(representative)
            elements.append({
                "type": "way", "id": way.get("id"),
                "center": {"lat": avg_lat, "lon": avg_lon},
                "tags": tags,
            })
        logger.info("OSM XML fallback retrieved %s tagged features (partial area)", len(elements))
        return elements

    def _elements_to_candidates(
        self,
        elements: List[Dict[str, Any]],
        center: LatLon,
        activity_id: str,
        radius: float,
        limit: int,
    ) -> List[PlaceCandidate]:
        candidates: List[PlaceCandidate] = []
        seen_ids = set()
        for element in elements:
            tags = element.get("tags") or {}
            if not self._is_accessible(tags) or not self._activity_matches(tags, activity_id):
                continue
            coord = self._extract_coordinate(element)
            if coord is None:
                continue
            distance = haversine_distance(center, coord)
            if distance > radius:
                continue
            element_id = element.get("id")
            if element_id is None:
                continue
            candidate_id = f"{element.get('type', 'node')}/{element_id}"
            if candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)
            place_type = (
                tags.get("leisure") or tags.get("sport")
                or tags.get("tourism") or tags.get("highway")
                or "recreation_area"
            )
            name = tags.get("name") or f"Unnamed {str(place_type).replace('_', ' ').title()}"
            try:
                candidate = PlaceCandidate(
                    id=candidate_id,
                    name=str(name),
                    place_type=str(place_type),
                    coordinate=coord,
                    distance_from_anchor_km=round(distance, 3),
                    osm_tags=tags,
                )
            except Exception as exc:
                logger.debug("Skipping invalid OSM candidate: %s", exc)
                continue
            candidates.append(candidate)
        candidates.sort(key=lambda c: (
            c.distance_from_anchor_km if c.distance_from_anchor_km is not None else float("inf")
        ))
        return candidates[:limit]

    async def discover_venues(
        self,
        center: LatLon,
        activity_id: str,
        radius_km: float = 5.0,
        limit: int = 10,
    ) -> List[PlaceCandidate]:
        """Find verified mapped features, using a partial-area XML fallback."""
        radius = self._validate_radius(radius_km)
        if limit <= 0:
            return []
        limit = min(int(limit), self.MAX_RESULTS)
        bbox = BoundingBox.from_center_radius(center, radius)
        query = self.build_venue_query(activity_id, bbox)
        elements: List[Dict[str, Any]] = []
        try:
            data = await self.execute_query(query)
            elements = data.get("elements", [])
            if not isinstance(elements, list):
                elements = []
        except Exception as exc:
            logger.warning("Overpass venue discovery unavailable: %s", exc)

        candidates = self._elements_to_candidates(elements, center, activity_id, radius, limit)
        if candidates:
            return candidates

        logger.warning("Using partial-coverage OSM XML venue fallback")
        try:
            fallback_elements = await self._fetch_osm_map_elements(center)
            return self._elements_to_candidates(
                fallback_elements, center, activity_id, radius, limit
            )
        except Exception as exc:
            logger.warning("OSM XML fallback unavailable: %s: %s", type(exc).__name__, exc)
            return []

    @staticmethod
    def _classify_context_feature(tags: Dict[str, Any]) -> Optional[str]:
        if "highway" in tags:
            return f"road_{tags['highway']}"
        if tags.get("landuse") == "industrial":
            return "industrial_zone"
        if tags.get("natural") == "water" or "waterway" in tags:
            return "water_body"
        if (
            tags.get("leisure") == "park"
            or tags.get("landuse") == "forest"
            or tags.get("natural") == "wood"
        ):
            return "park_canopy"
        return None

    async def enrich_with_geographic_context(
        self,
        candidates: List[PlaceCandidate],
        center: LatLon,
        radius_km: float = 5.0,
    ) -> List[PlaceCandidate]:
        """Attach nearby qualitative context; skip when Overpass is down."""
        if not candidates:
            return []
        radius = self._validate_radius(radius_km)
        if time.monotonic() < self._overpass_unavailable_until:
            logger.info("Skipping optional context query while Overpass is unavailable")
            return candidates
        bbox = BoundingBox.from_center_radius(
            center, min(self.MAX_RADIUS_KM, radius + 2.0)
        )
        try:
            data = await self.execute_query(self.build_context_query(bbox))
        except Exception as exc:
            logger.warning("Geographic context unavailable: %s", exc)
            return candidates
        context_features: List[Tuple[str, Optional[str], LatLon, Dict[str, Any]]] = []
        for element in data.get("elements", []):
            tags = element.get("tags") or {}
            feature_type = self._classify_context_feature(tags)
            if feature_type is None:
                continue
            coord = self._extract_coordinate(element)
            if coord is None:
                continue
            context_features.append((feature_type, tags.get("name"), coord, tags))
        for candidate in candidates:
            nearby: List[NearbyGeographicFeature] = []
            for feature_type, feature_name, feature_coord, feature_tags in context_features:
                distance_meters = haversine_distance(
                    feature_coord, candidate.coordinate
                ) * 1000.0
                if distance_meters > self.CONTEXT_DISTANCE_METERS:
                    continue
                bearing = calculate_bearing(
                    feature_coord, candidate.coordinate
                ) % 360.0
                try:
                    feature = NearbyGeographicFeature(
                        feature_type=feature_type,
                        name=feature_name,
                        distance_meters=round(distance_meters, 1),
                        bearing_degrees=bearing,
                        tags=feature_tags,
                    )
                except Exception as exc:
                    logger.debug("Skipping invalid context feature: %s", exc)
                    continue
                nearby.append(feature)
            nearby.sort(key=lambda f: f.distance_meters)
            candidate.nearby_context = nearby[:5]
        return candidates
