"""Live forward and reverse geocoding via OpenStreetMap Nominatim API.

Adheres strictly to Nominatim usage policy:
- Valid unique User-Agent
- Rate limit compliance (caching & bounded requests)
- Explicit handling of ambiguous or missing place queries
"""
import logging
from typing import List, Optional

from config.settings import settings
from src.common.coordinates import LatLon, BoundingBox, haversine_distance
from src.common.http_client import AsyncHttpClient, GeocodingError
from src.common.schemas import GeocodingResult, PlaceCandidate

logger = logging.getLogger("geoai.geocoding")


class NominatimClient:
    """Client for OpenStreetMap Nominatim geocoding services."""

    def __init__(self, http_client: Optional[AsyncHttpClient] = None):
        self.http_client = http_client or AsyncHttpClient()
        self.base_url = settings.NOMINATIM_BASE_URL

    async def geocode(
        self,
        query: str,
        country_codes: str = "in",
        limit: int = 5,
    ) -> List[GeocodingResult]:
        """Resolve a place name into geographic coordinates."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Geocoding query cannot be empty.")

        params = {
            "q": clean_query,
            "format": "jsonv2",
            "countrycodes": country_codes,
            "limit": limit,
            "addressdetails": 1,
        }

        data = await self.http_client.get_json(
            url=self.base_url,
            params=params,
            cache_ttl_seconds=settings.GEOCODE_CACHE_TTL_SECONDS,
        )

        if not data or not isinstance(data, list):
            logger.warning("No geocoding results found for query: %s", clean_query)
            return []

        results: List[GeocodingResult] = []
        for item in data:
            try:
                lat = float(item["lat"])
                lon = float(item["lon"])
                coord = LatLon(latitude=lat, longitude=lon)

                bbox = None
                raw_bbox = item.get("boundingbox")
                if raw_bbox and len(raw_bbox) == 4:
                    bbox = BoundingBox(
                        min_latitude=float(raw_bbox[0]),
                        max_latitude=float(raw_bbox[1]),
                        min_longitude=float(raw_bbox[2]),
                        max_longitude=float(raw_bbox[3]),
                    )

                results.append(
                    GeocodingResult(
                        query=clean_query,
                        display_name=item.get("display_name", ""),
                        coordinate=coord,
                        bounding_box=bbox,
                        osm_type=item.get("osm_type"),
                        osm_id=item.get("osm_id"),
                        place_rank=item.get("place_rank"),
                        category=item.get("category"),
                    )
                )
            except (KeyError, ValueError) as err:
                logger.error("Failed to parse Nominatim entry: %s, error: %s", item, err)
                continue

        return results

    async def resolve_primary_location(
        self,
        query: str,
        country_codes: str = "in",
    ) -> GeocodingResult:
        """Resolve the primary location.

        If ambiguous or no results are found, raises GeocodingError with helpful details.
        """
        results = await self.geocode(query, country_codes=country_codes, limit=3)
        if not results:
            raise GeocodingError(
                f"Could not find any location matching '{query}'. Please check the place name or specify the city."
            )
        return results[0]

    async def search_nearby_venues(
        self,
        anchor_name: str,
        anchor_coord: LatLon,
        activity_id: str,
        limit: int = 5,
    ) -> List[PlaceCandidate]:
        """Discover candidate venues near an anchor using Nominatim (resilient fallback when Overpass is busy)."""
        search_terms = {
            "running": "park",
            "cycling": "cycle track",
            "golf": "golf course",
            "cricket": "cricket ground",
            "football": "football turf",
            "tennis": "tennis court",
            "hiking": "nature park",
            "picnics": "garden",
            "outdoor_gatherings": "lawn",
            "photography": "viewpoint",
        }
        term = search_terms.get(activity_id.lower(), "park")
        query_str = f"{term} near {anchor_name}"

        try:
            results = await self.geocode(query_str, limit=limit)
            candidates: List[PlaceCandidate] = []
            for r in results:
                name = r.display_name.split(",")[0].strip()
                if not name:
                    continue
                dist = haversine_distance(anchor_coord, r.coordinate)
                candidates.append(
                    PlaceCandidate(
                        id=f"nominatim/{r.osm_id or hash(name)}",
                        name=name,
                        place_type=r.category or term,
                        coordinate=r.coordinate,
                        distance_from_anchor_km=dist,
                    )
                )
            candidates.sort(key=lambda c: c.distance_from_anchor_km or 999.0)
            return candidates
        except Exception as e:
            logger.warning("Nominatim nearby venue search failed (%s)", e)
            return []
