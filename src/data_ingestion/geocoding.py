
"""Worldwide geocoding and limited venue fallback via Nominatim.

Responsibilities:
- Resolve geographic anchors worldwide.
- Parse and validate Nominatim results.
- Avoid rejecting sufficiently specific place queries.
- Normalize conversational geographic expressions when necessary.
- Provide a conservative fallback for mapped venues.
- Enforce the requested geographic search radius.

Nominatim is primarily a geocoder, not a general nearby
point-of-interest discovery service. Overpass remains the
primary venue discovery provider.
"""

import logging
import re

from typing import List, Optional

from config.settings import settings

from src.common.coordinates import (
    BoundingBox,
    LatLon,
    haversine_distance,
)

from src.common.http_client import (
    AsyncHttpClient,
    GeocodingError,
)

from src.common.schemas import (
    GeocodingResult,
    PlaceCandidate,
)


logger = logging.getLogger("geoai.geocoding")


class NominatimClient:
    """Worldwide OpenStreetMap Nominatim client."""

    def __init__(
        self,
        http_client: Optional[AsyncHttpClient] = None,
    ):
        self.http_client = (
            http_client or AsyncHttpClient()
        )

        self.base_url = (
            settings.NOMINATIM_BASE_URL
        )

    @staticmethod
    def _normalize_location_query(query: str) -> str:
        """Normalize common geographic phrases for Nominatim.

        Examples:
            Central Park in New York City
            -> Central Park, New York City

            Marine Drive in Mumbai
            -> Marine Drive, Mumbai

        This normalization is only used as a fallback after
        the original query returns no results.
        """

        cleaned = re.sub(
            r"\s+",
            " ",
            query,
        ).strip()

        if not cleaned:
            return cleaned

        # Do not modify already structured geographic queries.
        if "," in cleaned:
            return cleaned

        # Match a place followed by a geographic qualifier.
        # Prefer the final occurrence of "in" or "near".
        matches = list(
            re.finditer(
                r"\s+(?:in|near)\s+",
                cleaned,
                flags=re.IGNORECASE,
            )
        )

        if not matches:
            return cleaned

        separator = matches[-1]

        place_name = cleaned[
            :separator.start()
        ].strip()

        region_name = cleaned[
            separator.end():
        ].strip()

        if not place_name or not region_name:
            return cleaned

        return (
            f"{place_name}, {region_name}"
        )

    @staticmethod
    def _is_specific_query(query: str) -> bool:
        """Identify queries containing meaningful location context."""

        text = query.strip().lower()

        # A comma-separated location generally includes context.
        if "," in text:
            return True

        # Natural-language geographic qualifiers.
        if re.search(
            r"\b(?:in|near|at|around)\b",
            text,
        ):
            return True

        # Several tokens usually provide more context
        # than a single ambiguous place name.
        if len(text.split()) >= 4:
            return True

        return False

    @staticmethod
    def _is_valid_venue_result(
        result: GeocodingResult,
        activity_id: str,
    ) -> bool:
        """Conservatively filter fallback venue categories."""

        category = (
            result.category or ""
        ).lower()

        activity_categories = {
            "running": {
                "leisure",
                "highway",
                "natural",
            },
            "cycling": {
                "highway",
                "leisure",
            },
            "golf": {
                "leisure",
            },
            "cricket": {
                "leisure",
            },
            "football": {
                "leisure",
            },
            "tennis": {
                "leisure",
            },
            "hiking": {
                "highway",
                "natural",
                "leisure",
            },
            "picnics": {
                "leisure",
                "natural",
            },
            "picnic": {
                "leisure",
                "natural",
            },
            "outdoor_gatherings": {
                "leisure",
                "natural",
            },
            "photography": {
                "tourism",
                "natural",
                "leisure",
            },
        }

        allowed = activity_categories.get(
            activity_id.lower(),
            {"leisure"},
        )

        return category in allowed

    async def geocode(
        self,
        query: str,
        country_codes: Optional[str] = None,
        limit: int = 5,
    ) -> List[GeocodingResult]:
        """Geocode a worldwide place name."""

        clean_query = query.strip()

        if not clean_query:
            raise ValueError(
                "Geocoding query cannot be empty."
            )

        params = {
            "q": clean_query,
            "format": "jsonv2",
            "limit": max(
                1,
                min(int(limit), 10),
            ),
            "addressdetails": 1,
        }

        if country_codes:
            params["countrycodes"] = (
                country_codes.strip().lower()
            )

        data = await self.http_client.get_json(
            url=self.base_url,
            params=params,
            cache_ttl_seconds=(
                settings.GEOCODE_CACHE_TTL_SECONDS
            ),
        )

        if not isinstance(data, list):
            raise GeocodingError(
                "Unexpected Nominatim response structure."
            )

        results: List[GeocodingResult] = []

        for item in data:
            if not isinstance(item, dict):
                continue

            try:
                coordinate = LatLon(
                    latitude=float(item["lat"]),
                    longitude=float(item["lon"]),
                )

                bounding_box = None

                raw_bbox = item.get(
                    "boundingbox"
                )

                if (
                    isinstance(raw_bbox, (list, tuple))
                    and len(raw_bbox) == 4
                ):
                    bounding_box = BoundingBox(
                        min_latitude=float(
                            raw_bbox[0]
                        ),
                        max_latitude=float(
                            raw_bbox[1]
                        ),
                        min_longitude=float(
                            raw_bbox[2]
                        ),
                        max_longitude=float(
                            raw_bbox[3]
                        ),
                    )

                result = GeocodingResult(
                    query=clean_query,
                    display_name=(
                        item.get("display_name")
                        or clean_query
                    ),
                    coordinate=coordinate,
                    bounding_box=bounding_box,
                    osm_type=item.get("osm_type"),
                    osm_id=item.get("osm_id"),
                    place_rank=item.get("place_rank"),
                    category=item.get("category"),
                )

                results.append(result)

            except (
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                logger.debug(
                    "Skipping invalid Nominatim result: %s",
                    exc,
                )

        return results

    async def resolve_primary_location(
        self,
        query: str,
        country_codes: Optional[str] = None,
    ) -> GeocodingResult:
        """Resolve a geographic anchor using Nominatim relevance.

        First try the original location expression.

        If it returns no results, try a normalized geographic
        expression without discarding the city or region.
        """

        clean_query = query.strip()

        if not clean_query:
            raise GeocodingError(
                "A geographic location is required."
            )

        queries_to_try = [clean_query]

        normalized_query = (
            self._normalize_location_query(
                clean_query
            )
        )

        if (
            normalized_query
            and normalized_query != clean_query
        ):
            queries_to_try.append(
                normalized_query
            )

        results: List[GeocodingResult] = []

        for candidate_query in queries_to_try:
            results = await self.geocode(
                query=candidate_query,
                country_codes=country_codes,
                limit=5,
            )

            if results:
                logger.info(
                    "Geocoding succeeded using query '%s'",
                    candidate_query,
                )
                break

            logger.info(
                "No Nominatim results for '%s'",
                candidate_query,
            )

        if not results:
            raise GeocodingError(
                f"Could not locate '{clean_query}'. "
                "Please provide a more specific location."
            )

        # A well-qualified query should not be rejected merely
        # because Nominatim returned multiple nearby matches.
        if (
            len(results) > 1
            and not self._is_specific_query(
                clean_query
            )
        ):
            first = results[0]
            second = results[1]

            # Compare locations, not just their display names.
            separation_km = haversine_distance(
                first.coordinate,
                second.coordinate,
            )

            if separation_km > 25.0:
                raise GeocodingError(
                    f"'{clean_query}' matches multiple "
                    "geographically distinct locations. "
                    "Please include a city or country."
                )

        selected = results[0]

        logger.info(
            "Resolved '%s' to '%s' (%f, %f)",
            clean_query,
            selected.display_name,
            selected.coordinate.latitude,
            selected.coordinate.longitude,
        )

        return selected

    async def search_nearby_venues(
        self,
        anchor_name: str,
        anchor_coord: LatLon,
        activity_id: str,
        limit: int = 5,
        radius_km: float = 5.0,
    ) -> List[PlaceCandidate]:
        """Conservative Nominatim fallback for activity venues.

        The query is an approximate geocoding search.
        It does not guarantee complete nearby venue coverage.
        """

        if radius_km <= 0 or limit <= 0:
            return []

        search_terms = {
            "running": "park",
            "cycling": "cycleway",
            "golf": "golf course",
            "cricket": "cricket ground",
            "football": "football pitch",
            "tennis": "tennis court",
            "hiking": "hiking trail",
            "picnic": "garden",
            "picnics": "garden",
            "outdoor_gatherings": "park",
            "photography": "viewpoint",
        }

        term = search_terms.get(
            activity_id.lower(),
            "park",
        )

        query = (
            f"{term}, {anchor_name}"
        )

        try:
            results = await self.geocode(
                query=query,
                limit=max(
                    5,
                    min(limit, 10),
                ),
            )

        except Exception as exc:
            logger.warning(
                "Nominatim fallback search failed for '%s': %s",
                anchor_name,
                exc,
            )
            return []

        candidates: List[PlaceCandidate] = []
        seen_ids = set()

        for result in results:
            if not self._is_valid_venue_result(
                result,
                activity_id,
            ):
                continue

            distance = haversine_distance(
                anchor_coord,
                result.coordinate,
            )

            if distance > radius_km:
                continue

            name = (
                result.display_name.split(",")[0].strip()
            )

            if not name:
                continue

            candidate_id = (
                f"nominatim/{result.osm_type}/"
                f"{result.osm_id}"
                if result.osm_id is not None
                else (
                    f"nominatim/"
                    f"{result.coordinate.latitude:.6f},"
                    f"{result.coordinate.longitude:.6f}"
                )
            )

            if candidate_id in seen_ids:
                continue

            seen_ids.add(candidate_id)

            try:
                candidate = PlaceCandidate(
                    id=candidate_id,
                    name=name,
                    place_type=(
                        result.category
                        or term.replace(" ", "_")
                    ),
                    coordinate=result.coordinate,
                    distance_from_anchor_km=round(
                        distance,
                        3,
                    ),
                )

            except Exception as exc:
                logger.debug(
                    "Skipping invalid fallback candidate: %s",
                    exc,
                )
                continue

            candidates.append(candidate)

        candidates.sort(
            key=lambda candidate: (
                candidate.distance_from_anchor_km
                if candidate.distance_from_anchor_km is not None
                else float("inf")
            )
        )

        return candidates[:limit]
