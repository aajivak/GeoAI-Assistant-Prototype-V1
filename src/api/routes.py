
"""FastAPI router for worldwide outdoor activity recommendations.

Pipeline:
1. Parse natural-language activity request.
2. Resolve geographic location and destination timezone.
3. Interpret requested time in destination-local time.
4. Discover real OSM venues within the requested radius.
5. Enrich venues with qualitative geographic context.
6. Retrieve UTC-aligned weather and air-quality forecasts.
7. Rank venue-hour combinations with local timezone.
8. Produce grounded deterministic or optional SLM explanation.

No synthetic venues or environmental observations are created.
"""

import logging
import re

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field
from timezonefinder import TimezoneFinder

from config.settings import settings

from src.common.schemas import (
    ParsedQuery,
    PersonalHealthProfile,
    PlaceCandidate,
)

from src.common.temporal import (
    TimeWindow,
    create_quick_window,
    generate_hourly_slots,
    get_timezone,
)

from src.data_ingestion.geocoding import NominatimClient
from src.data_ingestion.osm_places import OverpassClient
from src.data_ingestion.weather_client import OpenMeteoWeatherClient
from src.data_ingestion.aqi_client import OpenMeteoAQIClient

from src.nlp.query_parser import QueryParser

from src.scoring.candidate_ranker import (
    CandidateRanker,
    CandidateRankingResult,
    PlaceTimeSlotRecommendation,
)

from src.scoring.evaluator import DISCLAIMER_TEXT


logger = logging.getLogger("geoai.routes")

router = APIRouter(
    prefix="",
    tags=["Recommendations"],
)

UTC = timezone.utc
TIMEZONE_FINDER = TimezoneFinder()

MAX_EVALUATION_HOURS = 12
MAX_FORECAST_DAYS = 3


class RecommendRequest(BaseModel):
    """Natural-language outdoor activity recommendation request."""

    query: str = Field(
        ...,
        min_length=1,
        description="Outdoor activity question with location and time.",
        examples=[
            "Where can I go running tomorrow morning "
            "near Central Park in New York City?"
        ],
    )

    search_radius_km: Optional[float] = Field(
        None,
        gt=0,
        le=100,
    )

    user_profile: Optional[PersonalHealthProfile] = None

    use_slm: Optional[bool] = None


class RecommendResponse(BaseModel):
    """Outdoor recommendation with supporting evidence."""

    status: str

    has_suitable_option: bool

    no_suitable_option_reason: Optional[str] = None

    parsed_query: ParsedQuery

    clarification_question: Optional[str] = None

    explanation: Optional[str] = None

    explanation_source: str = "deterministic_engine"

    best_recommendation: Optional[
        PlaceTimeSlotRecommendation
    ] = None

    top_alternatives: List[
        PlaceTimeSlotRecommendation
    ] = Field(default_factory=list)

    temporal_comparison: Optional[str] = None

    total_venues_evaluated: int = 0

    total_space_time_slots_evaluated: int = 0

    model_diagnostics: Optional[Dict[str, Any]] = None

    disclaimer: str = DISCLAIMER_TEXT

    data_sources: List[str]

    attribution: str


ATTRIBUTION = (
    "Map data © OpenStreetMap contributors; "
    "weather and air-quality data by Open-Meteo.com."
)


def _distance_km(first, second) -> float:
    """Great-circle distance in kilometers."""

    from src.common.coordinates import haversine_distance

    return haversine_distance(first, second)


def _resolve_location_timezone(coordinate) -> str:
    """Find the destination's IANA timezone."""

    timezone_name = TIMEZONE_FINDER.timezone_at(
        lat=coordinate.latitude,
        lng=coordinate.longitude,
    )

    if timezone_name:
        return timezone_name

    logger.warning(
        "Timezone unavailable at (%s, %s). Falling back to UTC.",
        coordinate.latitude,
        coordinate.longitude,
    )

    return "UTC"


def _make_local_window(
    now_local: datetime,
    day_offset: int,
    hour: int,
    minute: int,
    duration_hours: float,
) -> TimeWindow:
    """Create a destination-local window with elapsed duration."""

    target_date = (
        now_local.date()
        + timedelta(days=day_offset)
    )

    start = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        hour,
        minute,
        tzinfo=now_local.tzinfo,
    )

    end = (
        start.astimezone(UTC)
        + timedelta(hours=duration_hours)
    ).astimezone(now_local.tzinfo)

    return TimeWindow(
        start_time=start,
        end_time=end,
    )


def _resolve_local_time_window(
    raw_query: str,
    timezone_name: str,
) -> TimeWindow:
    """Interpret supported time expressions at the destination.

    This is a bounded natural-language parser, not a complete
    calendar-language interpreter. Unsupported specific date
    expressions should not silently be treated as 'now'.
    """

    local_tz = get_timezone(timezone_name)

    now_local = datetime.now(UTC).astimezone(
        local_tz
    )

    text = re.sub(
        r"\s+",
        " ",
        raw_query.lower(),
    ).strip()

    # Handle explicit clock time before generic day periods.
    explicit_time = re.search(
        r"\b(?:at|around|by)\s+"
        r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
        text,
    )

    if explicit_time:
        hour = int(explicit_time.group(1))
        minute = int(explicit_time.group(2) or 0)
        meridiem = explicit_time.group(3)

        if not (
            1 <= hour <= 12
            and 0 <= minute <= 59
        ):
            raise ValueError(
                "Invalid explicit clock time."
            )

        hour = hour % 12

        if meridiem == "pm":
            hour += 12

        day_offset = (
            1 if re.search(r"\btomorrow\b", text)
            else 0
        )

        window = _make_local_window(
            now_local,
            day_offset,
            hour,
            minute,
            2.0,
        )

        if (
            day_offset == 0
            and window.start_time <= now_local
        ):
            window = _make_local_window(
                now_local,
                1,
                hour,
                minute,
                2.0,
            )

        return window

    # Explicit named dayparts.
    periods = {
        "morning": (6, 0, 4.0),
        "afternoon": (12, 0, 5.0),
        "evening": (16, 30, 3.5),
        "night": (19, 0, 4.0),
    }

    for period, (hour, minute, duration) in periods.items():
        if f"tomorrow {period}" in text:
            return _make_local_window(
                now_local,
                1,
                hour,
                minute,
                duration,
            )

    for period, (hour, minute, duration) in periods.items():
        if (
            f"this {period}" in text
            or f"today {period}" in text
        ):
            window = _make_local_window(
                now_local,
                0,
                hour,
                minute,
                duration,
            )

            if window.end_time <= now_local:
                raise ValueError(
                    f"The requested {period} window has passed "
                    "at the destination."
                )

            return TimeWindow(
                start_time=max(
                    window.start_time,
                    now_local,
                ),
                end_time=window.end_time,
            )

    if re.search(r"\btomorrow\b", text):
        return _make_local_window(
            now_local,
            1,
            6,
            0,
            14.0,
        )

    if re.search(
        r"\bnext\s+(?:two|2)\s+days\b",
        text,
    ):
        return create_quick_window(
            duration_hours=48,
            base_time=now_local,
        )

    if re.search(
        r"\b(?:weekend|saturday|sunday)\b",
        text,
    ):
        if "sunday" in text and "weekend" not in text:
            target_weekday = 6
        else:
            target_weekday = 5

        days_ahead = (
            target_weekday - now_local.weekday()
        ) % 7

        if (
            days_ahead == 0
            and now_local.hour >= 18
        ):
            days_ahead = 7

        return _make_local_window(
            now_local,
            days_ahead,
            6,
            0,
            12.0,
        )

    # Do not silently reinterpret specific dates we do not parse.
    if re.search(
        r"\b\d{4}-\d{1,2}-\d{1,2}\b"
        r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        text,
    ):
        raise ValueError(
            "Explicit calendar dates are not supported "
            "by the current destination-time parser."
        )

    return create_quick_window(
        duration_hours=8,
        base_time=now_local,
    )


def _clarification_response(
    parsed_query: ParsedQuery,
    question: str,
) -> RecommendResponse:
    """Return a response requesting additional information."""

    return RecommendResponse(
        status="needs_clarification",
        has_suitable_option=False,
        parsed_query=parsed_query,
        clarification_question=question,
        explanation=question,
        explanation_source="deterministic_engine",
        disclaimer=DISCLAIMER_TEXT,
        data_sources=[],
        attribution=ATTRIBUTION,
    )


def _unavailable_response(
    parsed_query: ParsedQuery,
    reason: str,
    data_sources: Optional[List[str]] = None,
) -> RecommendResponse:
    """Return an explicit no-result response without fabricated data."""

    return RecommendResponse(
        status="no_data",
        has_suitable_option=False,
        no_suitable_option_reason=reason,
        parsed_query=parsed_query,
        explanation=reason,
        explanation_source="deterministic_engine",
        disclaimer=DISCLAIMER_TEXT,
        data_sources=data_sources or [],
        attribution=ATTRIBUTION,
    )


@router.post(
    "/recommend",
    response_model=RecommendResponse,
)
async def get_recommendation(
    request: RecommendRequest,
) -> RecommendResponse:
    """Generate outdoor recommendations for a requested destination."""

    # --------------------------------------------------
    # 1. Parse query
    # --------------------------------------------------

    parser = QueryParser()
    parsed_query = parser.parse(
        request.query
    )

    if request.search_radius_km is not None:
        parsed_query.search_radius_km = (
            request.search_radius_km
        )

    if request.user_profile is not None:
        current = parsed_query.health_profile
        explicit = request.user_profile

        current.has_asthma = (
            current.has_asthma or explicit.has_asthma
        )
        current.is_senior = (
            current.is_senior or explicit.is_senior
        )
        current.is_child = (
            current.is_child or explicit.is_child
        )
        current.is_pregnant = (
            current.is_pregnant or explicit.is_pregnant
        )

    if parsed_query.needs_clarification:
        return _clarification_response(
            parsed_query,
            parsed_query.clarification_question
            or "Please clarify the activity and location.",
        )

    # --------------------------------------------------
    # 2. Geocode destination
    # --------------------------------------------------

    nom_client = NominatimClient()

    try:
        geo_result = await (
            nom_client.resolve_primary_location(
                parsed_query.spatial_anchor or ""
            )
        )

        anchor_coord = geo_result.coordinate

    except Exception as exc:
        logger.warning(
            "Geocoding failed for '%s': %s",
            parsed_query.spatial_anchor,
            exc,
        )

        return _clarification_response(
            parsed_query,
            "Could not resolve the requested location. "
            "Please provide a more specific place or city.",
        )

    # --------------------------------------------------
    # 3. Destination-local time
    # --------------------------------------------------

    try:
        timezone_name = _resolve_location_timezone(
            anchor_coord
        )

        local_window = _resolve_local_time_window(
            request.query,
            timezone_name,
        )

        parsed_query.resolved_time_window = (
            local_window
        )

    except Exception as exc:
        logger.warning(
            "Destination-time interpretation failed: %s",
            exc,
        )

        return _clarification_response(
            parsed_query,
            str(exc),
        )

    # --------------------------------------------------
    # 4. Discover real venues
    # --------------------------------------------------

    radius_km = (
        parsed_query.search_radius_km
    )

    activity_id = (
        parsed_query.activity or "running"
    )

    overpass_client = OverpassClient()

    candidates: List[PlaceCandidate] = []

    venue_source = None

    try:
        candidates = await (
            overpass_client.discover_venues(
                center=anchor_coord,
                activity_id=activity_id,
                radius_km=radius_km,
                limit=8,
            )
        )

        if candidates:
            venue_source = "OpenStreetMap Overpass"

    except Exception as exc:
        logger.warning(
            "Overpass venue discovery failed: %s",
            exc,
        )

    if not candidates:
        try:
            candidates = await (
                nom_client.search_nearby_venues(
                    anchor_name=(
                        parsed_query.spatial_anchor or ""
                    ),
                    anchor_coord=anchor_coord,
                    activity_id=activity_id,
                    radius_km=radius_km,
                    limit=5,
                )
            )

            if candidates:
                venue_source = "OpenStreetMap Nominatim"

        except Exception as exc:
            logger.warning(
                "Nominatim venue fallback failed: %s",
                exc,
            )

    # Enforce the actual requested search radius.
    candidates = [
        candidate
        for candidate in candidates
        if _distance_km(
            anchor_coord,
            candidate.coordinate,
        ) <= radius_km
    ]

    if not candidates:
        return _unavailable_response(
            parsed_query,
            "No verified outdoor activity venues were found "
            "within the requested search radius. "
            "Try another location, activity, or radius.",
            [
                "OpenStreetMap venue discovery",
            ],
        )

    # --------------------------------------------------
    # 5. Enrich geographic context
    # --------------------------------------------------

    try:
        candidates = await (
            overpass_client.enrich_with_geographic_context(
                candidates=candidates,
                center=anchor_coord,
                radius_km=radius_km,
            )
        )

    except Exception as exc:
        logger.warning(
            "Geographic context unavailable: %s",
            exc,
        )

    # --------------------------------------------------
    # 6. Prepare requested forecast hours
    # --------------------------------------------------

    target_slots = generate_hourly_slots(
        local_window
    )

    target_slots = target_slots[
        :MAX_EVALUATION_HOURS
    ]

    if not target_slots:
        return _unavailable_response(
            parsed_query,
            "No valid hourly evaluation slots were generated.",
            [venue_source] if venue_source else [],
        )

    now_utc = datetime.now(UTC)

    # Open-Meteo forecast requests are limited to 3 days here.
    # Reject windows entirely outside the configured horizon.
    horizon_end = (
        now_utc
        + timedelta(days=MAX_FORECAST_DAYS)
    )

    available_slots = [
        slot
        for slot in target_slots
        if (
            now_utc <= slot.astimezone(UTC)
            < horizon_end
        )
    ]

    if not available_slots:
        return _unavailable_response(
            parsed_query,
            "The requested activity window is outside "
            "the currently supported forecast horizon.",
            [venue_source] if venue_source else [],
        )

    # --------------------------------------------------
    # 7. Retrieve environmental forecasts
    # --------------------------------------------------

    weather_client = OpenMeteoWeatherClient()
    aqi_client = OpenMeteoAQIClient()

    data_sources: List[str] = [
        venue_source or "OpenStreetMap",
    ]

    weather_forecasts = []
    aqi_forecasts = []

    try:
        weather_forecasts = await (
            weather_client.fetch_hourly_forecast(
                anchor_coord,
                forecast_days=MAX_FORECAST_DAYS,
            )
        )

        if weather_forecasts:
            data_sources.append(
                "Open-Meteo Weather Forecast"
            )

    except Exception as exc:
        logger.warning(
            "Weather retrieval failed: %s",
            exc,
        )

    try:
        aqi_forecasts = await (
            aqi_client.fetch_hourly_aqi(
                anchor_coord,
                forecast_days=MAX_FORECAST_DAYS,
            )
        )

        if aqi_forecasts:
            data_sources.append(
                "Open-Meteo Air Quality Forecast"
            )

    except Exception as exc:
        logger.warning(
            "AQI retrieval failed: %s",
            exc,
        )

    if not weather_forecasts:
        return _unavailable_response(
            parsed_query,
            "Weather forecast data is currently unavailable. "
            "The system cannot assess outdoor suitability "
            "without a matching weather forecast.",
            data_sources,
        )

    # --------------------------------------------------
    # 8. Rank venue-hour combinations
    # --------------------------------------------------

    ranker = CandidateRanker()

    ranking_result = ranker.rank_candidates(
        activity_id=activity_id,
        candidates=candidates,
        weather_forecasts=weather_forecasts,
        aqi_forecasts=aqi_forecasts,
        target_slots=available_slots,
        health_profile=parsed_query.health_profile,
        top_k=5,
        local_timezone=timezone_name,
    )

    if not ranking_result.top_candidates:
        return _unavailable_response(
            parsed_query,
            ranking_result.no_suitable_option_reason
            or (
                "No matching hourly weather observations "
                "were available for the requested time."
            ),
            data_sources,
        )

    # --------------------------------------------------
    # 9. Generate grounded explanation
    # --------------------------------------------------

    explanation = None
    explanation_source = "deterministic_engine"
    model_diagnostics = None

    if request.use_slm is False:
        should_use_slm = False

    elif request.use_slm is True:
        should_use_slm = True

    else:
        should_use_slm = (
            settings.ENABLE_SLM_BY_DEFAULT
        )

    if should_use_slm:
        try:
            from src.slm.service import get_slm_service

            slm_service = get_slm_service()

            if not slm_service.is_loaded():
                slm_service.load_model(
                    settings.DEFAULT_SLM_MODEL_ID
                )

            if slm_service.is_loaded():
                explanation, model_diagnostics = (
                    await slm_service.generate_explanation(
                        parsed_query,
                        ranking_result,
                    )
                )

                if explanation:
                    explanation_source = (
                        "slm_qwen2.5_1.5b"
                    )

        except Exception as exc:
            logger.warning(
                "SLM generation failed: %s",
                exc,
            )

    if not explanation:
        from src.slm.deterministic_generator import (
            generate_deterministic_explanation,
        )

        explanation = (
            generate_deterministic_explanation(
                parsed_query,
                ranking_result,
            )
        )

        explanation_source = (
            "deterministic_engine"
        )

    # --------------------------------------------------
    # 10. Return result
    # --------------------------------------------------

    return RecommendResponse(
        status="success",
        has_suitable_option=(
            ranking_result.has_suitable_option
        ),
        no_suitable_option_reason=(
            ranking_result.no_suitable_option_reason
        ),
        parsed_query=parsed_query,
        explanation=explanation,
        explanation_source=explanation_source,
        best_recommendation=(
            ranking_result.best_recommendation
        ),
        top_alternatives=(
            ranking_result.top_candidates[1:]
        ),
        temporal_comparison=(
            ranking_result.temporal_comparison_note
        ),
        total_venues_evaluated=(
            ranking_result.total_venues_evaluated
        ),
        total_space_time_slots_evaluated=(
            ranking_result.total_slots_evaluated
        ),
        model_diagnostics=model_diagnostics,
        disclaimer=DISCLAIMER_TEXT,
        data_sources=data_sources,
        attribution=ATTRIBUTION,
    )
