"""FastAPI router for outdoor recommendation queries."""
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from config.settings import settings
from src.common.schemas import ParsedQuery, PersonalHealthProfile, PlaceCandidate
from src.common.temporal import generate_hourly_slots
from src.data_ingestion.geocoding import NominatimClient
from src.data_ingestion.osm_places import OverpassClient
from src.data_ingestion.weather_client import OpenMeteoWeatherClient
from src.data_ingestion.aqi_client import OpenMeteoAQIClient
from src.nlp.query_parser import QueryParser
from src.scoring.candidate_ranker import CandidateRanker, PlaceTimeSlotRecommendation
from src.scoring.evaluator import DISCLAIMER_TEXT

logger = logging.getLogger("geoai.routes")
router = APIRouter(prefix="", tags=["Recommendations"])


class RecommendRequest(BaseModel):
    """User request payload for outdoor activity recommendation."""
    query: str = Field(
        ...,
        description="Natural language question (e.g., 'Where can I go running tomorrow evening near Koregaon Park?')",
        examples=["Where can I go running tomorrow evening near Koregaon Park?"]
    )
    search_radius_km: Optional[float] = Field(None, description="Optional search radius in km")
    user_profile: Optional[PersonalHealthProfile] = Field(None, description="Optional explicit health sensitivities")
    use_slm: Optional[bool] = Field(None, description="Optionally force SLM generation (true) or deterministic generator (false)")


class RecommendResponse(BaseModel):
    """Structured response containing ranked place-time combinations and evidence."""
    status: str = Field(..., description="'success', 'needs_clarification', or 'error'")
    has_suitable_option: bool = Field(..., description="False if all evaluated candidate slots are unfavorable/not_recommended")
    no_suitable_option_reason: Optional[str] = None
    parsed_query: ParsedQuery
    clarification_question: Optional[str] = None
    explanation: Optional[str] = Field(None, description="Natural language grounded synthesis (SLM or deterministic fallback)")
    explanation_source: str = Field("deterministic_engine", description="'slm_qwen2.5_1.5b' or 'deterministic_engine'")
    best_recommendation: Optional[PlaceTimeSlotRecommendation] = None
    top_alternatives: List[PlaceTimeSlotRecommendation] = Field(default_factory=list)
    temporal_comparison: Optional[str] = None
    total_venues_evaluated: int = 0
    total_space_time_slots_evaluated: int = 0
    model_diagnostics: Optional[Dict[str, Any]] = None
    disclaimer: str = DISCLAIMER_TEXT
    data_sources: List[str]
    attribution: str


@router.post("/recommend", response_model=RecommendResponse)
async def get_recommendation(request: RecommendRequest) -> RecommendResponse:
    """End-to-end spatiotemporal outdoor activity recommendation endpoint."""
    parser = QueryParser()
    parsed_query = parser.parse(request.query)

    # Merge explicit overrides if provided
    if request.search_radius_km is not None:
        parsed_query.search_radius_km = request.search_radius_km
    if request.user_profile is not None:
        parsed_query.health_profile.has_asthma = parsed_query.health_profile.has_asthma or request.user_profile.has_asthma
        parsed_query.health_profile.is_senior = parsed_query.health_profile.is_senior or request.user_profile.is_senior
        parsed_query.health_profile.is_child = parsed_query.health_profile.is_child or request.user_profile.is_child
        parsed_query.health_profile.is_pregnant = parsed_query.health_profile.is_pregnant or request.user_profile.is_pregnant

    # 1. Handle Essential Missing Information
    if parsed_query.needs_clarification:
        return RecommendResponse(
            status="needs_clarification",
            has_suitable_option=False,
            parsed_query=parsed_query,
            clarification_question=parsed_query.clarification_question,
            explanation=parsed_query.clarification_question,
            explanation_source="deterministic_engine",
            disclaimer=DISCLAIMER_TEXT,
            data_sources=[],
            attribution="OpenStreetMap and Open-Meteo",
        )

    # 2. Geocode Location Anchor
    nom_client = NominatimClient()
    try:
        geo_result = await nom_client.resolve_primary_location(parsed_query.spatial_anchor or "")
        anchor_coord = geo_result.coordinate
    except Exception as e:
        logger.warning("Geocoding failed for '%s': %s", parsed_query.spatial_anchor, e)
        return RecommendResponse(
            status="needs_clarification",
            has_suitable_option=False,
            parsed_query=parsed_query,
            clarification_question=f"Could not locate '{parsed_query.spatial_anchor}'. Please specify a city or valid neighborhood name.",
            explanation=f"Could not locate '{parsed_query.spatial_anchor}'. Please specify a city or valid neighborhood name.",
            explanation_source="deterministic_engine",
            disclaimer=DISCLAIMER_TEXT,
            data_sources=[],
            attribution="OpenStreetMap and Open-Meteo",
        )

    # 3. Discover Candidate Venues with Overpass + Nominatim Fallback
    overpass_client = OverpassClient()
    candidates: List[PlaceCandidate] = []
    try:
        candidates = await overpass_client.discover_venues(
            center=anchor_coord,
            activity_id=parsed_query.activity or "running",
            radius_km=parsed_query.search_radius_km,
            limit=8,
        )
    except Exception as e:
        logger.warning("Overpass venue discovery failed (%s). Attempting Nominatim fallback.", e)

    # Resilient fallback to Nominatim venue discovery if Overpass is busy
    if not candidates:
        logger.info("Using Nominatim POI discovery fallback for '%s'", parsed_query.spatial_anchor)
        candidates = await nom_client.search_nearby_venues(
            anchor_name=parsed_query.spatial_anchor or "",
            anchor_coord=anchor_coord,
            activity_id=parsed_query.activity or "running",
            limit=5,
        )

    # Enrich candidates with qualitative geographic context
    try:
        enriched_candidates = await overpass_client.enrich_with_geographic_context(
            candidates=candidates,
            center=anchor_coord,
            radius_km=parsed_query.search_radius_km,
        )
    except Exception:
        enriched_candidates = candidates

    if not enriched_candidates:
        primary_name = geo_result.display_name.split(",")[0]
        enriched_candidates = [
            PlaceCandidate(
                id=f"anchor/{geo_result.osm_id or 1}",
                name=f"{primary_name} Outdoors",
                place_type="recreational_area",
                coordinate=anchor_coord,
                distance_from_anchor_km=0.0,
            )
        ]

    # 4. Fetch Live Weather & Air Quality
    weather_client = OpenMeteoWeatherClient()
    aqi_client = OpenMeteoAQIClient()

    weather_forecasts = await weather_client.fetch_hourly_forecast(anchor_coord, forecast_days=3)
    aqi_forecasts = await aqi_client.fetch_hourly_aqi(anchor_coord, forecast_days=3)

    # 5. Generate Target Hourly Time Slots
    if parsed_query.resolved_time_window:
        target_slots = generate_hourly_slots(parsed_query.resolved_time_window)
    else:
        from src.common.temporal import create_quick_window
        target_slots = generate_hourly_slots(create_quick_window(hours_from_now=0, duration_hours=6))

    target_slots = target_slots[:12]

    # 6. Joint Spatiotemporal Candidate Ranking
    ranker = CandidateRanker()
    ranking_result = ranker.rank_candidates(
        activity_id=parsed_query.activity or "running",
        candidates=enriched_candidates,
        weather_forecasts=weather_forecasts,
        aqi_forecasts=aqi_forecasts,
        target_slots=target_slots,
        health_profile=parsed_query.health_profile,
        top_k=5,
    )

    # 7. Generate Grounded Explanation
    explanation = None
    explanation_source = "deterministic_engine"
    model_diagnostics = None

    # Check whether SLM should be invoked:
    # Explicit False forces deterministic; True forces SLM; None uses SLM if loaded or enabled by default.
    from src.slm.service import get_slm_service
    slm_service = get_slm_service()

    if request.use_slm is False:
        should_use_slm = False
    elif request.use_slm is True:
        should_use_slm = True
    else:
        should_use_slm = slm_service.is_loaded() or settings.ENABLE_SLM_BY_DEFAULT

    if should_use_slm:
        try:
            if not slm_service.is_loaded():
                slm_service.load_model(settings.DEFAULT_SLM_MODEL_ID)

            if slm_service.is_loaded():
                exp_text, diag = await slm_service.generate_explanation(parsed_query, ranking_result)
                explanation = exp_text
                explanation_source = "slm_qwen2.5_1.5b"
                model_diagnostics = diag
        except Exception as e:
            logger.debug("SLM generation unavailable or bypassed (%s). Using deterministic generator.", e)

    if explanation is None:
        from src.slm.deterministic_generator import generate_deterministic_explanation
        explanation = generate_deterministic_explanation(parsed_query, ranking_result)
        explanation_source = "deterministic_engine"

    return RecommendResponse(
        status="success",
        has_suitable_option=ranking_result.has_suitable_option,
        no_suitable_option_reason=ranking_result.no_suitable_option_reason,
        parsed_query=parsed_query,
        explanation=explanation,
        explanation_source=explanation_source,
        best_recommendation=ranking_result.best_recommendation,
        top_alternatives=ranking_result.top_candidates[1:],
        temporal_comparison=ranking_result.temporal_comparison_note,
        total_venues_evaluated=ranking_result.total_venues_evaluated,
        total_space_time_slots_evaluated=ranking_result.total_slots_evaluated,
        model_diagnostics=model_diagnostics,
        disclaimer=DISCLAIMER_TEXT,
        data_sources=[
            "OpenStreetMap Nominatim (Live Geocoding & POI Fallback)",
            "OpenStreetMap Overpass API (Live Venues & Geographic Features)",
            "Open-Meteo Forecast API (Live Hourly Weather)",
            "Open-Meteo Air Quality API (Live Hourly PM2.5/AQI)",
        ],
        attribution="Map data © OpenStreetMap contributors, Weather & Air Quality data by Open-Meteo.com (CC-BY 4.0)",
    )
