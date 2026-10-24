"""FastAPI Application for GeoAI Outdoor Activity Recommendation Assistant.

Provides health checks, activity catalogs, and the core POST /recommend endpoint.
"""
from datetime import datetime
from typing import Any, Dict, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.activity_definitions import SUPPORTED_ACTIVITIES, ActivityDefinition
from config.settings import settings
from src.api.routes import router as recommend_router
from src.common.temporal import get_ist_now

app = FastAPI(
    title="GeoAI Outdoor Activity Recommendation API",
    description="Live geospatial, weather, and air-quality outdoor activity recommendation service for GeoAI Hackathon @ CSRE, IIT Bombay.",
    version="1.0.0",
)

# Enable CORS for public judging and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(recommend_router)


class HealthResponse(BaseModel):
    status: str
    timestamp_ist: str
    service: str
    version: str
    supported_activities_count: int
    data_sources: List[str]
    attribution: str


class ActivityListResponse(BaseModel):
    total_count: int
    activities: List[ActivityDefinition]


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def get_health() -> HealthResponse:
    """System health check and live status endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp_ist=get_ist_now().isoformat(),
        service="GeoAI Outdoor Activity Recommendation Assistant",
        version="1.0.0",
        supported_activities_count=len(SUPPORTED_ACTIVITIES),
        data_sources=[
            "OpenStreetMap Nominatim (Geocoding)",
            "OpenStreetMap Overpass API (Places & Context)",
            "Open-Meteo Weather API (Live Forecast)",
            "Open-Meteo Air Quality API (Hourly AQI)",
        ],
        attribution="Map data © OpenStreetMap contributors, Weather & Air Quality data by Open-Meteo.com (CC-BY 4.0)",
    )


@app.get("/activities", response_model=ActivityListResponse, tags=["Catalog"])
async def list_activities() -> ActivityListResponse:
    """Return all supported outdoor activities and their geospatial requirements."""
    activity_list = list(SUPPORTED_ACTIVITIES.values())
    return ActivityListResponse(
        total_count=len(activity_list),
        activities=activity_list,
    )
