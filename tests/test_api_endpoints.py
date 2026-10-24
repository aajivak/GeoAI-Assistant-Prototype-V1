"""Unit tests for FastAPI endpoints (/health and /activities)."""
from fastapi.testclient import TestClient
import pytest

from src.api.app import app

client = TestClient(app)


def test_get_health():
    """Verify that GET /health returns HTTP 200, status healthy, and attribution."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp_ist" in data
    assert data["supported_activities_count"] >= 10
    assert "OpenStreetMap" in data["attribution"]
    assert "Open-Meteo" in data["attribution"]


def test_get_activities():
    """Verify that GET /activities lists at least 10 supported outdoor activities."""
    response = client.get("/activities")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] >= 10
    activities = data["activities"]

    activity_ids = [a["id"] for a in activities]
    required_activities = [
        "running",
        "cycling",
        "golf",
        "cricket",
        "football",
        "tennis",
        "hiking",
        "picnics",
        "outdoor_gatherings",
        "photography",
    ]
    for req in required_activities:
        assert req in activity_ids, f"Required activity '{req}' missing from /activities"


def test_get_root_gui():
    """Verify that GET / serves the standalone web GUI index.html."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "GeoAI Assistant" in response.text
    assert "recommendForm" in response.text
