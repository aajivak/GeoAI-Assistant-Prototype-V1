"""Global project settings and environment configuration.

Strictly adheres to project constraints:
- All cache and model weight directories remain on D:
- No hardcoded secrets or API keys
- Resilient timeouts and retry policies for external open APIs
"""
import os
from pathlib import Path


class Settings:
    """Project-wide settings object."""

    # Project paths on D: drive
    PROJECT_ROOT: Path = Path(r"D:\AI_V2\GeoAI_Assistant")
    CACHE_DIR: Path = Path(r"D:\AI_V2\.cache\geoai_cache")
    HF_CACHE_DIR: Path = Path(r"D:\AI_V2\.cache\huggingface")

    # Timezone
    DEFAULT_TIMEZONE: str = "Asia/Kolkata"

    # External API Base URLs
    NOMINATIM_BASE_URL: str = "https://nominatim.openstreetmap.org/search"
    OVERPASS_BASE_URL: str = "https://overpass-api.de/api/interpreter"
    OVERPASS_FALLBACK_URL: str = "https://overpass.kumi.systems/api/interpreter"
    OPEN_METEO_WEATHER_URL: str = "https://api.open-meteo.com/v1/forecast"
    OPEN_METEO_AQI_URL: str = "https://air-quality-api.open-meteo.com/v1/air-quality"

    # User-Agent header (Mandatory by OSM and Nominatim usage policies)
    USER_AGENT: str = "GeoAI-IITB-Outdoor-Assistant/1.0 (csre-hackathon-2026)"

    # HTTP Client Configuration
    HTTP_TIMEOUT_SECONDS: float = 12.0
    HTTP_CONNECT_TIMEOUT_SECONDS: float = 5.0
    HTTP_MAX_RETRIES: int = 3
    HTTP_BACKOFF_FACTOR: float = 1.5

    # Cache TTLs (seconds)
    GEOCODE_CACHE_TTL_SECONDS: int = 86400 * 7  # 7 days for static place geocoding
    OSM_PLACES_CACHE_TTL_SECONDS: int = 86400    # 24 hours for venue discovery
    WEATHER_CACHE_TTL_SECONDS: int = 1800       # 30 minutes for hourly forecasts
    AQI_CACHE_TTL_SECONDS: int = 1800           # 30 minutes for air quality forecasts

    # Server configuration
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    # SLM Configuration
    DEFAULT_SLM_MODEL_ID: str = "Qwen/Qwen2.5-1.5B-Instruct"
    ENABLE_SLM_BY_DEFAULT: bool = os.environ.get("GEOAI_ENABLE_SLM", "false").lower() == "true"

    def __init__(self):
        # Ensure cache directories exist on D: drive without touching any external projects
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(self.HF_CACHE_DIR)


settings = Settings()
