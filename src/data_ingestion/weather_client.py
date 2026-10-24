"""Live hourly weather forecast client using Open-Meteo Forecast API.

Adheres strictly to the invariant:
Missing API observations MUST remain None. Never replace missing measurements with invented defaults.
"""
from datetime import datetime
import logging
from typing import Dict, List, Optional

from config.settings import settings
from src.common.coordinates import LatLon
from src.common.http_client import AsyncHttpClient
from src.common.schemas import HourlyWeatherObservation
from src.common.temporal import IST

logger = logging.getLogger("geoai.weather")


class OpenMeteoWeatherClient:
    """Client for Open-Meteo Weather Forecast API."""

    def __init__(self, http_client: Optional[AsyncHttpClient] = None):
        self.http_client = http_client or AsyncHttpClient()
        self.base_url = settings.OPEN_METEO_WEATHER_URL

    async def fetch_hourly_forecast(
        self,
        coordinate: LatLon,
        forecast_days: int = 3,
    ) -> List[HourlyWeatherObservation]:
        """Fetch hourly weather forecast for a coordinate.

        Returns a list of HourlyWeatherObservation items in IST.
        Missing data fields remain None.
        """
        params = {
            "latitude": coordinate.latitude,
            "longitude": coordinate.longitude,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation_probability",
                "precipitation",
                "wind_speed_10m",
                "wind_direction_10m",
                "uv_index",
                "cloud_cover",
                "weather_code",
            ],
            "timezone": settings.DEFAULT_TIMEZONE,
            "forecast_days": min(max(1, forecast_days), 7),
        }

        data = await self.http_client.get_json(
            url=self.base_url,
            params=params,
            cache_ttl_seconds=settings.WEATHER_CACHE_TTL_SECONDS,
        )

        hourly_block = data.get("hourly", {})
        times: List[str] = hourly_block.get("time", [])

        if not times:
            logger.warning("Empty hourly weather data returned for (%f, %f)", coordinate.latitude, coordinate.longitude)
            return []

        temps = hourly_block.get("temperature_2m", [])
        humidities = hourly_block.get("relative_humidity_2m", [])
        precip_probs = hourly_block.get("precipitation_probability", [])
        precips = hourly_block.get("precipitation", [])
        wind_speeds = hourly_block.get("wind_speed_10m", [])
        wind_dirs = hourly_block.get("wind_direction_10m", [])
        uvs = hourly_block.get("uv_index", [])
        clouds = hourly_block.get("cloud_cover", [])
        weather_codes = hourly_block.get("weather_code", [])

        observations: List[HourlyWeatherObservation] = []

        for idx, t_str in enumerate(times):
            # Parse ISO time and localize to IST
            # Open-Meteo format: "2026-10-24T12:00"
            try:
                dt = datetime.fromisoformat(t_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=IST)
            except ValueError:
                continue

            # Invariant: missing values (None in JSON or out of bounds) remain None
            obs = HourlyWeatherObservation(
                time=dt,
                temperature_2m=float(temps[idx]) if idx < len(temps) and temps[idx] is not None else None,
                relative_humidity_2m=float(humidities[idx]) if idx < len(humidities) and humidities[idx] is not None else None,
                precipitation_probability=float(precip_probs[idx]) if idx < len(precip_probs) and precip_probs[idx] is not None else None,
                precipitation=float(precips[idx]) if idx < len(precips) and precips[idx] is not None else None,
                wind_speed_10m=float(wind_speeds[idx]) if idx < len(wind_speeds) and wind_speeds[idx] is not None else None,
                wind_direction_10m=float(wind_dirs[idx]) if idx < len(wind_dirs) and wind_dirs[idx] is not None else None,
                uv_index=float(uvs[idx]) if idx < len(uvs) and uvs[idx] is not None else None,
                cloud_cover=float(clouds[idx]) if idx < len(clouds) and clouds[idx] is not None else None,
                weather_code=int(weather_codes[idx]) if idx < len(weather_codes) and weather_codes[idx] is not None else None,
            )
            observations.append(obs)

        return observations
