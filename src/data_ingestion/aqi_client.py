"""Live hourly air quality forecast client using Open-Meteo Air Quality API.

Adheres strictly to the invariant:
Missing API observations MUST remain None. Never replace missing measurements with invented defaults.
"""
from datetime import datetime
import logging
from typing import List, Optional

from config.settings import settings
from src.common.coordinates import LatLon
from src.common.http_client import AsyncHttpClient
from src.common.schemas import HourlyAQIObservation
from src.common.temporal import IST

logger = logging.getLogger("geoai.aqi")


class OpenMeteoAQIClient:
    """Client for Open-Meteo Air Quality API."""

    def __init__(self, http_client: Optional[AsyncHttpClient] = None):
        self.http_client = http_client or AsyncHttpClient()
        self.base_url = settings.OPEN_METEO_AQI_URL

    async def fetch_hourly_aqi(
        self,
        coordinate: LatLon,
        forecast_days: int = 3,
    ) -> List[HourlyAQIObservation]:
        """Fetch hourly air quality observations for a coordinate.

        Returns a list of HourlyAQIObservation items in IST.
        Missing data fields remain None.
        """
        params = {
            "latitude": coordinate.latitude,
            "longitude": coordinate.longitude,
            "hourly": [
                "pm2_5",
                "pm10",
                "us_aqi",
                "european_aqi",
            ],
            "timezone": settings.DEFAULT_TIMEZONE,
            "forecast_days": min(max(1, forecast_days), 7),
        }

        data = await self.http_client.get_json(
            url=self.base_url,
            params=params,
            cache_ttl_seconds=settings.AQI_CACHE_TTL_SECONDS,
        )

        hourly_block = data.get("hourly", {})
        times: List[str] = hourly_block.get("time", [])

        if not times:
            logger.warning("Empty hourly AQI data returned for (%f, %f)", coordinate.latitude, coordinate.longitude)
            return []

        pm25_list = hourly_block.get("pm2_5", [])
        pm10_list = hourly_block.get("pm10", [])
        us_aqi_list = hourly_block.get("us_aqi", [])
        eu_aqi_list = hourly_block.get("european_aqi", [])

        observations: List[HourlyAQIObservation] = []

        for idx, t_str in enumerate(times):
            try:
                dt = datetime.fromisoformat(t_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=IST)
            except ValueError:
                continue

            # Invariant: missing values remain None
            obs = HourlyAQIObservation(
                time=dt,
                pm2_5=float(pm25_list[idx]) if idx < len(pm25_list) and pm25_list[idx] is not None else None,
                pm10=float(pm10_list[idx]) if idx < len(pm10_list) and pm10_list[idx] is not None else None,
                us_aqi=int(us_aqi_list[idx]) if idx < len(us_aqi_list) and us_aqi_list[idx] is not None else None,
                european_aqi=int(eu_aqi_list[idx]) if idx < len(eu_aqi_list) and eu_aqi_list[idx] is not None else None,
            )
            observations.append(obs)

        return observations
