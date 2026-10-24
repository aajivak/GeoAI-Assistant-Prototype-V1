
"""Live hourly air quality forecasts using Open-Meteo Air Quality API.

All timestamps are normalized to UTC for worldwide
spatiotemporal recommendation evaluation.

Invariant:
Missing API observations remain None.
Never invent or substitute environmental measurements.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from config.settings import settings
from src.common.coordinates import LatLon
from src.common.http_client import AsyncHttpClient
from src.common.schemas import HourlyAQIObservation

logger = logging.getLogger("geoai.aqi")

UTC = timezone.utc


class OpenMeteoAQIClient:
    """Worldwide hourly air-quality forecast client."""

    def __init__(
        self,
        http_client: Optional[AsyncHttpClient] = None,
    ):
        self.http_client = http_client or AsyncHttpClient()
        self.base_url = settings.OPEN_METEO_AQI_URL

    async def fetch_hourly_aqi(
        self,
        coordinate: LatLon,
        forecast_days: int = 3,
    ) -> List[HourlyAQIObservation]:
        """Fetch hourly air-quality forecasts with UTC timestamps."""

        params = {
            "latitude": coordinate.latitude,
            "longitude": coordinate.longitude,
            "hourly": [
                "pm2_5",
                "pm10",
                "us_aqi",
                "european_aqi",
            ],
            "timezone": "UTC",
            "forecast_days": min(
                max(1, forecast_days),
                7,
            ),
        }

        data = await self.http_client.get_json(
            url=self.base_url,
            params=params,
            cache_ttl_seconds=settings.AQI_CACHE_TTL_SECONDS,
        )

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        if not times:
            logger.warning(
                "No AQI observations for (%f, %f)",
                coordinate.latitude,
                coordinate.longitude,
            )
            return []

        def get_value(
            field: str,
            index: int,
            converter=float,
        ):
            values = hourly.get(field, [])

            if index >= len(values):
                return None

            value = values[index]

            if value is None:
                return None

            try:
                return converter(value)
            except (TypeError, ValueError):
                return None

        observations: List[HourlyAQIObservation] = []

        for index, time_string in enumerate(times):
            try:
                timestamp = datetime.fromisoformat(
                    time_string
                )

                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(
                        tzinfo=UTC
                    )
                else:
                    timestamp = timestamp.astimezone(
                        UTC
                    )

            except (TypeError, ValueError):
                logger.warning(
                    "Skipping invalid AQI timestamp: %s",
                    time_string,
                )
                continue

            observations.append(
                HourlyAQIObservation(
                    time=timestamp,
                    pm2_5=get_value(
                        "pm2_5",
                        index,
                    ),
                    pm10=get_value(
                        "pm10",
                        index,
                    ),
                    us_aqi=get_value(
                        "us_aqi",
                        index,
                        int,
                    ),
                    european_aqi=get_value(
                        "european_aqi",
                        index,
                        int,
                    ),
                )
            )

        logger.info(
            "Loaded %d UTC AQI observations for (%f, %f)",
            len(observations),
            coordinate.latitude,
            coordinate.longitude,
        )

        return observations
