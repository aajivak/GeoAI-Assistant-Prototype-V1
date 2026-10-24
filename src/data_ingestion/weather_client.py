
"""Live hourly weather forecasts using Open-Meteo.

All observation timestamps are normalized to UTC so forecasts
can be compared reliably with timezone-aware activity slots.

Missing API measurements remain None.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from config.settings import settings
from src.common.coordinates import LatLon
from src.common.http_client import AsyncHttpClient
from src.common.schemas import HourlyWeatherObservation

logger = logging.getLogger("geoai.weather")

UTC = timezone.utc


class OpenMeteoWeatherClient:
    """Worldwide hourly weather forecast client."""

    def __init__(
        self,
        http_client: Optional[AsyncHttpClient] = None,
    ):
        self.http_client = http_client or AsyncHttpClient()
        self.base_url = settings.OPEN_METEO_WEATHER_URL

    async def fetch_hourly_forecast(
        self,
        coordinate: LatLon,
        forecast_days: int = 3,
    ) -> List[HourlyWeatherObservation]:
        """Fetch hourly forecasts with UTC timestamps."""

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
            "timezone": "UTC",
            "forecast_days": min(
                max(1, forecast_days),
                7,
            ),
        }

        data = await self.http_client.get_json(
            url=self.base_url,
            params=params,
            cache_ttl_seconds=settings.WEATHER_CACHE_TTL_SECONDS,
        )

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        if not times:
            logger.warning(
                "No weather observations for (%f, %f)",
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
            except (ValueError, TypeError):
                return None

        observations: List[HourlyWeatherObservation] = []

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

            except (ValueError, TypeError):
                logger.warning(
                    "Skipping invalid weather timestamp: %s",
                    time_string,
                )
                continue

            observations.append(
                HourlyWeatherObservation(
                    time=timestamp,
                    temperature_2m=get_value(
                        "temperature_2m", index
                    ),
                    relative_humidity_2m=get_value(
                        "relative_humidity_2m", index
                    ),
                    precipitation_probability=get_value(
                        "precipitation_probability", index
                    ),
                    precipitation=get_value(
                        "precipitation", index
                    ),
                    wind_speed_10m=get_value(
                        "wind_speed_10m", index
                    ),
                    wind_direction_10m=get_value(
                        "wind_direction_10m", index
                    ),
                    uv_index=get_value(
                        "uv_index", index
                    ),
                    cloud_cover=get_value(
                        "cloud_cover", index
                    ),
                    weather_code=get_value(
                        "weather_code",
                        index,
                        int,
                    ),
                )
            )

        logger.info(
            "Loaded %d UTC weather observations for (%f, %f)",
            len(observations),
            coordinate.latitude,
            coordinate.longitude,
        )

        return observations
