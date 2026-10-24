
"""Timezone-aware temporal utilities for GeoAI.

Supports:
- UTC and Indian Standard Time
- Worldwide IANA time zones
- Timezone-aware time windows
- Hourly slot generation across DST transitions
- Backward-compatible IST helpers

Location-to-timezone resolution belongs to the geospatial layer.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, model_validator


UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30), name="IST")


def get_ist_now() -> datetime:
    """Return current time in Indian Standard Time."""
    return datetime.now(UTC).astimezone(IST)


def get_utc_now() -> datetime:
    """Return current time in UTC."""
    return datetime.now(UTC)


def get_timezone(name: str):
    """Load an IANA timezone such as America/New_York."""
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"Unknown timezone '{name}'. "
            "Install the tzdata package if timezone data "
            "is unavailable on this system."
        ) from exc


def get_local_now(timezone_name: str) -> datetime:
    """Return current local time in a named timezone."""
    return get_utc_now().astimezone(
        get_timezone(timezone_name)
    )


class TimeWindow(BaseModel):
    """A bounded interval with timezone-aware endpoints."""

    start_time: datetime
    end_time: datetime

    @model_validator(mode="after")
    def validate_window(self) -> "TimeWindow":
        # Retain backward compatibility for naive timestamps.
        # Existing callers historically interpret these as IST.
        if self.start_time.tzinfo is None:
            self.start_time = self.start_time.replace(
                tzinfo=IST
            )

        if self.end_time.tzinfo is None:
            self.end_time = self.end_time.replace(
                tzinfo=IST
            )

        # Compare absolute instants, even if zones differ.
        start_utc = self.start_time.astimezone(UTC)
        end_utc = self.end_time.astimezone(UTC)

        if end_utc <= start_utc:
            raise ValueError(
                "end_time must be strictly after start_time"
            )

        return self

    @property
    def duration_hours(self) -> float:
        """Actual elapsed duration, including DST changes."""
        delta = (
            self.end_time.astimezone(UTC)
            - self.start_time.astimezone(UTC)
        )

        return round(
            delta.total_seconds() / 3600.0,
            2,
        )


def generate_hourly_slots(
    window: TimeWindow,
) -> List[datetime]:
    """Generate hourly timestamps inside the requested interval.

    UTC iteration avoids skipped or duplicated absolute instants
    during daylight-saving transitions.

    Returned timestamps use the window's start timezone.
    """

    local_zone = window.start_time.tzinfo

    start_utc = window.start_time.astimezone(UTC)
    end_utc = window.end_time.astimezone(UTC)

    current = start_utc.replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    # Include only hourly boundaries at or after start.
    if current < start_utc:
        current += timedelta(hours=1)

    slots: List[datetime] = []

    while current < end_utc:
        slots.append(current.astimezone(local_zone))
        current += timedelta(hours=1)

    return slots


def create_quick_window(
    hours_from_now: int = 0,
    duration_hours: int = 4,
    base_time: Optional[datetime] = None,
) -> TimeWindow:
    """Create a relative window, preserving the base timezone."""

    start = base_time or get_ist_now()

    if start.tzinfo is None:
        start = start.replace(tzinfo=IST)

    start_utc = (
        start.astimezone(UTC)
        + timedelta(hours=hours_from_now)
    )

    end_utc = start_utc + timedelta(
        hours=duration_hours
    )

    return TimeWindow(
        start_time=start_utc.astimezone(start.tzinfo),
        end_time=end_utc.astimezone(start.tzinfo),
    )
