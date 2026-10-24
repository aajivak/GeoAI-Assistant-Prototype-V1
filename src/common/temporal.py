"""Timezone-aware temporal handling and hourly time-window generation.

Standardized to Indian Standard Time (IST, UTC+05:30) per Avani temporal controls.
Uses standard library timezone definition to guarantee seamless cross-platform execution on Windows without tzdata.
"""
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator

# Indian Standard Time (UTC+05:30)
IST = timezone(timedelta(hours=5, minutes=30), name="IST")


def get_ist_now() -> datetime:
    """Return the current datetime localized to Indian Standard Time (IST)."""
    return datetime.now(timezone.utc).astimezone(IST)


class TimeWindow(BaseModel):
    """A bounded time interval with timezone awareness."""
    start_time: datetime
    end_time: datetime

    @model_validator(mode="after")
    def validate_window(self) -> "TimeWindow":
        # Ensure both times are timezone-aware in IST
        if self.start_time.tzinfo is None:
            self.start_time = self.start_time.replace(tzinfo=IST)
        else:
            self.start_time = self.start_time.astimezone(IST)

        if self.end_time.tzinfo is None:
            self.end_time = self.end_time.replace(tzinfo=IST)
        else:
            self.end_time = self.end_time.astimezone(IST)

        if self.end_time <= self.start_time:
            raise ValueError(f"end_time ({self.end_time}) must be strictly after start_time ({self.start_time})")

        return self

    @property
    def duration_hours(self) -> float:
        """Return window duration in fractional hours."""
        delta = self.end_time - self.start_time
        return round(delta.total_seconds() / 3600.0, 2)


def generate_hourly_slots(window: TimeWindow) -> List[datetime]:
    """Generate hourly timestamp slots within a time window.

    Rounds down the start time to the beginning of the hour and steps forward by 1 hour.
    """
    current = window.start_time.replace(minute=0, second=0, microsecond=0)
    slots: List[datetime] = []

    while current < window.end_time:
        if current >= window.start_time or (current + timedelta(hours=1)) > window.start_time:
            slots.append(current)
        current += timedelta(hours=1)

    return slots


def create_quick_window(
    hours_from_now: int = 0,
    duration_hours: int = 4,
    base_time: Optional[datetime] = None,
) -> TimeWindow:
    """Convenience helper to create a time window relative to a base time or now."""
    start = (base_time or get_ist_now()) + timedelta(hours=hours_from_now)
    end = start + timedelta(hours=duration_hours)
    return TimeWindow(start_time=start, end_time=end)
