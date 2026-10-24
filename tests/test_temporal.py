"""Unit tests for timezone handling, time-window validation, and hourly slot generation."""
from datetime import datetime, timedelta, timezone
import pytest

from src.common.temporal import (
    IST,
    TimeWindow,
    generate_hourly_slots,
    get_ist_now,
)


def test_ist_timezone():
    """Verify that IST timezone is UTC+05:30."""
    now_ist = get_ist_now()
    assert now_ist.tzinfo is not None
    assert now_ist.utcoffset() == timedelta(hours=5, minutes=30)


def test_time_window_validation():
    """Verify that start_time must be strictly before end_time."""
    now = get_ist_now()
    valid_window = TimeWindow(start_time=now, end_time=now + timedelta(hours=3))
    assert valid_window.duration_hours == 3.0

    # Inverted window should raise ValueError
    with pytest.raises(ValueError):
        TimeWindow(start_time=now + timedelta(hours=1), end_time=now)


def test_hourly_slot_generation():
    """Verify generation of hourly discrete slots."""
    start = datetime(2026, 10, 24, 6, 15, tzinfo=IST)
    end = datetime(2026, 10, 24, 10, 0, tzinfo=IST)
    window = TimeWindow(start_time=start, end_time=end)

    slots = generate_hourly_slots(window)
    assert len(slots) >= 4

    # Verify all slots are spaced by 1 hour and on the hour mark
    for slot in slots:
        assert slot.minute == 0
        assert slot.second == 0
        assert slot.tzinfo is not None
