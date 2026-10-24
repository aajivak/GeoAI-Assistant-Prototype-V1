"""Unit tests for natural-language query parser across all 10 activities and edge cases."""
import pytest

from src.nlp.query_parser import QueryParser


@pytest.fixture
def parser():
    return QueryParser()


def test_parse_all_ten_activities(parser):
    """Verify that all ten hackathon activities are correctly extracted."""
    queries = [
        ("Where can I go running tomorrow evening near Koregaon Park?", "running"),
        ("Is it okay to cycle to work at 8 AM near Indiranagar?", "cycling"),
        ("Good place and time for golf on Saturday within 40 km of Andheri?", "golf"),
        ("Best time this weekend for a kids' cricket match within 15 km of Whitefield?", "cricket"),
        ("Any open football turf near Hinjewadi today?", "football"),
        ("Outdoor tennis after work today near Hinjewadi, any good spots?", "tennis"),
        ("Is Saturday or Sunday better for a hike near Lonavala?", "hiking"),
        ("Planning a relaxing picnic near Cubbon Park this Sunday", "picnics"),
        ("Planning an outdoor birthday party on Sunday afternoon in Pune", "outdoor_gatherings"),
        ("Where can I take sunset photos this evening around Bandra?", "photography"),
    ]

    for q, expected_act in queries:
        parsed = parser.parse(q)
        assert parsed.activity == expected_act, f"Failed on '{q}': got {parsed.activity}, expected {expected_act}"


def test_parse_health_profiles(parser):
    """Verify extraction of asthma, senior, child, and pregnancy circumstances."""
    # Asthma
    p1 = parser.parse("I have mild asthma, where's good for a morning walk near Powai in the next two days?")
    assert p1.health_profile.has_asthma is True
    assert p1.health_profile.is_senior is False

    # Senior / Elderly
    p2 = parser.parse("When is it safe for my elderly parents to walk outdoors tomorrow in Bengaluru?")
    assert p2.health_profile.is_senior is True

    # Child / Kids
    p3 = parser.parse("Cricket practice for kids near Whitefield")
    assert p3.health_profile.is_child is True

    # Pregnancy
    p4 = parser.parse("Safe gentle park walk during pregnancy near Jayanagar")
    assert p4.health_profile.is_pregnant is True


def test_parse_radius_extraction(parser):
    """Verify extraction of explicit radius or default to 8.0 km."""
    p1 = parser.parse("Cricket match within 15 km of Whitefield")
    assert p1.search_radius_km == 15.0

    p2 = parser.parse("Golf on Saturday within 40 km of Andheri")
    assert p2.search_radius_km == 40.0

    p3 = parser.parse("Running near Powai")
    assert p3.search_radius_km == 8.0  # default


def test_missing_location_requires_clarification(parser):
    """Verify that a query missing location explicitly asks for clarification rather than silently guessing."""
    p = parser.parse("Where can I go running tomorrow morning?")
    assert p.needs_clarification is True
    assert p.spatial_anchor is None
    assert p.clarification_question is not None
    assert "location" in p.clarification_question.lower()


def test_missing_activity_requires_clarification(parser):
    """Verify that a query missing an identifiable activity asks for clarification."""
    p = parser.parse("Best place to go outdoors tomorrow near Bandra?")
    assert p.needs_clarification is True
    assert p.activity is None
    assert p.clarification_question is not None
    assert "activity" in p.clarification_question.lower()
