"""Natural-language query parser for outdoor activity recommendation.

Extracts:
- Activity (10 supported activities)
- Spatial anchor location
- Search radius (explicit or default)
- Time window (relative expressions converted to IST TimeWindow)
- Personal health profile (asthma, senior, child, pregnancy)

Handles vague, incomplete, or ambiguous queries by explicitly requesting clarification
rather than silently guessing. Designed to easily interface with an SLM parser in Phase 4.
"""
from datetime import datetime, timedelta
import re
from typing import Optional, Tuple

from config.activity_definitions import SUPPORTED_ACTIVITIES
from src.common.schemas import ParsedQuery, PersonalHealthProfile
from src.common.temporal import IST, TimeWindow, get_ist_now


class QueryParser:
    """Rule-assisted natural language query interpreter."""

    ACTIVITY_SYNONYMS = {
        "running": [
            r"\brunning\b", r"\brun\b", r"\bjogging\b", r"\bjog\b",
            r"\bmorning walk\b", r"\bevening walk\b", r"\bwalk\b", r"\bwalking\b",
        ],
        "cycling": [
            r"\bcycling\b", r"\bcycle\b", r"\bbiking\b", r"\bbike\b", r"\bbicycle\b",
        ],
        "golf": [
            r"\bgolf\b", r"\bgolfing\b", r"\bgolf course\b",
        ],
        "cricket": [
            r"\bcricket\b", r"\bcricket match\b", r"\bcricket practice\b", r"\bnets\b",
        ],
        "football": [
            r"\bfootball\b", r"\bsoccer\b", r"\bfive-a-side\b", r"\b5-a-side\b", r"\bturf\b",
        ],
        "tennis": [
            r"\btennis\b", r"\btennis court\b",
        ],
        "hiking": [
            r"\bhiking\b", r"\bhike\b", r"\bnature walk\b", r"\btrek\b", r"\btrekking\b",
        ],
        "picnics": [
            r"\bpicnic\b", r"\bpicnics\b", r"\bpark outing\b", r"\bpicnic spot\b",
        ],
        "outdoor_gatherings": [
            r"\bevent\b", r"\bgathering\b", r"\bbirthday party\b", r"\bwedding\b",
            r"\bfair\b", r"\blawn event\b", r"\boutdoor party\b",
        ],
        "photography": [
            r"\bphotography\b", r"\bphoto\b", r"\bphotos\b", r"\bsunset photos\b",
            r"\bsunrise photos\b", r"\bgolden hour\b", r"\bphotoshoot\b",
        ],
    }

    LOCATION_PREPOSITIONS = [
        r"near\s+([A-Za-z0-9\s,\.-]+?)(?:\s+(?:tomorrow|today|this|on|at|within|for|in|with)|\?|$)",
        r"in\s+([A-Za-z0-9\s,\.-]+?)(?:\s+(?:tomorrow|today|this|on|at|within|for|in|with)|\?|$)",
        r"around\s+([A-Za-z0-9\s,\.-]+?)(?:\s+(?:tomorrow|today|this|on|at|within|for|in|with)|\?|$)",
        r"of\s+([A-Za-z0-9\s,\.-]+?)(?:\s+(?:tomorrow|today|this|on|at|within|for|in|with)|\?|$)",
    ]

    def parse(self, raw_query: str) -> ParsedQuery:
        """Parse free-form user query into a structured ParsedQuery object."""
        clean_text = raw_query.strip()
        lower_text = clean_text.lower()

        # 1. Activity identification
        activity_id = self._extract_activity(lower_text)

        # 2. Personal Health Profile extraction
        health_profile = self._extract_health_profile(lower_text)

        # 3. Spatial Anchor extraction
        spatial_anchor = self._extract_location(clean_text)

        # 4. Search Radius extraction
        search_radius = self._extract_radius(lower_text)

        # 5. Temporal Range extraction
        time_desc, resolved_window = self._extract_time_window(lower_text)

        # 6. Ambiguity & Essential Information Validation
        needs_clarification = False
        clarification_msg = None

        if not spatial_anchor:
            needs_clarification = True
            clarification_msg = "Please specify a location or neighborhood for the activity (e.g. 'near Koregaon Park', 'around Bandra', or 'in Powai')."
        elif not activity_id:
            needs_clarification = True
            clarification_msg = "Could you clarify which outdoor activity you are planning? (We support running, cycling, golf, cricket, football, tennis, hiking, picnics, gatherings, and photography)."

        return ParsedQuery(
            raw_query=clean_text,
            activity=activity_id,
            spatial_anchor=spatial_anchor,
            search_radius_km=search_radius,
            time_window_description=time_desc,
            resolved_time_window=resolved_window,
            health_profile=health_profile,
            needs_clarification=needs_clarification,
            clarification_question=clarification_msg,
        )

    def _extract_activity(self, text: str) -> Optional[str]:
        """Match activity keywords against supported activities."""
        # Check specific multi-word patterns first
        for act_id, patterns in self.ACTIVITY_SYNONYMS.items():
            for pat in patterns:
                if re.search(pat, text, re.IGNORECASE):
                    return act_id
        return None

    def _extract_health_profile(self, text: str) -> PersonalHealthProfile:
        """Identify health or vulnerability circumstances."""
        has_asthma = bool(re.search(r"\b(asthma|asthmatic|wheezing|respiratory|breathing)\b", text))
        is_senior = bool(re.search(r"\b(parents|senior|seniors|elderly|grandparents|older adults|aging)\b", text))
        is_child = bool(re.search(r"\b(kids|kid|child|children|baby|toddler)\b", text))
        is_pregnant = bool(re.search(r"\b(pregnant|pregnancy)\b", text))

        notes = []
        if has_asthma:
            notes.append("sensitive to air pollution/PM2.5")
        if is_senior:
            notes.append("heat/gradient sensitive")
        if is_child:
            notes.append("heat/UV/traffic buffer sensitive")
        if is_pregnant:
            notes.append("thermal comfort and shade priority")

        return PersonalHealthProfile(
            has_asthma=has_asthma,
            is_senior=is_senior,
            is_child=is_child,
            is_pregnant=is_pregnant,
            notes="; ".join(notes) if notes else None,
        )

    def _extract_location(self, text: str) -> Optional[str]:
        """Extract geographic anchor name following prepositions."""
        # Strip trailing punctuation
        cleaned = re.sub(r"[?!.,]+$", "", text).strip()

        for pattern in self.LOCATION_PREPOSITIONS:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                loc = match.group(1).strip()
                # Clean up any trailing query words
                loc = re.sub(
                    r"\b(tomorrow|today|evening|morning|afternoon|after work|this|on|at|within|any good spots)\b.*$",
                    "",
                    loc,
                    flags=re.IGNORECASE,
                ).strip()
                if len(loc) >= 2:
                    return loc

        # Fallback check for common known hackathon cities/areas
        known_places = [
            "Koregaon Park", "Powai", "Whitefield", "Andheri", "Bandra",
            "Lonavala", "Bengaluru", "Pune", "Mumbai", "Hinjewadi",
        ]
        for kp in known_places:
            if kp.lower() in text.lower():
                return kp

        return None

    def _extract_radius(self, text: str) -> float:
        """Extract explicit search radius or default to 8.0 km."""
        match = re.search(r"within\s+(\d+(?:\.\d+)?)\s*(?:km|kilometers|kms)", text)
        if match:
            return float(match.group(1))
        return 8.0  # sensible default radius

    def _extract_time_window(self, text: str) -> Tuple[str, TimeWindow]:
        """Resolve natural time expressions to concrete IST TimeWindow."""
        now = get_ist_now()

        # "tomorrow morning" -> tomorrow 06:00 to 10:00
        if "tomorrow morning" in text:
            t = (now + timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)
            return "tomorrow morning", TimeWindow(start_time=t, end_time=t + timedelta(hours=4))

        # "tomorrow evening" -> tomorrow 16:30 to 20:00
        if "tomorrow evening" in text:
            t = (now + timedelta(days=1)).replace(hour=16, minute=30, second=0, microsecond=0)
            return "tomorrow evening", TimeWindow(start_time=t, end_time=t + timedelta(hours=4))

        # "tomorrow" -> tomorrow 06:00 to 20:00
        if "tomorrow" in text:
            t = (now + timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)
            return "tomorrow", TimeWindow(start_time=t, end_time=t + timedelta(hours=14))

        # "this evening" / "sunset" -> today 16:30 to 19:30
        if "this evening" in text or "sunset" in text or "after work today" in text:
            t = now.replace(hour=16, minute=30, second=0, microsecond=0)
            # If already past 16:30 today, evaluate next 3 hours
            if now.hour >= 18:
                t = now
            return "this evening", TimeWindow(start_time=t, end_time=t + timedelta(hours=3, minutes=30))

        # "at 8 am" / "8 am"
        match_hour = re.search(r"\bat\s+(\d{1,2})\s*(am|pm)\b", text)
        if match_hour:
            h = int(match_hour.group(1))
            meridiem = match_hour.group(2).lower()
            if meridiem == "pm" and h != 12:
                h += 12
            elif meridiem == "am" and h == 12:
                h = 0
            # Target start time at specified hour today (or tomorrow if hour has passed)
            target_start = now.replace(hour=h, minute=0, second=0, microsecond=0)
            if target_start < now:
                target_start += timedelta(days=1)
            return f"at {match_hour.group(1)} {meridiem.upper()}", TimeWindow(
                start_time=target_start,
                end_time=target_start + timedelta(hours=2),
            )

        # "in the next two days" / "next 2 days" -> next 48 hours
        if "next two days" in text or "next 2 days" in text:
            return "next two days", TimeWindow(start_time=now, end_time=now + timedelta(hours=48))

        # "this weekend" / "saturday" / "sunday"
        if "weekend" in text or "saturday" in text or "sunday" in text:
            days_ahead = (5 - now.weekday()) % 7  # 5 = Saturday
            if days_ahead == 0 and now.hour >= 18:
                days_ahead = 7
            sat = (now + timedelta(days=days_ahead)).replace(hour=6, minute=0, second=0, microsecond=0)
            return "this weekend", TimeWindow(start_time=sat, end_time=sat + timedelta(hours=36))

        # Default fallback: next 6 hours starting from current hour
        default_start = now.replace(minute=0, second=0, microsecond=0)
        return "upcoming hours", TimeWindow(start_time=default_start, end_time=default_start + timedelta(hours=8))
