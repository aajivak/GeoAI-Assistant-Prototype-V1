
"""Rule-assisted natural-language query parser for GeoAI.

Extracts:
- Supported outdoor activity
- Worldwide geographic anchor
- Search radius
- Requested time window (currently IST)
- Personal health sensitivities

Designed to preserve location context and remain compatible
with the existing ParsedQuery and TimeWindow schemas.
"""

import re
from datetime import timedelta
from typing import Optional, Tuple

from src.common.schemas import ParsedQuery, PersonalHealthProfile
from src.common.temporal import TimeWindow, get_ist_now


class QueryParser:
    """Parse outdoor activity questions into structured requests."""

    ACTIVITY_SYNONYMS = {
        "running": [
            r"\brunning\b", r"\brun\b", r"\bjogging\b",
            r"\bjog\b", r"\bmorning walk\b",
            r"\bevening walk\b", r"\bwalking\b", r"\bwalk\b",
        ],
        "cycling": [
            r"\bcycling\b", r"\bcycle\b", r"\bbiking\b",
            r"\bbike\b", r"\bbicycle\b",
        ],
        "golf": [r"\bgolf\b", r"\bgolfing\b"],
        "cricket": [r"\bcricket\b", r"\bnets\b"],
        "football": [
            r"\bfootball\b", r"\bsoccer\b",
            r"\bfive-a-side\b", r"\b5-a-side\b",
        ],
        "tennis": [r"\btennis\b"],
        "hiking": [
            r"\bhiking\b", r"\bhike\b", r"\btrekking\b",
            r"\btrek\b", r"\bnature walk\b",
        ],
        "picnics": [
            r"\bpicnic\b", r"\bpicnics\b",
            r"\bpark outing\b",
        ],
        "outdoor_gatherings": [
            r"\bevent\b", r"\bgathering\b",
            r"\bbirthday party\b", r"\bwedding\b",
            r"\bfair\b", r"\boutdoor party\b",
        ],
        "photography": [
            r"\bphotography\b", r"\bphotos\b",
            r"\bphoto\b", r"\bphotoshoot\b",
            r"\bgolden hour\b",
        ],
    }

    # Expressions that usually indicate the end of a location phrase.
    LOCATION_END = re.compile(
        r"\b(?:tomorrow|today|tonight|this\s+weekend|"
        r"next\s+week|next\s+\d+\s+days|"
        r"within\s+\d+(?:\.\d+)?\s*(?:km|kms|kilometers|miles)|"
        r"at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)|"
        r"with\s+(?:my|a|an)\s+|"
        r"based\s+on|depending\s+on|"
        r"can\s+you|would\s+you|"
        r"what\s+is|what\s+are|"
        r"which\s+is|which\s+are)\b",
        re.IGNORECASE,
    )

    def parse(self, raw_query: str) -> ParsedQuery:
        clean_text = raw_query.strip()
        lower_text = clean_text.lower()

        activity = self._extract_activity(lower_text)
        health_profile = self._extract_health_profile(lower_text)
        location = self._extract_location(clean_text)
        radius = self._extract_radius(lower_text)
        time_description, time_window = self._extract_time_window(
            lower_text
        )

        clarification = None

        if not location:
            clarification = (
                "Please specify a location, city, or neighborhood. "
                "For example: 'near Central Park, New York City, USA'."
            )
        elif not activity:
            clarification = (
                "Which outdoor activity are you planning? "
                "Supported activities include running, cycling, "
                "golf, cricket, football, tennis, hiking, "
                "picnics, gatherings, and photography."
            )

        return ParsedQuery(
            raw_query=clean_text,
            activity=activity,
            spatial_anchor=location,
            search_radius_km=radius,
            time_window_description=time_description,
            resolved_time_window=time_window,
            health_profile=health_profile,
            needs_clarification=clarification is not None,
            clarification_question=clarification,
        )

    def _extract_activity(self, text: str) -> Optional[str]:
        for activity, patterns in self.ACTIVITY_SYNONYMS.items():
            if any(re.search(pattern, text) for pattern in patterns):
                return activity
        return None

    def _extract_health_profile(
        self, text: str
    ) -> PersonalHealthProfile:
        asthma = bool(re.search(
            r"\b(asthma|asthmatic|wheezing|respiratory)\b",
            text,
        ))
        senior = bool(re.search(
            r"\b(senior|seniors|elderly|grandparents|"
            r"older adults)\b",
            text,
        ))
        child = bool(re.search(
            r"\b(kids|kid|child|children|baby|toddler)\b",
            text,
        ))
        pregnant = bool(re.search(
            r"\b(pregnant|pregnancy)\b",
            text,
        ))

        notes = []
        if asthma:
            notes.append("sensitive to air pollution/PM2.5")
        if senior:
            notes.append("heat and exertion sensitivity")
        if child:
            notes.append("heat, UV and traffic sensitivity")
        if pregnant:
            notes.append("thermal comfort and shade priority")

        return PersonalHealthProfile(
            has_asthma=asthma,
            is_senior=senior,
            is_child=child,
            is_pregnant=pregnant,
            notes="; ".join(notes) if notes else None,
        )

    def _extract_location(self, text: str) -> Optional[str]:
        """Preserve complete place, city, state and country names."""

        # Prefer the last explicit geographic preposition.
        # "near Central Park in New York City" must retain both.
        matches = list(re.finditer(
            r"\b(?:near|around|nearby|in|at)\s+",
            text,
            flags=re.IGNORECASE,
        ))

        for match in matches:
            # Skip "at 8 AM", which is a time, not a place.
            remaining = text[match.end():].strip()

            if re.match(
                r"\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
                remaining,
                re.IGNORECASE,
            ):
                continue

            end_match = self.LOCATION_END.search(remaining)
            if end_match:
                remaining = remaining[:end_match.start()]

            # Stop at question or sentence boundary.
            remaining = re.split(r"[?!]", remaining, maxsplit=1)[0]

            # Remove trailing explanatory clauses.
            remaining = re.split(
                r",?\s+(?:so\s+that|because|"
                r"considering|according\s+to)\b",
                remaining,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]

            remaining = remaining.strip(" ,.-")

            # Reject obviously non-geographic fragments.
            if not remaining or len(remaining) < 2:
                continue

            if re.fullmatch(
                r"(?:the\s+)?(?:morning|evening|afternoon|"
                r"night|weekend|next\s+few\s+hours)",
                remaining,
                re.IGNORECASE,
            ):
                continue

            return remaining

        return None

    def _extract_radius(self, text: str) -> float:
        match = re.search(
            r"\b(?:within|inside|radius\s+of|"
            r"up\s+to|around)\s+"
            r"(\d+(?:\.\d+)?)\s*"
            r"(km|kms|kilometers|kilometres|miles|mi)\b",
            text,
        )

        if not match:
            return 8.0

        value = float(match.group(1))
        unit = match.group(2)

        if unit in ("miles", "mi"):
            value *= 1.609344

        return max(0.1, min(value, 100.0))

    def _extract_time_window(
        self, text: str
    ) -> Tuple[str, TimeWindow]:
        """Interpret time phrases in IST for current compatibility."""

        now = get_ist_now()

        def make_window(
            days_ahead: int,
            hour: int,
            minute: int,
            duration_hours: float,
            description: str,
        ) -> Tuple[str, TimeWindow]:
            start = (
                now + timedelta(days=days_ahead)
            ).replace(
                hour=hour,
                minute=minute,
                second=0,
                microsecond=0,
            )
            return description, TimeWindow(
                start_time=start,
                end_time=start + timedelta(hours=duration_hours),
            )

        # Tomorrow expressions
        if "tomorrow morning" in text:
            return make_window(
                1, 6, 0, 4, "tomorrow morning"
            )

        if "tomorrow afternoon" in text:
            return make_window(
                1, 12, 0, 5, "tomorrow afternoon"
            )

        if "tomorrow evening" in text:
            return make_window(
                1, 16, 30, 3.5, "tomorrow evening"
            )

        if "tomorrow" in text:
            return make_window(
                1, 6, 0, 14, "tomorrow"
            )

        # Today expressions
        if "this morning" in text or "today morning" in text:
            return make_window(
                0, 6, 0, 4, "this morning"
            )

        if "this afternoon" in text:
            return make_window(
                0, 12, 0, 5, "this afternoon"
            )

        if (
            "this evening" in text
            or "today evening" in text
            or "sunset" in text
        ):
            start = now.replace(
                hour=16,
                minute=30,
                second=0,
                microsecond=0,
            )

            if start < now:
                start = now

            return "this evening", TimeWindow(
                start_time=start,
                end_time=start + timedelta(hours=3),
            )

        # Explicit time, e.g. "at 8 AM" or "at 6:30 PM".
        explicit = re.search(
            r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
            text,
        )

        if explicit:
            hour = int(explicit.group(1))
            minute = int(explicit.group(2) or 0)
            meridiem = explicit.group(3)

            if 1 <= hour <= 12 and minute <= 59:
                hour = hour % 12
                if meridiem == "pm":
                    hour += 12

                start = now.replace(
                    hour=hour,
                    minute=minute,
                    second=0,
                    microsecond=0,
                )

                if start <= now:
                    start += timedelta(days=1)

                return "specified time", TimeWindow(
                    start_time=start,
                    end_time=start + timedelta(hours=2),
                )

        if "next two days" in text or "next 2 days" in text:
            return "next two days", TimeWindow(
                start_time=now,
                end_time=now + timedelta(hours=48),
            )

        if any(
            word in text
            for word in ("weekend", "saturday", "sunday")
        ):
            days_ahead = (5 - now.weekday()) % 7

            if days_ahead == 0 and now.hour >= 18:
                days_ahead = 7

            return make_window(
                days_ahead, 6, 0, 36, "this weekend"
            )

        # Default: upcoming eight hours.
        start = now.replace(
            minute=0,
            second=0,
            microsecond=0,
        )

        return "upcoming hours", TimeWindow(
            start_time=start,
            end_time=start + timedelta(hours=8),
        )
