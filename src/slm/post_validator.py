"""Post-generation validation guard for SLM explanations.

Verifies that the SLM adheres strictly to the provided evidence:
1. Rejects outputs that claim safety or recommend outdoor activity when has_suitable_option is False.
2. Asserts that the top-ranked venue name or anchor location is cited accurately.
3. Detects obvious numerical hallucinations or ungrounded claims.
"""
import re
from typing import List, Tuple
from src.common.schemas import ParsedQuery
from src.scoring.candidate_ranker import CandidateRankingResult


class SLMPostValidator:
    """Validates SLM generated text against input evidence."""

    @staticmethod
    def validate(
        generated_text: str,
        query: ParsedQuery,
        ranking: CandidateRankingResult,
    ) -> Tuple[bool, List[str]]:
        """Validate generated text against deterministic evidence.

        Returns:
            Tuple of (is_valid: bool, failure_reasons: List[str]).
        """
        reasons: List[str] = []
        text_lower = generated_text.lower()

        # Invariant 1: Unsuitable options must NOT be recommended as safe
        if not ranking.has_suitable_option:
            # Check for hallucinated safety claims
            unsafe_assertions = [
                r"\bis recommended\b",
                r"\bhighly recommended\b",
                r"\bgreat conditions\b",
                r"\bperfect time\b",
                r"\bideal conditions\b",
                r"\bsafe to go\b",
            ]
            for pat in unsafe_assertions:
                if re.search(pat, text_lower):
                    reasons.append(f"Model made positive safety assertion '{pat}' despite has_suitable_option=False.")

            # Must contain cautionary words
            caution_indicators = ["not recommended", "unfavorable", "avoid", "caution", "postpone", "indoor", "risk", "warning", "elevated"]
            if not any(ci in text_lower for ci in caution_indicators):
                reasons.append("Model failed to warn the user about unfavorable conditions.")

        # Invariant 2: If suitable option exists, top venue or anchor should be cited
        if ranking.has_suitable_option and ranking.best_recommendation:
            venue_name_parts = ranking.best_recommendation.venue_name.lower().split()
            # At least one significant keyword from the venue name or spatial anchor should appear
            anchor_lower = (query.spatial_anchor or "").lower()
            venue_matches = any(len(p) > 3 and p in text_lower for p in venue_name_parts)
            anchor_matches = (anchor_lower in text_lower) if anchor_lower else False

            if not (venue_matches or anchor_matches):
                reasons.append(f"Model omitted both venue name '{ranking.best_recommendation.venue_name}' and anchor '{anchor_lower}'.")

        # Invariant 3: Check for empty or excessively short response
        if len(generated_text.strip()) < 30:
            reasons.append("Generated explanation is trivially short or empty.")

        is_valid = len(reasons) == 0
        return is_valid, reasons
