"""Singleton SLM Service orchestrating evidence-grounded generation, validation, and deterministic fallbacks."""
import logging
from typing import Any, Dict, Optional, Tuple

from src.common.schemas import ParsedQuery
from src.scoring.candidate_ranker import CandidateRankingResult
from src.slm.deterministic_generator import generate_deterministic_explanation
from src.slm.hf_runner import HFTransformersSLMRunner
from src.slm.post_validator import SLMPostValidator
from src.slm.prompt_builder import build_grounded_slm_prompt

logger = logging.getLogger("geoai.slm_service")


class SLMService:
    """Manages SLM lifecycle, inference execution, and post-validation guards."""

    _instance: Optional["SLMService"] = None

    def __init__(self):
        self.runner = HFTransformersSLMRunner()
        self.validator = SLMPostValidator()

    @classmethod
    def get_instance(cls) -> "SLMService":
        if cls._instance is None:
            cls._instance = SLMService()
        return cls._instance

    def load_model(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct") -> bool:
        """Load model into CPU memory."""
        return self.runner.load_model(model_id_or_path=model_id, device="cpu")

    def is_loaded(self) -> bool:
        return self.runner.is_loaded()

    async def generate_explanation(
        self,
        query: ParsedQuery,
        ranking: CandidateRankingResult,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate an evidence-grounded explanation.

        Validates output with post_validator. If validation fails or model is not loaded,
        falls back to deterministic generator.
        """
        if not self.is_loaded():
            logger.debug("SLM is not loaded. Using deterministic explanation generator.")
            fallback = generate_deterministic_explanation(query, ranking)
            return fallback, {"source": "deterministic_engine", "model_loaded": False}

        prompt = build_grounded_slm_prompt(query, ranking)
        raw_text, diagnostics = self.runner.generate(prompt=prompt, max_new_tokens=140, temperature=0.2)

        # Post-generation validation guard
        is_valid, validation_errors = self.validator.validate(raw_text, query, ranking)

        if not is_valid:
            logger.warning(
                "SLM output failed post-validation (%s). Substituting deterministic fallback.",
                validation_errors,
            )
            fallback = generate_deterministic_explanation(query, ranking)
            diagnostics["validation_passed"] = False
            diagnostics["validation_errors"] = validation_errors
            diagnostics["fallback_applied"] = True
            return fallback, diagnostics

        diagnostics["validation_passed"] = True
        diagnostics["fallback_applied"] = False
        return raw_text, diagnostics


def get_slm_service() -> SLMService:
    """Convenience accessor for SLM service singleton."""
    return SLMService.get_instance()
