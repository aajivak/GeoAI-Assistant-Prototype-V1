"""Abstract Base Class for Small Language Model (SLM) Inference Runrunners.

Designed to be decoupled from the FastAPI backend and reusable across Avani and TerraEngine.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class BaseSLMRunner(ABC):
    """Abstract interface for local SLM inference engines."""

    @abstractmethod
    def load_model(self, model_id_or_path: str, device: str = "cpu") -> bool:
        """Load model weights and tokenizer into memory."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 250,
        temperature: float = 0.2,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate text response.

        Returns:
            Tuple of (generated_text, diagnostic_metrics_dict).
        """
        pass
