r"""Hugging Face Transformers SLM Runner for Qwen 2.5 / Gemma open-weights models.

Adheres strictly to project constraints:
1. All weights and caches reside on D:\AI_V2\.cache\huggingface
2. CPU execution with Intel Core i5 optimizations (torch.bfloat16 and torch.inference_mode)
3. Benchmarks tokens/second, actual RAM usage (MB via psutil), and generation latency
4. Decoupled and reusable for Avani
"""
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

from config.settings import settings
from src.slm.base import BaseSLMRunner

logger = logging.getLogger("geoai.slm_runner")


class HFTransformersSLMRunner(BaseSLMRunner):
    """Local inference runner using Hugging Face transformers with measured CPU benchmarks."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_id: Optional[str] = None
        self.device: str = "cpu"
        self.dtype_str: str = "bfloat16"
        self.load_duration_seconds: float = 0.0
        self.model_ram_mb: float = 0.0

    def load_model(
        self,
        model_id_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cpu",
        use_bfloat16: bool = True,
    ) -> bool:
        """Load tokenizer and weights from D: drive Hugging Face cache using from_pretrained()."""
        try:
            import psutil
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            logger.error("torch, transformers, or psutil not installed: %s", e)
            return False

        logger.info("Loading model '%s' on %s (Cache: %s)...", model_id_or_path, device, settings.HF_CACHE_DIR)
        start_t = time.perf_counter()

        os.environ["HF_HOME"] = str(settings.HF_CACHE_DIR)
        process = psutil.Process()
        ram_before_mb = process.memory_info().rss / (1024 * 1024)

        chosen_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        self.dtype_str = "bfloat16" if use_bfloat16 else "float32"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id_or_path,
                cache_dir=str(settings.HF_CACHE_DIR),
            )

            # Intel 13th Gen CPU: bfloat16 cuts memory to ~3.1 GB and doubles memory bandwidth efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path,
                cache_dir=str(settings.HF_CACHE_DIR),
                dtype=chosen_dtype,
                low_cpu_mem_usage=True,
            )
            self.model.eval()

            ram_after_mb = process.memory_info().rss / (1024 * 1024)
            self.model_ram_mb = round(ram_after_mb - ram_before_mb, 2)
            self.model_id = model_id_or_path
            self.device = device
            self.load_duration_seconds = round(time.perf_counter() - start_t, 2)

            logger.info(
                "Successfully loaded '%s' in %.2fs (RAM: %.2f MB, dtype: %s).",
                model_id_or_path, self.load_duration_seconds, self.model_ram_mb, self.dtype_str
            )
            return True
        except Exception as e:
            logger.error("Failed to load model '%s': %s", model_id_or_path, e)
            self.model = None
            self.tokenizer = None
            return False

    def is_loaded(self) -> bool:
        """Return True if model and tokenizer are active in memory."""
        return self.model is not None and self.tokenizer is not None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 220,
        temperature: float = 0.2,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate grounded natural-language text with measured CPU inference benchmarks."""
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        import psutil
        import torch

        process = psutil.Process()
        ram_start_mb = process.memory_info().rss / (1024 * 1024)

        start_t = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_token_count = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0.0),
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][input_token_count:]
        generated_token_count = len(gen_tokens)
        elapsed_seconds = time.perf_counter() - start_t

        tokens_per_sec = round(generated_token_count / max(0.01, elapsed_seconds), 2)
        response_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        ram_end_mb = process.memory_info().rss / (1024 * 1024)

        diagnostics = {
            "model_id": self.model_id,
            "device": self.device,
            "dtype": self.dtype_str,
            "model_ram_mb": self.model_ram_mb,
            "current_process_ram_mb": round(ram_end_mb, 2),
            "load_time_seconds": self.load_duration_seconds,
            "inference_duration_seconds": round(elapsed_seconds, 2),
            "input_tokens": input_token_count,
            "generated_tokens": generated_token_count,
            "tokens_per_second": tokens_per_sec,
        }

        return response_text, diagnostics
