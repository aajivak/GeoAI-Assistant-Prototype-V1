---
license: apache-2.0
language:
  - en
base_model: Qwen/Qwen2.5-1.5B-Instruct
pipeline_tag: text-generation
tags:
  - geoai
  - outdoor-activities
  - recommendation
  - small-language-model
  - geospatial
  - qwen2.5
---

# GeoAI Outdoor Activity Recommendation Assistant — Prototype V1

**GeoAI Hackathon @ CSRE, IIT Bombay**

This repository documents the **unchanged** [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) base model used as an evidence-grounded natural-language explanation component in the GeoAI Outdoor Activity Recommendation Assistant.

**No fine-tuning or new model weights are claimed.** This repository provides a model card, application integration metadata, and a reproducible base-model loading example. It does **not** host a standalone checkpoint or the complete running FastAPI service.

- **Source code and application:** [GeoAI Assistant Prototype V1](https://github.com/aajivak/GeoAI-Assistant-Prototype-V1)
- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Parameters:** approximately 1.54 billion
- **Base model license:** Apache-2.0
- **Observed development environment:** Intel Core i5-1334U, 16 GB RAM, PyTorch CPU inference with `torch.bfloat16`

## Intended use

The language model synthesizes explanations of outdoor-activity recommendations from a structured evidence table prepared by a separate deterministic backend. It does not independently fetch live weather, discover venues, calculate geospatial relationships, or assign final suitability scores.

The application pipeline is:

1. **Interpret the request:** Extract the outdoor activity, location, time window, search radius, and optional health sensitivities.
2. **Collect geographic and environmental evidence:** Use OpenStreetMap Nominatim and Overpass for geocoding and candidate venues, and Open-Meteo for hourly weather and air-quality forecasts.
3. **Evaluate candidate venue/time combinations:** Apply activity-specific heuristic scores, contextual indicators, and exclusion rules in Python.
4. **Generate an explanation:** Provide Qwen with the resulting evidence and require an explanation consistent with the computed recommendation. The application includes post-generation checks and may fall back to a deterministic explanation.

## Repository contents

- `README.md` — this model card
- `geoai_config.json` — metadata describing the base-model dependency and GeoAI integration
- `load_base_model.py` — minimal example loading the original pretrained model

**Model loading:** This repository does not contain weights. To load the actual language model, call `from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")`, as demonstrated below. Do not use this repository ID as a checkpoint path.

## Loading the unchanged base model

Install compatible versions of PyTorch, Transformers, and Accelerate in your Python environment. CPU inference may be slow.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
model.eval()

# Illustrative synthetic evidence; NOT a live Powai forecast.
evidence = """Activity: Running
Location: Example City
Suitable option found: No
Reason: Air quality is unfavorable for strenuous outdoor activity.
Action: Recommend postponement or an indoor alternative.
"""

messages = [
    {
        "role": "system",
        "content": (
            "You are an evidence-grounded outdoor activity assistant. "
            "Use only the supplied evidence. Do not invent venues, "
            "measurements, or safety assurances."
        ),
    },
    {
        "role": "user",
        "content": f"Explain the recommendation using this evidence:\n{evidence}",
    },
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt")
with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

The example demonstrates base-model inference only. The full recommendation pipeline, evidence construction, safety logic, and output validation are implemented in the [application source repository](https://github.com/aajivak/GeoAI-Assistant-Prototype-V1).

## Safety approach

The application applies **heuristic**, activity-dependent air-quality and environmental exclusions before generating the language-model explanation. Reported implementation rules include a PM2.5 threshold above 55 µg/m³ for strenuous cardio activities and stricter treatment for respiratory-sensitive users. These are application heuristics, **not medical certification or universal health thresholds**. US AQI and PM2.5 concentration are distinct measurements and should not be treated as numerically interchangeable.

A geographic search anchor may be used when venue discovery fails. Such an anchor is **not a verified sports venue**, and the interface should identify it accordingly.

## Observed prototype performance

Development testing on an Intel Core i5-1334U CPU with 16 GB RAM reported approximately 0.96–1.74 generated tokens per second in different runs. Performance varies with prompt length, output length, software environment, and machine load. These are prototype observations, not a standardized benchmark.

## Limitations

- Live data can be incomplete, stale, or unavailable, and OpenStreetMap venue coverage varies by area.
- Suitability scores are heuristic estimates, not guarantees of safety or suitability.
- The language model may generate unsupported details; application-level validation is still necessary.
- Local CPU generation can take several minutes per request.
- The current repository does not contain model weights, a fine-tuned adapter, or a hosted inference endpoint.

**Disclaimer:** This project does not provide certified medical, public-health, or meteorological advice. Users should consult official advisories and exercise personal judgment.

## Data attribution

- Map and geographic data: © OpenStreetMap contributors
- Weather and air-quality forecast data: Open-Meteo (observe applicable attribution and licensing terms)
- Base model: Qwen/Qwen2.5-1.5B-Instruct, Apache-2.0
