---
license: apache-2.0
tags:
- geoai
- outdoor-activities
- recommendation
- small-language-model
- geospatial
- qwen2.5
datasets:
- openstreetmap
- open-meteo
pipeline_tag: text-generation
base_model: Qwen/Qwen2.5-1.5B-Instruct
language:
- en
---

# GeoAI Outdoor Activity Recommendation SLM

This model repository hosts the configuration, model card, and evidence-grounded inference runtime for the **GeoAI Outdoor Activity Recommendation Assistant**, developed for the **GeoAI Hackathon at CSRE, IIT Bombay**.

## Model Overview

- **Base Model:** [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **Parameter Count:** 1.54 Billion parameters (strictly under the hackathon 3.0B ceiling)
- **Architecture:** Causal Language Model with RoPE, SwiGLU, and sliding window attention
- **License:** Apache 2.0 (Open weights, permissive research and commercial use)
- **Primary Runtime Platform:** CPU Inference (`torch.bfloat16`) on standard laptop hardware (Intel Core i5, 16 GB RAM)

## Intended Use

The model is utilized as the **Evidence-Grounded Explainer and Advisory Synthesizer** within the GeoAI Assistant. 

### Separation of Concerns & Grounding Invariant
Small language models must not be asked to memorize global weather forecasts or calculate geographic bearings. In this architecture:
1. **Deterministic Geospatial Engine:** Solves coordinate projections, computes wind dispersion plume vectors relative to highways and industrial corridors, evaluates Open-Meteo live hourly weather & AQI forecasts, and computes normalized weighted-average suitability scores across candidate venues.
2. **SLM Advisory Engine:** Receives an immutable **Evidence Table** containing verified observations, and synthesizes natural-language explanations:
   - What venue and time are recommended, and why they rank above alternatives.
   - What to prepare, pack, and monitor.
   - Explicit warnings and postponement advice when environmental conditions are unfavorable.

## Compliance with Hackathon Rules

Per the GeoAI Hackathon rulebook:
> *"If you use a base model unchanged, still create a repository with your configuration and a model card naming the base model and how you use it. It should load with from_pretrained(...)."*

This repository complies by documenting the base model architecture, disclosing all live open data sources (OpenStreetMap, Open-Meteo), specifying generation hyper-parameters, and providing reproducible loading instructions.

## Loading and Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 2. Load Model onto CPU in bfloat16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
model.eval()

# 3. Grounded Prompt Template
prompt = """<|im_start|>system
You are an evidence-grounded GeoAI assistant. Explain the recommendation using ONLY the facts in the Evidence Table below.
<|im_end|>
<|im_start|>user
User Query: Where can I go running tomorrow morning near Powai?
--- EVIDENCE TABLE ---
Activity: Running
Search Location: Powai
Has Suitable Option: True
Top Ranked Venue: Powai Vihar Park (0.49 km away)
Best Time Slot: 2026-10-25 06:00 IST
Suitability Score: 78.1/100 (RECOMMENDED)
Temperature: 25.5°C | Rain: 0.0% | Wind: 1.9 km/h
PM2.5: 22.0 µg/m³ | US AQI: 72
Geographic Context: Located within green park canopy.
--- END EVIDENCE ---
<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(prompt, return_tensors="pt")
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=180,
        temperature=0.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Safety and Environmental Health Thresholds

The system pairs the SLM with deterministic heuristic safety exclusions:
- **Cardio Air-Quality Exclusion:** For strenuous running, cycling, or football, PM2.5 levels exceeding 55 µg/m³ (US AQI > 150) automatically trigger a `not_recommended` status (`has_suitable_option = False`).
- **Asthma Safeguards:** Particulate thresholds are conservatively tightened to PM2.5 > 35 µg/m³ for asthmatic and respiratory-sensitive individuals.
- **Wind Dispersion Plume Modeling:** Venues situated within 1.5 km downwind of expressways during heavy traffic hours receive qualitative plume warnings.

## Limitations and Disclaimer

- **Heuristic Advice:** Suitability scores and generated text are engineering heuristics based on public open forecasts. They do not constitute certified medical, clinical, or official meteorological advice.
- **Data Latency:** Dependent on the upstream update frequencies of Open-Meteo forecasts and OpenStreetMap Overpass servers.
