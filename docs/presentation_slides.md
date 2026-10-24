# GeoAI Outdoor Activity Recommendation Assistant
## CSRE, IIT Bombay — GeoAI Hackathon Presentation (4–6 Slides)

---

### Slide 1: The Problem & Our Approach
**Title:** Contextual, Evidence-Grounded Outdoor Recommendation  
**Subtitle:** Why standard weather forecasts fail individual outdoor planners  

- **The Real-World Gap:** Weather forecasts report regional averages (e.g., "Mumbai 29°C, AQI 160"). But outdoor enthusiasts need specific, contextual guidance: *Which exact nearby park is optimal right now, what hours should be avoided, and how do wind and traffic affect safety?*
- **The Microclimate Reality:** A park downwind of a busy expressway at 8:00 AM suffers from trapped vehicular emissions, but once the wind shifts by 5:00 PM, it becomes pleasant. 
- **Our Solution:** A modular GeoAI assistant that combines deterministic geospatial simulation with an open-weights Small Language Model (SLM).
- **Core Principle:** Strict separation between deterministic geospatial/temporal ranking and language model explanations — zero fabricated weather or phantom places.

---

### Slide 2: End-to-End System Architecture
**Title:** Decoupled Three-Tier Architecture (TerraEngine & Avani Foundations)  

```
[ Natural Language Query ]
            │
            ▼
┌──────────────────────────────────────────────┐
│  Tier 1: Live Data Ingestion & Primitives   │
│  - OSM Nominatim (Live Geocoding)            │
│  - OSM Overpass API (Live Venues & Context)  │
│  - Open-Meteo Weather API (Hourly Forecast)  │
│  - Open-Meteo AQI API (Hourly PM2.5 / PM10)  │
└──────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│  Tier 2: Spatial Simulation & Scoring        │
│  - Wind dispersion vector plume calculation  │
│  - Microclimate buffers (canopy, water)      │
│  - Normalized weighted-average suitability   │
│  - Activity & personal profile sensitivity   │
│  - Joint (Venue × Hour) candidate ranker     │
└──────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│  Tier 3: SLM Evidence-Grounded Synthesis    │
│  - Local Qwen 2.5 1.5B Instruct (bfloat16)   │
│  - Strict immutable Evidence Table           │
│  - Post-generation validation & safety guard │
│  - FastAPI REST Delivery (GET / POST)        │
└──────────────────────────────────────────────┘
```

- **Data Layer:** 100% free, open APIs with polite caching and automatic rate-limit fallbacks.
- **Simulation Layer:** Normalized weighted scoring ($S \in [0, 100]$) with explicit penalties and safety exclusions.
- **Synthesis Layer:** Open-weights SLM running strictly locally on CPU without external proprietary APIs.

---

### Slide 3: Worked Example — Powai & Koregaon Park
**Title:** Spatiotemporal Reasoning in Action  

- **Scenario A: Running near Powai with High Live PM2.5**
  - *Query:* "Where can I go running tomorrow morning near Powai?"
  - *Live Observations:* Temp 25.5°C, Rain 0%, PM2.5 61.9 µg/m³ (US AQI 164).
  - *Safety Decision:* The system enforces an absolute safety exclusion for strenuous cardio when PM2.5 > 55 µg/m³. Status: **`not_recommended`** (`has_suitable_option = False`).
  - *SLM Advisory:* Rather than falsely recommending the top relative park, the assistant alerts the user to elevated particulate levels and advises an indoor workout or postponing until air quality clears.

- **Scenario B: Asthma Sensitivity & Wind Dispersion**
  - *Query:* "I have mild asthma, where's good for a morning walk near Powai in the next two days?"
  - *Evaluation:* Asthmatic threshold tightened to PM2.5 > 35 µg/m³. Downwind highway plume exposure incurs an explicit `-15` point penalty.
  - *Outcome:* The assistant recommends low-traffic green spaces with vegetative canopy buffers during early morning hours when wind blows away from arterial expressways.

---

### Slide 4: Small Language Model (SLM) Runtime
**Title:** Open-Weights Inference & Grounding Engine  

- **Model Selection:** **Qwen/Qwen2.5-1.5B-Instruct**
  - Parameter Count: **1.54 Billion** (well below the 3.0B hackathon limit).
  - License: **Apache 2.0** (100% permissive open source).
  - Weight Architecture: Loaded with Hugging Face `from_pretrained()` in `torch.bfloat16`.
- **CPU Execution on Intel Core i5-1334U (16 GB RAM):**
  - Disk Footprint: **2.89 GB** (14 files on `D:\AI_V2\.cache\huggingface`).
  - Cold Model Load Time: **5.37–12.84s** directly from local NVMe cache.
  - Net Memory Footprint: **2.87–2.98 GB RAM** (measured in `torch.bfloat16` via `psutil`).
  - Measured CPU Generation Throughput: **1.0–1.74 tokens/second** on CPU without discrete GPU.
  - Target Output: **70–140 tokens** (grounded 3–5 sentence natural-language explanation).
- **Zero-Hallucination Guard:**
  - The model receives only verified deterministic facts in an Evidence Table.
  - An automated post-validator rejects outputs that claim safety when conditions are unfavorable, falling back to deterministic synthesis if validation fails.

---

### Slide 5: Engineering Rigor & Deliverables
**Title:** Hackathon Deliverables & Verification  

- **Public Deliverables:**
  1. *Live Public API:* FastAPI service actively running at **`https://indicated-origin-boutique-treo.trycloudflare.com`** with interactive `/docs`, `GET /health`, `GET /activities`, and `POST /recommend`.
  2. *Hugging Face Hub:* Public repository containing the model card, base model pointer, and reproduction config.
  3. *Source Repository & README:* Clean, typed Python codebase on `D:\AI_V2\GeoAI_Assistant`.
  4. *Test Suite:* Comprehensive test coverage (44 passing automated tests: 42 unit/mock + 2 live integration smoke tests).
- **Supported Activities (10+):** Running, Cycling, Golf, Cricket, Football, Tennis, Hiking, Picnics, Outdoor Gatherings, Photography.
- **Personal Health Profiles:** Dynamic sensitivity modifiers for asthma, senior citizens, children, and pregnancy.

---

### Slide 6: Limitations & Future Horizons
**Title:** Honest Limitations & Architectural Evolution  

- **Current Limitations:**
  - *Public API Rate Limits:* Overpass API occasionally throttles during peak global hours (mitigated by our automatic Nominatim POI discovery fallback).
  - *Macro-Forecast Resolution:* Open-Meteo forecasts operate on a grid; microclimates inside dense tree canopies are modeled qualitatively rather than by on-site micrometeorological sensor nets.
  - *Missing Observations:* Some stations do not report hourly PM10 or UV; our engine preserves these as Unknown and re-normalizes weights rather than fabricating values.
- **Future Integration:**
  - Decoupled GIS and SLM runtime ready for direct integration into **Avani** for temporal GIS workflows.
  - On-device edge deployment for rural and park administration dashboards.
