# GeoAI Outdoor Activity Recommendation Assistant
**GeoAI Hackathon @ CSRE, IIT Bombay**

An evidence-grounded AI assistant that determines the optimal nearby place and time for outdoor activities using live geospatial context, microclimate analysis, real-time weather forecasts, and hourly air-quality data.

---

## 1. Quickstart & How to Run

### System Prerequisites
- **OS:** Windows 11
- **Platform:** Intel Core i5 / x86_64, 16 GB RAM (No dedicated GPU required)
- **Python:** 3.14+ (Installed in `D:\AI_V2\GeoAI_Assistant\.venv`)
- **Storage:** All environments, local caches, and model weights reside on `D:\`

### Setup and Launch
```powershell
# 1. Navigate to the project root
cd D:\AI_V2\GeoAI_Assistant

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Start the FastAPI Application Server
.\.venv\Scripts\uvicorn.exe src.api.app:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API documentation (Swagger UI) is available at:  
👉 **`http://127.0.0.1:8000/docs`**

---

## 2. API Endpoints & Live URL

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System health check, local time in IST, active version, data source attribution |
| `GET` | `/activities` | Catalog of 10+ supported activities, ideal thresholds, and required venue tags |
| `POST` | `/recommend` | End-to-end spatiotemporal outdoor activity recommendation engine |

### Sample Request (`POST /recommend`)
```json
{
  "query": "Where can I go running tomorrow morning near Koregaon Park?",
  "search_radius_km": 5.0
}
```

### Public Live Endpoint URL
- Local Host: `http://127.0.0.1:8000`
- Public HTTPS Endpoint (Judging Window): **`https://indicated-origin-boutique-treo.trycloudflare.com`**
- Interactive Swagger UI: **`https://indicated-origin-boutique-treo.trycloudflare.com/docs`**
- System Health Check: **`https://indicated-origin-boutique-treo.trycloudflare.com/health`**

---

## 3. Language Model Disclosure & CPU Benchmarks

- **Model Name:** [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **Parameter Count:** 1.54 Billion parameters (strictly compliant with the hackathon's ≤ 3.0B ceiling)
- **License:** Apache 2.0 (Open weights, permissive use)
- **Inference Runtime:** Hugging Face `transformers.AutoModelForCausalLM.from_pretrained(...)` executing strictly locally on CPU in `torch.bfloat16`.
- **Hugging Face Repository Card:** [docs/hf_model_card.md](file:///D:/AI_V2/GeoAI_Assistant/docs/hf_model_card.md)

### Measured CPU Benchmarks (Intel Core i5-1334U, 16 GB RAM)
| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Model Parameter Count** | 1.54 Billion | Within hackathon limit |
| **Disk Cache Footprint** | 2.89 GB (14 files) | Strictly locked on `D:\AI_V2\.cache\huggingface` |
| **Cold Load Time (from disk)** | 5.37 – 12.84 seconds | Fast local disk loading |
| **Net Model RAM Usage** | 2,935.5 MB (~2.87 GB) | Measured via `psutil` in `torch.bfloat16` |
| **Total Process RAM** | 3,257.7 – 3,431.1 MB | Well within 16 GB hardware capacity |
| **CPU Generation Throughput** | 0.96 – 1.74 tokens/sec | Native AVX2 vectorization on CPU |
| **Output Token Length** | 70 – 150 tokens | Grounded 3–5 sentence explanations |

---

## 4. Live Data Sources & Attribution

All data is fetched live at query time from free and open APIs. No datasets were supplied in advance:

1. **OpenStreetMap via Nominatim:** Forward geocoding and location boundary resolution.  
   *Attribution:* © OpenStreetMap contributors.
2. **OpenStreetMap via Overpass API:** POI venue discovery (parks, running tracks, sports pitches, trails, courts) and contextual features (highways, industrial polygons, water bodies).  
   *Attribution:* © OpenStreetMap contributors.
3. **Open-Meteo Weather Forecast API:** Live hourly forecasts for temperature, precipitation probability, precipitation volume, wind speed, wind direction, UV index, and cloud cover.  
   *Attribution:* Open-Meteo.com (CC-BY 4.0).
4. **Open-Meteo Air Quality API:** Live hourly PM2.5, PM10, European AQI, and US AQI forecasts.  
   *Attribution:* Open-Meteo.com (CC-BY 4.0).

---

## 5. Supported Activities & Safety Exclusions

The assistant supports ten core outdoor activities:
1. **Running / Jogging**
2. **Cycling (Leisure / Commute)**
3. **Golf**
4. **Cricket (Match / Practice)**
5. **Outdoor Football / Five-a-side**
6. **Outdoor Tennis**
7. **Hiking / Nature Walk**
8. **Picnic / Park Outing**
9. **Outdoor Events / Gatherings**
10. **Outdoor Photography Sessions**

### Absolute Safety Distinction
- **Relative Ranking:** Ranks evaluated venue-hour combinations from best to worst.
- **Absolute Status:** Assigns `recommended`, `caution`, or `not_recommended`.
- **Cardio Safety Exclusion:** For high-exertion cardio (running, football, cycling), PM2.5 levels exceeding 55 µg/m³ (US AQI > 150) automatically trigger `not_recommended` status (`has_suitable_option = False`), advising indoor exercise or session postponement.
- **Health Profile Safeguards:** Dynamic thresholds for asthma (stricter PM2.5 limit < 35 µg/m³), senior citizens (heat stress > 32°C and slippery surface warnings), children (UV index > 6), and pregnancy.

---

## 6. Running Tests

```powershell
# Run all offline unit, safety regression, and ranking tests:
.\.venv\Scripts\pytest.exe tests -k "not live" -v

# Run the complete test suite including live external API integration:
.\.venv\Scripts\pytest.exe tests -v
```

---

## 7. Disclaimer
All suitability scores, health adjustments, and generated recommendations are heuristic engineering estimates derived from open meteorological forecasts and geographic indicators. They do not constitute certified medical, clinical, or meteorological advice.
