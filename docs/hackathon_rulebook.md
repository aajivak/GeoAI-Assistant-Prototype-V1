# GeoAI Hackathon @ CSRE, IIT Bombay
## Challenge Specification & Rulebook Reference

### Objective
Build a modular AI assistant (Small Language Model - SLM) that recommends where and when to head outdoors based on live geospatial context, weather, and air-quality data.

### Supported Activities (Minimum 10)
1. Running or jogging
2. Cycling (leisure or commute)
3. Golf
4. Cricket (recreational match or practice)
5. Outdoor football or five-a-side
6. Outdoor tennis
7. Hiking or a nature walk
8. Picnic or park outing
9. Outdoor event or gathering (wedding, fair, birthday)
10. Outdoor photography session

### Key Requirements
- Identify activity, search radius, time range, and personal details (asthma, senior, child, pregnancy).
- Consider multiple nearby places across upcoming hours rather than a single static location.
- Account for geospatial context: wind direction, roads/traffic, nearby industry, water bodies, and green cover.
- Provide practical advice: what to carry, what to watch for, when to avoid.
- Strict data grounding: support every claim with fetched live data; never invent weather or places.
- Language Model: Open-weights model of at most 3 billion parameters running independently. Loadable with `from_pretrained(...)`.
- Free & Open Data: OpenStreetMap (Overpass API, Nominatim), Open-Meteo, OpenAQ, WAQI, Sunrise-Sunset API. Disclose all sources and respect rate limits.
- Deliverables:
  1. Live HTTP API endpoint kept accessible throughout judging.
  2. Public Hugging Face model repository with model card and open license.
  3. Source code repository.
  4. README with run instructions, base model, data sources, and endpoint URL.
  5. 4–6 slides presentation.
