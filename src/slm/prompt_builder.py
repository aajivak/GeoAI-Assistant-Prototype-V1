"""Strict evidence-grounded prompt constructor for Qwen 2.5 / Gemma SLM.

Injects ONLY deterministic facts:
- Parsed user query
- Ranked candidate places and hours
- Actual weather and AQI measurements
- Geographic context indicators
- Recommendation status (recommended, caution, not_recommended)
- Explicit instructions: never invent places, forecast values, or safety claims.
"""
from typing import Dict, Any
from src.common.schemas import ParsedQuery
from src.scoring.candidate_ranker import CandidateRankingResult


def build_grounded_slm_prompt(
    query: ParsedQuery,
    ranking: CandidateRankingResult,
) -> str:
    """Build a prompt containing only verified deterministic evidence."""
    best = ranking.best_recommendation
    activity = (query.activity or "outdoor activity").title()
    anchor = query.spatial_anchor or "the designated area"

    # Assemble structured evidence facts
    evidence_lines = [
        f"Activity: {activity}",
        f"Search Location: {anchor}",
        f"Has Suitable Option: {ranking.has_suitable_option}",
    ]

    if not ranking.has_suitable_option:
        evidence_lines.append(f"Safety Alert: {ranking.no_suitable_option_reason}")

    if best:
        w = best.weather_summary
        a = best.aqi_summary
        evidence_lines.extend([
            f"Top Ranked Venue: {best.venue_name} ({best.place_type})",
            f"Distance: {best.distance_km:.2f} km",
            f"Best Time Slot: {best.slot_time_ist}",
            f"Suitability Score: {best.suitability_score}/100",
            f"Absolute Status: {best.recommendation_status.upper()}",
            f"Temperature: {w.get('temperature_c')}°C",
            f"Humidity: {w.get('humidity_percent')}%",
            f"Rain Probability: {w.get('rain_probability_percent')}%",
            f"Wind Speed: {w.get('wind_speed_kmh')} km/h",
            f"PM2.5: {a.get('pm2_5')} µg/m³",
            f"US AQI: {a.get('us_aqi')}",
            f"Geographic Context: {best.geographic_context.summary_advisory}",
        ])
        if ranking.temporal_comparison_note:
            evidence_lines.append(f"Timing Comparison: {ranking.temporal_comparison_note}")
        if best.score_breakdown.heuristic_notes:
            evidence_lines.append(f"Advisories: {'; '.join(best.score_breakdown.heuristic_notes)}")
        if best.score_breakdown.uncertainty_warnings:
            evidence_lines.append(f"Data Uncertainty: {'; '.join(best.score_breakdown.uncertainty_warnings)}")

    evidence_text = "\n".join(evidence_lines)

    # Qwen 2.5 chat template format
    system_instruction = (
        "You are an evidence-grounded GeoAI outdoor assistant. "
        "Explain the outdoor activity recommendation using ONLY the facts in the Evidence Table below.\n"
        "STRICT CONSTRAINTS:\n"
        "1. Never invent places, weather numbers, PM2.5 values, or facilities not in the Evidence Table.\n"
        "2. If 'Has Suitable Option' is False, clearly inform the user that outdoor activity is NOT recommended, "
        "cite the exact unfavorable environmental reasons, and suggest postponing or exercising indoors.\n"
        "3. If a suitable option exists, clearly state the venue, optimal time, why it ranks above alternatives, "
        "and relevant warnings.\n"
        "4. Keep the explanation concise (3-5 sentences)."
    )

    prompt = (
        f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
        f"<|im_start|>user\nUser Question: {query.raw_query}\n\n"
        f"--- EVIDENCE TABLE ---\n{evidence_text}\n--- END EVIDENCE ---\n"
        f"Provide the grounded advice:<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt
