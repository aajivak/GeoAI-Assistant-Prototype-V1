"""Deterministic evidence-grounded explanation generator.

Produces transparent, factual natural-language synthesis directly from the structured evidence.
Serves as both the baseline grounded explainer and the infallible fallback if SLM output fails validation.
"""
from src.common.schemas import ParsedQuery
from src.scoring.candidate_ranker import CandidateRankingResult


def generate_deterministic_explanation(
    query: ParsedQuery,
    ranking: CandidateRankingResult,
) -> str:
    """Generate an evidence-grounded explanation strictly referencing the evaluated metrics."""
    activity_name = (query.activity or "outdoor activity").title()
    anchor = query.spatial_anchor or "your designated area"

    # Scenario 1: No suitable options found across the evaluated window
    if not ranking.has_suitable_option:
        best = ranking.best_recommendation
        parts = [
            f"⚠️ NO SUITABLE OPTION FOUND: Outdoor {activity_name.lower()} is currently not recommended near {anchor}.",
        ]
        if ranking.no_suitable_option_reason:
            parts.append(f"\n{ranking.no_suitable_option_reason}")

        if best:
            parts.append(
                f"\nFor reference, the highest relative slot was {best.venue_name} at {best.slot_time_ist}, "
                f"which only scored {best.suitability_score}/100 (Status: {best.recommendation_status.upper()}). "
                f"Observed environmental metrics: Temp {best.weather_summary.get('temperature_c')}°C, "
                f"PM2.5 {best.aqi_summary.get('pm2_5')} µg/m³, US AQI {best.aqi_summary.get('us_aqi')}."
            )
            if best.score_breakdown.unfavorable_reasons:
                parts.append(f"Exclusions: {'; '.join(best.score_breakdown.unfavorable_reasons)}")

        parts.append(
            "\nRecommended Action: Consider exercising indoors or postponing your session until ambient air quality improves."
        )
        return "\n".join(parts)

    # Scenario 2: Favorable or acceptable slot found
    best = ranking.best_recommendation
    if not best:
        return f"No venue or weather records were available to evaluate {activity_name} near {anchor}."

    w = best.weather_summary
    a = best.aqi_summary
    status_tag = "RECOMMENDED" if best.recommendation_status == "recommended" else "EXERCISE WITH CAUTION"

    lines = [
        f"🎯 {status_tag}: Best place and time for {activity_name.lower()} is {best.venue_name} at {best.slot_time_ist}.",
        f"• Suitability Score: {best.suitability_score}/100 ({best.rating_category})",
        f"• Location: {best.distance_km:.2f} km from {anchor} ({best.place_type.replace('_', ' ')})",
        f"• Weather: {w.get('temperature_c')}°C, Humidity {w.get('humidity_percent')}%, Rain Probability {w.get('rain_probability_percent')}%, Wind {w.get('wind_speed_kmh')} km/h",
    ]

    if a.get("pm2_5") is not None:
        lines.append(f"• Air Quality: PM2.5 {a.get('pm2_5')} µg/m³ (US AQI: {a.get('us_aqi')})")

    # Geographic indicators
    if best.geographic_context and best.geographic_context.indicators:
        for ind in best.geographic_context.indicators[:2]:
            lines.append(f"• Geographic Context: {ind.description}")

    # Temporal comparison rationale
    if ranking.temporal_comparison_note:
        lines.append(f"• Timing Rationale: {ranking.temporal_comparison_note}")

    # Health / Personal Sensitivity Notes
    if best.score_breakdown.heuristic_notes:
        lines.append(f"• Advisories & Preparation: {'; '.join(best.score_breakdown.heuristic_notes)}")

    # Uncertainty Warnings
    if best.score_breakdown.uncertainty_warnings:
        lines.append(f"• Data Uncertainty: {'; '.join(best.score_breakdown.uncertainty_warnings)}")

    return "\n".join(lines)
