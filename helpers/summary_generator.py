"""
Generate lightweight natural language summaries without external APIs.
"""
from __future__ import annotations

from math import isfinite
from typing import Dict, List

from .query_parser import ParsedQuery

METRIC_MESSAGES = {
    "price": "pricing trends",
    "demand": "buyer demand",
    "sales": "sales values",
    "supply": "new supply and inventory levels",
    "general": "overall performance metrics",
}

PROPERTY_LABELS = {
    "flat": "flats",
    "office": "offices",
    "shop": "shops",
    "carpet_area": "carpet area supplied",
}


def _metric_phrase(parsed: ParsedQuery) -> str:
    base = METRIC_MESSAGES.get(parsed.metric, METRIC_MESSAGES["general"])
    if parsed.property_types:
        property_phrase = " & ".join(PROPERTY_LABELS.get(prop, prop) for prop in parsed.property_types)
        return f"{base} for {property_phrase}"
    return base


def _time_phrase(parsed: ParsedQuery) -> str:
    if parsed.year:
        return f"in {parsed.year}"
    if parsed.year_range:
        start, end = parsed.year_range
        return f"from {start} to {end}"
    if parsed.last_n_years:
        return f"across the last {parsed.last_n_years} years"
    return "across the available years"


def _format_value(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    if not isinstance(value, (int, float)):
        return str(value)
    if not isfinite(value):
        return "N/A"

    absolute = abs(value)
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.1f}K"
    if absolute.is_integer():
        return f"{int(value)}"
    return f"{value:.2f}"


def _stats_sentence(stats: Dict[str, Dict[str, float | int | None]]) -> str:
    if not stats:
        return ""
    parts: List[str] = []
    if "average" in stats and "value" in stats["average"]:
        parts.append(f"average {_format_value(stats['average']['value'])}")
    if "max" in stats:
        value = _format_value(stats["max"].get("value"))
        year = stats["max"].get("year")
        suffix = f" in {year}" if year else ""
        parts.append(f"peak of {value}{suffix}")
    if "min" in stats:
        value = _format_value(stats["min"].get("value"))
        year = stats["min"].get("year")
        suffix = f" in {year}" if year else ""
        parts.append(f"lowest value of {value}{suffix}")
    if "growth_rate" in stats:
        rate = stats["growth_rate"].get("value")
        direction = stats["growth_rate"].get("direction", "change")
        if rate is not None:
            parts.append(f"{direction} of {_format_value(rate)}% across the selected years")
    if not parts:
        return ""
    return " Key stats: " + "; ".join(parts) + "."


def generate_summary(parsed: ParsedQuery, context: Dict | None = None) -> str:
    """
    Produce a deterministic, human-readable summary based on the parsed intent.
    """
    context = context or {}

    if parsed.coordinate:
        location = parsed.locations[0] if parsed.locations else "the requested location"
        lat = context.get("coordinate", {}).get("lat")
        lng = context.get("coordinate", {}).get("lng")
        if lat is None or lng is None:
            return f"Latitude/longitude for {location} could not be located in the dataset."
        return f"Coordinates for {location}: lat {_format_value(lat)}, lng {_format_value(lng)}."

    if parsed.near:
        location = parsed.locations[0] if parsed.locations else "the target locality"
        nearby = context.get("nearby", [])
        if not nearby:
            return f"No nearby localities were detected for {location}."
        names = ", ".join(entry["final location"] for entry in nearby)
        return f"Localities closest to {location} include {names}."

    if parsed.ranking:
        ranking_rows = context.get("ranking", [])
        if not ranking_rows:
            return "No ranking data was available for the requested metric."
        top = ranking_rows[0]
        time_phrase = f"in {top.get('year')}" if top.get("year") else "based on the latest data"
        return (
            f"{top['final location']} leads {_metric_phrase(parsed)} {time_phrase} "
            f"with {_format_value(top.get('value'))}."
        )

    if parsed.comparison and len(parsed.locations) >= 2:
        winner = context.get("comparison", {}).get("winner")
        loser = context.get("comparison", {}).get("loser")
        if winner and loser:
            diff = context.get("comparison", {}).get("difference")
            diff_phrase = f" by {_format_value(diff)}" if diff is not None else ""
            return (
                f"{winner} currently outperforms {loser} for {_metric_phrase(parsed)}"
                f"{diff_phrase}. Use the chart to inspect the year-wise divergence."
            )
        return (
            f"Comparing {_metric_phrase(parsed)} between {parsed.locations[0]} and {parsed.locations[1]} "
            f"{_time_phrase(parsed)}."
        )

    if parsed.general_insight and parsed.locations:
        return (
            f"Overview of {_metric_phrase(parsed)} for {parsed.locations[0]} {_time_phrase(parsed)}."
        )

    if parsed.all_locations and not parsed.locations:
        stat_sentence = _stats_sentence(context.get("stats", {}))
        return f"Aggregated {_metric_phrase(parsed)} across Pune {_time_phrase(parsed)}.{stat_sentence}"

    if not parsed.locations:
        return "No matching location was found in the dataset. Please refine the query."

    base = (
        f"Showing {_metric_phrase(parsed)} for {parsed.locations[0]} {_time_phrase(parsed)}."
    )
    stat_sentence = _stats_sentence(context.get("stats", {}))
    return base + stat_sentence

