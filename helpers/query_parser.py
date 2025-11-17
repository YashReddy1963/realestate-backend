"""
Extended rule-based parser for extracting intent, locations, metric, timeframe,
and other modifiers from a natural language query.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

from .excel_reader import list_locations


@dataclass
class ParsedQuery:
    raw_query: str
    locations: List[str]
    metric: str
    property_types: List[str] = field(default_factory=list)
    comparison: bool = False
    ranking: bool = False
    ranking_order: str = "desc"
    ranking_limit: int | None = None
    coordinate: bool = False
    near: bool = False
    year: int | None = None
    year_range: Tuple[int, int] | None = None
    last_n_years: int | None = None
    stats: List[str] = field(default_factory=list)
    extrema: bool = False
    general_insight: bool = False
    all_locations: bool = False


METRIC_KEYWORDS = {
    "price": (
        "price",
        "prices",
        "rate",
        "rates",
        "valuation",
        "cost",
        "pricing",
    ),
    "demand": (
        "demand",
        "sold",
        "sales",
        "absorption",
        "bookings",
        "units sold",
    ),
    "supply": (
        "supply",
        "supplied",
        "inventory",
        "stock",
        "pipeline",
        "units available",
        "carpet area",
    ),
    "sales": (
        "total_sales",
        "total sales",
        "revenue",
        "turnover",
    ),
}

PROPERTY_KEYWORDS = {
    "flat": ("flat", "flats", "apartment", "apartments", "residential"),
    "office": ("office", "offices"),
    "shop": ("shop", "shops", "retail", "stores"),
}

SPECIAL_PROPERTY_KEYWORDS = {
    "carpet_area": ("carpet area", "carpet sq", "sqft supplied"),
}

COMPARISON_KEYWORDS = (
    "compare",
    " vs ",
    " vs. ",
    "vs",
    "v/s",
    "versus",
    "between",
    "against",
    "compared to",
    "better than",
    "performed better",
)

GENERAL_INSIGHT_KEYWORDS = (
    "overview",
    "insight",
    "analysis",
    "summary",
    "summarize",
    "explain",
    "trend",
    "outlook",
)

COORDINATE_KEYWORDS = (
    "coordinate",
    "coordinates",
    "lat",
    "lng",
    "longitude",
    "latitude",
    "location pin",
)

NEAR_KEYWORDS = ("near", "nearby", "close to", "around")

RANKING_DESC_KEYWORDS = ("highest", "top", "most", "fastest", "best")
RANKING_ASC_KEYWORDS = ("lowest", "least", "slowest", "bottom", "worst", "minimum")

STAT_KEYWORDS = {
    "average": ("average", "avg", "mean"),
    "max": ("highest", "max", "peak"),
    "min": ("lowest", "min", "minimum"),
    "growth_rate": ("growth rate", "growth", "increase", "rise", "drop", "decline"),
}

ALL_LOCATIONS_KEYWORDS = (
    "all locations",
    "across all",
    "overall",
    "entire city",
    "whole city",
    "across pune",
    "citywide",
    "across the city",
)

YEAR_RANGE_REGEX = re.compile(r"(20\d{2})\s*(?:to|-|through|until|till|and)\s*(20\d{2})")
YEAR_REGEX = re.compile(r"(20\d{2})")
LAST_N_YEARS_REGEX = re.compile(r"(?:last|past)\s+(\d+)\s+years?")
TOP_N_REGEX = re.compile(r"top\s+(\d+)")

DEFAULT_LOCATIONS = list_locations()


def _match_locations(query: str, locations: Sequence[str]) -> List[str]:
    found: List[str] = []
    lowered_query = query.lower()

    for location in locations:
        pattern = re.compile(rf"\b{re.escape(location.lower())}\b")
        if pattern.search(lowered_query):
            found.append(location)
        if len(found) == 2:
            break

    return found


def _detect_metric(lowered_query: str) -> str:
    for metric, keywords in METRIC_KEYWORDS.items():
        if any(keyword in lowered_query for keyword in keywords):
            return metric
    return "general"


def _detect_property_types(lowered_query: str) -> List[str]:
    detected: List[str] = []
    for property_type, keywords in PROPERTY_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(keyword)}\b", lowered_query) for keyword in keywords):
            detected.append(property_type)

    for special_type, keywords in SPECIAL_PROPERTY_KEYWORDS.items():
        if any(keyword in lowered_query for keyword in keywords):
            detected.append(special_type)

    # Remove duplicates while preserving order
    seen = set()
    unique_types: List[str] = []
    for property_type in detected:
        if property_type not in seen:
            unique_types.append(property_type)
            seen.add(property_type)
    return unique_types


def _extract_years(lowered_query: str) -> tuple[int | None, Tuple[int, int] | None]:
    range_match = YEAR_RANGE_REGEX.search(lowered_query)
    if range_match:
        start, end = sorted(int(value) for value in range_match.groups())
        return None, (start, end)

    years = [int(value) for value in YEAR_REGEX.findall(lowered_query)]
    unique_years = sorted(set(years))
    if len(unique_years) == 1:
        return unique_years[0], None
    if len(unique_years) >= 2:
        return None, (unique_years[0], unique_years[-1])
    return None, None


def _extract_last_n_years(lowered_query: str) -> int | None:
    match = LAST_N_YEARS_REGEX.search(lowered_query)
    if match:
        try:
            value = int(match.group(1))
            return value if value > 0 else None
        except ValueError:  # pragma: no cover - defensive
            return None
    return None


def _detect_ranking(lowered_query: str) -> tuple[bool, str, int | None]:
    ranking = _detect_boolean(lowered_query, RANKING_DESC_KEYWORDS + RANKING_ASC_KEYWORDS, word_boundary=True)
    order = "desc"
    if _detect_boolean(lowered_query, RANKING_ASC_KEYWORDS, word_boundary=True):
        order = "asc"

    match = TOP_N_REGEX.search(lowered_query)
    limit = None
    if match:
        try:
            limit = int(match.group(1))
            ranking = True
            order = "desc"
        except ValueError:  # pragma: no cover - defensive
            limit = None
    return ranking, order, limit


def _detect_stats(lowered_query: str) -> List[str]:
    stats: List[str] = []
    for stat, keywords in STAT_KEYWORDS.items():
        if any(keyword in lowered_query for keyword in keywords):
            stats.append(stat)
    return stats


def _detect_boolean(lowered_query: str, keywords: Sequence[str], word_boundary: bool = False) -> bool:
    if word_boundary:
        return any(re.search(rf"\b{re.escape(keyword)}\b", lowered_query) for keyword in keywords)
    return any(keyword in lowered_query for keyword in keywords)


def parse_query(query: str, locations: Iterable[str] | None = None) -> ParsedQuery:
    """
    Parse the user sentence and return extracted metadata.
    """
    trimmed_query = (query or "").strip()
    lowered_query = trimmed_query.lower()
    candidate_locations = list(locations) if locations is not None else DEFAULT_LOCATIONS

    parsed_locations = _match_locations(trimmed_query, candidate_locations)
    metric = _detect_metric(lowered_query)
    property_types = _detect_property_types(lowered_query)
    year, year_range = _extract_years(lowered_query)
    last_n_years = _extract_last_n_years(lowered_query)
    ranking_flag, ranking_order, ranking_limit = _detect_ranking(lowered_query)
    stats = _detect_stats(lowered_query)
    coordinate = _detect_boolean(lowered_query, COORDINATE_KEYWORDS, word_boundary=True)
    near = _detect_boolean(lowered_query, NEAR_KEYWORDS, word_boundary=True)
    all_locations = _detect_boolean(lowered_query, ALL_LOCATIONS_KEYWORDS)

    explicit_comparison = any(keyword in lowered_query for keyword in COMPARISON_KEYWORDS)
    comparison = explicit_comparison or len(parsed_locations) >= 2

    # Ranking keywords used with a single location indicate local extrema instead of cross-location ranking.
    ranking = ranking_flag and not parsed_locations
    extrema = ranking_flag and bool(parsed_locations)

    general_insight = metric == "general" or any(
        keyword in lowered_query for keyword in GENERAL_INSIGHT_KEYWORDS
    )

    return ParsedQuery(
        raw_query=trimmed_query,
        locations=parsed_locations,
        metric=metric,
        property_types=property_types,
        comparison=comparison,
        ranking=ranking,
        ranking_order=ranking_order,
        ranking_limit=ranking_limit,
        coordinate=coordinate,
        near=near,
        year=year,
        year_range=year_range,
        last_n_years=last_n_years,
        stats=stats,
        extrema=extrema,
        general_insight=general_insight,
        all_locations=all_locations,
    )

