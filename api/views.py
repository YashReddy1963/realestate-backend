
from __future__ import annotations

import csv
import json
from math import atan2, cos, radians, sin, sqrt
from typing import Dict, List, Mapping

import pandas as pd
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from helpers.chart_builder import (
    build_bar_chart,
    build_line_chart,
    build_multi_line_chart,
    build_multi_line_from_columns,
)
from helpers.data_filter import (
    aggregate_across_locations,
    filter_dataset,
    filter_two_locations,
    latest_records_per_location,
    ranking_frame,
)
from helpers.excel_reader import get_dataframe, list_locations
from helpers.query_parser import ParsedQuery, parse_query
from helpers.summary_generator import generate_summary

BASE_METRIC_COLUMNS = {
    "price": "flat - weighted average rate",
    "demand": "total sold - igr",
    "supply": "total units",
    "sales": "total_sales - igr",
    "general": "total_sales - igr",
}

PROPERTY_METRIC_COLUMNS = {
    "price": {
        "flat": "flat - weighted average rate",
        "office": "office - weighted average rate",
        "shop": "shop - weighted average rate",
    },
    "demand": {
        "flat": "flat_sold - igr",
        "office": "office_sold - igr",
        "shop": "shop_sold - igr",
    },
    "supply": {
        "flat": "flat total",
        "office": "office total",
        "shop": "shop total",
        "carpet_area": "total carpet area supplied (sqft)",
    },
    "sales": {
        "flat": "flat_sold - igr",
        "office": "office_sold - igr",
        "shop": "shop_sold - igr",
    },
}

PROPERTY_LABELS = {
    "flat": "Flat",
    "office": "Office",
    "shop": "Shop",
    "carpet_area": "Carpet Area",
}

RANKING_KEYWORDS_LIMIT = ("which", "highest", "lowest", "most", "least")
DEFAULT_RANKING_LIMIT = 5
NEARBY_LIMIT = 3


def _metric_column_for_property(metric: str, property_type: str | None) -> str | None:
    if not property_type:
        return None
    metric_map = PROPERTY_METRIC_COLUMNS.get(metric, {})
    return metric_map.get(property_type)


def _resolve_value_column(parsed: ParsedQuery) -> str:
    if parsed.property_types:
        first_column = _metric_column_for_property(parsed.metric, parsed.property_types[0])
        if first_column:
            return first_column
    return BASE_METRIC_COLUMNS.get(parsed.metric, BASE_METRIC_COLUMNS["general"])


def _multi_property_columns(parsed: ParsedQuery) -> Mapping[str, str]:
    columns: Dict[str, str] = {}
    if len(parsed.property_types) <= 1:
        return columns
    for property_type in parsed.property_types:
        column = _metric_column_for_property(parsed.metric, property_type)
        if column:
            label = PROPERTY_LABELS.get(property_type, property_type.title())
            columns[f"{label} {_metric_label(parsed.metric)}"] = column
    return columns


def _metric_label(metric: str) -> str:
    return {
        "price": "Price",
        "demand": "Demand",
        "supply": "Supply",
        "sales": "Sales",
        "general": "Metric",
    }.get(metric, "Metric")


def _compute_stats(df: pd.DataFrame, column: str, requested_stats: List[str]) -> Dict[str, Dict]:
    stats: Dict[str, Dict] = {}
    if column not in df.columns or df.empty or not requested_stats:
        return stats

    series = df[column]
    numeric_series = series.dropna()
    if numeric_series.empty:
        return stats

    if "average" in requested_stats:
        stats["average"] = {"value": float(numeric_series.mean())}

    if "max" in requested_stats:
        idx_max = numeric_series.idxmax()
        stats["max"] = {
            "value": float(series.loc[idx_max]),
            "year": int(df.loc[idx_max, "year"]) if "year" in df.columns else None,
        }

    if "min" in requested_stats:
        idx_min = numeric_series.idxmin()
        stats["min"] = {
            "value": float(series.loc[idx_min]),
            "year": int(df.loc[idx_min, "year"]) if "year" in df.columns else None,
        }

    if "growth_rate" in requested_stats and len(numeric_series) >= 2:
        sorted_df = df.sort_values("year")
        first_value = sorted_df[column].dropna().iloc[0]
        last_value = sorted_df[column].dropna().iloc[-1]
        if pd.notna(first_value) and first_value != 0:
            growth = ((last_value - first_value) / first_value) * 100
            stats["growth_rate"] = {
                "value": round(float(growth), 2),
                "direction": "increase" if growth >= 0 else "decrease",
            }
    return stats


def _resolve_ranking_limit(parsed: ParsedQuery) -> int:
    if parsed.ranking_limit:
        return parsed.ranking_limit
    lowered = parsed.raw_query.lower()
    if any(keyword in lowered for keyword in RANKING_KEYWORDS_LIMIT):
        return 1
    return DEFAULT_RANKING_LIMIT


def _distance_km(first_row: pd.Series, second_row: pd.Series) -> float:
    R = 6371  # Earth radius in km
    lat1, lon1 = radians(first_row["loc_lat"]), radians(first_row["loc_lng"])
    lat2, lon2 = radians(second_row["loc_lat"]), radians(second_row["loc_lng"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def _nearby_localities(df: pd.DataFrame, location: str) -> List[Dict]:
    latest = latest_records_per_location(df)
    base_row = latest[_normalize_series(latest["final location"]) == location.lower()]
    if base_row.empty:
        return []
    base = base_row.iloc[0]
    others = latest[_normalize_series(latest["final location"]) != location.lower()].copy()
    if others.empty:
        return []
    others["distance_km"] = others.apply(lambda row: round(_distance_km(base, row), 2), axis=1)
    nearest = others.sort_values("distance_km").head(NEARBY_LIMIT)
    return nearest[["final location", "loc_lat", "loc_lng", "distance_km"]].to_dict(orient="records")


def _normalize_series(series: pd.Series) -> pd.Series:
    return series.fillna("").str.strip().str.lower()


def _build_response(summary: str, chart_type: str, chart_data: Dict, table_data: List[Dict]) -> JsonResponse:
    return JsonResponse(
        {
            "summary": summary,
            "chart_type": chart_type,
            "chart_data": chart_data,
            "table_data": table_data,
        }
    )


def _handle_coordinate(df: pd.DataFrame, parsed: ParsedQuery) -> JsonResponse:
    if not parsed.locations:
        return JsonResponse({"detail": "Please mention a location for coordinate queries."}, status=400)
    location = parsed.locations[0]
    filtered = filter_dataset(df, location)
    if filtered.empty:
        return JsonResponse({"detail": "Requested location is missing in the data."}, status=404)
    latest_row = filtered.sort_values("year").iloc[-1]
    coordinate = {"lat": latest_row.get("loc_lat"), "lng": latest_row.get("loc_lng")}

    context: Dict[str, Dict] = {"coordinate": coordinate}
    table_data = [
        {"final location": location, "loc_lat": coordinate["lat"], "loc_lng": coordinate["lng"]}
    ]

    if parsed.near:
        nearby = _nearby_localities(df, location)
        context["nearby"] = nearby
        table_data = nearby

    summary = generate_summary(parsed, context)
    return _build_response(summary, "none", {}, table_data)


def _handle_ranking(df: pd.DataFrame, parsed: ParsedQuery) -> JsonResponse:
    value_column = _resolve_value_column(parsed)
    ranking_df = ranking_frame(df, year=parsed.year)
    if value_column not in ranking_df.columns:
        return JsonResponse({"detail": f"Column '{value_column}' is missing in the data."}, status=400)
    ranking_df = ranking_df.dropna(subset=[value_column])
    if ranking_df.empty:
        return JsonResponse({"detail": "No data available for ranking."}, status=404)

    ranking_df = ranking_df.sort_values(
        value_column,
        ascending=parsed.ranking_order == "asc",
    )
    ranking_df = ranking_df.head(_resolve_ranking_limit(parsed))
    labels = ranking_df["final location"].tolist()
    values = ranking_df[value_column].tolist()
    chart_data = build_bar_chart(labels, values)

    table_records: List[Dict] = []
    for idx, (_, row) in enumerate(ranking_df.iterrows(), start=1):
        table_records.append(
            {
                "rank": idx,
                "final location": row["final location"],
                "value": row[value_column],
                "year": row.get("year"),
            }
        )

    context = {"ranking": table_records}
    summary = generate_summary(parsed, context)
    return _build_response(summary, "bar", chart_data, table_records)


def _handle_comparison(df: pd.DataFrame, parsed: ParsedQuery) -> JsonResponse:
    if len(parsed.locations) < 2:
        return JsonResponse({"detail": "Please mention two locations for comparison."}, status=400)
    value_column = _resolve_value_column(parsed)
    first_location, second_location = parsed.locations[:2]
    first_df, second_df = filter_two_locations(
        df,
        first_location,
        second_location,
        year=parsed.year,
        year_range=parsed.year_range,
        last_n_years=parsed.last_n_years,
    )

    if first_df.empty or second_df.empty:
        return JsonResponse({"detail": "Requested locations are missing in the data."}, status=404)
    if value_column not in first_df.columns or value_column not in second_df.columns:
        return JsonResponse({"detail": f"Column '{value_column}' is missing in the data."}, status=400)

    chart_data = build_multi_line_chart(
        {
            first_location: first_df,
            second_location: second_df,
        },
        value_column=value_column,
    )
    table_data = pd.concat([first_df, second_df]).sort_values(["final location", "year"])
    table_records = table_data.to_dict(orient="records")

    comparison_context = {}
    first_latest = first_df.sort_values("year").iloc[-1][value_column]
    second_latest = second_df.sort_values("year").iloc[-1][value_column]
    if pd.notna(first_latest) and pd.notna(second_latest):
        if first_latest > second_latest:
            comparison_context = {
                "winner": first_location,
                "loser": second_location,
                "difference": float(first_latest) - float(second_latest),
            }
        elif second_latest > first_latest:
            comparison_context = {
                "winner": second_location,
                "loser": first_location,
                "difference": float(second_latest) - float(first_latest),
            }

    summary = generate_summary(parsed, {"comparison": comparison_context})
    return _build_response(summary, "multi_line", chart_data, table_records)


def _handle_single_location(df: pd.DataFrame, parsed: ParsedQuery) -> JsonResponse:
    value_column = _resolve_value_column(parsed)
    multi_property_columns = _multi_property_columns(parsed)

    if parsed.all_locations and not parsed.locations:
        working = aggregate_across_locations(
            df,
            year=parsed.year,
            year_range=parsed.year_range,
            last_n_years=parsed.last_n_years,
        )
        if working.empty:
            return JsonResponse({"detail": "No matching data was found for Pune-wide query."}, status=404)

        if value_column not in working.columns:
            return JsonResponse({"detail": f"Column '{value_column}' is missing in the data."}, status=400)

        if parsed.year and not parsed.year_range:
            total_value = working[value_column].sum()
            chart_data = build_bar_chart([str(parsed.year)], [total_value])
            table_records = [{"year": parsed.year, value_column: total_value}]
        else:
            grouped = working.groupby("year", as_index=False)[value_column].sum()
            chart_data = build_line_chart(grouped, value_column)
            table_records = grouped.to_dict(orient="records")

        stats = _compute_stats(pd.DataFrame(table_records), value_column, parsed.stats)
        summary = generate_summary(parsed, {"stats": stats})
        return _build_response(summary, "line" if len(table_records) > 1 else "bar", chart_data, table_records)

    if not parsed.locations:
        return JsonResponse({"detail": "Please mention at least one location in your query."}, status=400)

    location = parsed.locations[0]
    filtered_df = filter_dataset(
        df,
        location=location,
        year=parsed.year,
        year_range=parsed.year_range,
        last_n_years=parsed.last_n_years,
    )
    if filtered_df.empty:
        return JsonResponse({"detail": "Requested location is missing in the data."}, status=404)
    if value_column not in filtered_df.columns:
        return JsonResponse({"detail": f"Column '{value_column}' is missing in the data."}, status=400)

    if multi_property_columns:
        chart_data = build_multi_line_from_columns(filtered_df, multi_property_columns)
        chart_type = "multi_line"
    else:
        chart_data = build_line_chart(filtered_df, value_column)
        chart_type = "line"

    stats = _compute_stats(filtered_df, value_column, parsed.stats)
    summary = generate_summary(parsed, {"stats": stats})
    table_records = filtered_df.to_dict(orient="records")
    return _build_response(summary, chart_type, chart_data, table_records)


@csrf_exempt
@require_POST
def handle_query(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"detail": "Invalid JSON body."}, status=400)

    query = (payload.get("query") or "").strip()
    if not query:
        return JsonResponse({"detail": "Query text is required."}, status=400)

    df = get_dataframe()
    parsed = parse_query(query, list_locations())

    if parsed.coordinate or parsed.near:
        return _handle_coordinate(df, parsed)
    if parsed.ranking:
        return _handle_ranking(df, parsed)
    if parsed.comparison and len(parsed.locations) >= 2:
        return _handle_comparison(df, parsed)

    return _handle_single_location(df, parsed)


@require_GET
def download_filtered_csv(request):
    location = (request.GET.get("location") or "").strip()
    if not location:
        return JsonResponse({"detail": "location query parameter is required."}, status=400)

    df = get_dataframe()
    filtered_df = filter_dataset(df, location)
    if filtered_df.empty:
        return JsonResponse({"detail": "Requested location is missing in the data."}, status=404)

    csv_buffer = filtered_df.to_csv(
        index=False,
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )

    response = HttpResponse(csv_buffer, content_type="text/csv; charset=utf-8")
    safe_location = location.replace(" ", "_")
    response["Content-Disposition"] = f'attachment; filename="{safe_location}_data.csv"'
    return response
