
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

import pandas as pd


def _sanitize_values(values: Iterable[float | int | None]) -> List[float | None]:
    sanitized: List[float | None] = []
    for value in values:
        if value is None:
            sanitized.append(None)
        elif pd.notna(value):
            sanitized.append(float(value))
        else:
            sanitized.append(None)
    return sanitized


def build_line_chart(
    df: pd.DataFrame,
    value_column: str,
    label_column: str = "year",
) -> Dict[str, List]:
    """
    Preparing a single-series payload with chronological ordering.
    """
    sorted_df = df.sort_values(label_column)
    labels = sorted_df[label_column].astype(str).tolist()
    values = _sanitize_values(sorted_df[value_column].tolist())
    return {"labels": labels, "values": values}


def build_multi_line_chart(
    frames: Mapping[str, pd.DataFrame],
    value_column: str,
    label_column: str = "year",
) -> Dict[str, Dict]:
    """
    Buildding a multi-series payload keyed by the mapping keys (e.g., locations).
    """
    combined_labels = sorted(
        {
            label
            for df in frames.values()
            for label in df[label_column].astype(str).tolist()
        }
    )

    series_payload: Dict[str, List[float | None]] = {}
    for series_name, series_df in frames.items():
        working = series_df.copy()
        working = working.set_index(working[label_column].astype(str))
        series_payload[series_name] = [
            float(working.loc[label, value_column])
            if label in working.index and pd.notna(working.loc[label, value_column])
            else None
            for label in combined_labels
        ]

    return {"labels": combined_labels, "series": series_payload}


def build_multi_line_from_columns(
    df: pd.DataFrame,
    column_map: Mapping[str, str],
    label_column: str = "year",
) -> Dict[str, Dict]:
    """
    Building a multi-series payload by reading multiple columns from the same dataframe.
    """
    sorted_df = df.sort_values(label_column)
    labels = sorted_df[label_column].astype(str).tolist()
    series_payload = {
        series_name: _sanitize_values(sorted_df[column].tolist())
        for series_name, column in column_map.items()
    }
    return {"labels": labels, "series": series_payload}


def build_bar_chart(labels: List[str], values: List[float | int | None]) -> Dict[str, List]:
    """
    Building a simple bar chart payload.
    """
    return {"labels": labels, "values": _sanitize_values(values)}

