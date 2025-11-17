
from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd


def _normalize(series: pd.Series) -> pd.Series:
    return series.fillna("").str.strip().str.lower()


def _location_mask(df: pd.DataFrame, location: str | None) -> pd.Series:
    if not location:
        return pd.Series([True] * len(df), index=df.index)
    return _normalize(df["final location"]) == location.strip().lower()


def filter_dataset(
    df: pd.DataFrame,
    location: str | None = None,
    year: int | None = None,
    year_range: Tuple[int, int] | None = None,
    last_n_years: int | None = None,
) -> pd.DataFrame:
    """
    Generic filter that can narrow down by location, a specific year, or a year range.
    """
    filtered = df[_location_mask(df, location)].copy()

    if year_range:
        start, end = sorted(year_range)
        filtered = filtered[(filtered["year"] >= start) & (filtered["year"] <= end)]

    if year is not None:
        filtered = filtered[filtered["year"] == year]

    filtered = filtered.sort_values("year").reset_index(drop=True)

    if last_n_years and not filtered.empty:
        available_years = sorted(filtered["year"].unique())[-last_n_years:]
        filtered = filtered[filtered["year"].isin(available_years)].reset_index(drop=True)

    return filtered


def filter_two_locations(
    df: pd.DataFrame,
    first_location: str,
    second_location: str,
    year: int | None = None,
    year_range: Tuple[int, int] | None = None,
    last_n_years: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return two independent dataframes for the requested locations with identical filters.
    """
    return (
        filter_dataset(df, first_location, year=year, year_range=year_range, last_n_years=last_n_years),
        filter_dataset(df, second_location, year=year, year_range=year_range, last_n_years=last_n_years),
    )


def latest_records_per_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the latest entry per location (highest year).
    """
    if df.empty:
        return df
    sorted_df = df.sort_values(["final location", "year"])
    idx = sorted_df.groupby("final location")["year"].idxmax()
    return sorted_df.loc[idx].reset_index(drop=True)


def ranking_frame(
    df: pd.DataFrame,
    year: int | None = None,
) -> pd.DataFrame:
    """
    Prepare a dataframe that contains one entry per location for ranking outputs.
    """
    working = df.copy()
    if year is not None:
        working = working[working["year"] == year]
    else:
        working = latest_records_per_location(working)
    return working.reset_index(drop=True)


def aggregate_across_locations(
    df: pd.DataFrame,
    locations: Iterable[str] | None = None,
    year: int | None = None,
    year_range: Tuple[int, int] | None = None,
    last_n_years: int | None = None,
) -> pd.DataFrame:
    """
    Return a filtered dataframe spanning multiple (or all) locations with optional time bounds.
    """
    working = df.copy()
    if locations:
        mask = _normalize(working["final location"]).isin([loc.strip().lower() for loc in locations])
        working = working[mask]
    if year_range:
        start, end = sorted(year_range)
        working = working[(working["year"] >= start) & (working["year"] <= end)]
    if year is not None:
        working = working[working["year"] == year]
    working = working.sort_values(["final location", "year"]).reset_index(drop=True)

    if last_n_years and not working.empty:
        available_years = sorted(working["year"].unique())[-last_n_years:]
        working = working[working["year"].isin(available_years)].reset_index(drop=True)
    return working

