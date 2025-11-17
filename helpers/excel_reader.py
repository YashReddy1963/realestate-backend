"""
Utility to load the real estate Excel dataset once and expose safe accessors.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from django.conf import settings

EXCEL_FILE_PATH = Path(
    getattr(settings, "EXCEL_FILE_PATH", settings.BASE_DIR / "media" / "Sample_data.xlsx")
)

try:
    _DATAFRAME = pd.read_excel(EXCEL_FILE_PATH)
except FileNotFoundError as exc:  # pragma: no cover - configuration error
    raise RuntimeError(f"Excel file not found at {EXCEL_FILE_PATH}") from exc


def get_dataframe() -> pd.DataFrame:
    """Return a deep copy of the in-memory dataframe."""
    return _DATAFRAME.copy(deep=True)


def list_locations() -> List[str]:
    """Expose the distinct locations present in the dataset."""
    return sorted(
        {str(value) for value in _DATAFRAME["final location"].dropna().unique().tolist()}
    )

