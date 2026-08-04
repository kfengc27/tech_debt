"""Helpers for loading repository complexity history for Django charts."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


def load_evolution_data(
    repository_name: str,
    results_dir: str | Path = "complexity_results",
) -> list[dict[str, Any]]:
    """Load and clean the CSV produced by complexity_over_time_v2.py.

    The returned structure is JSON-serialisable and can be passed directly to
    Django's ``json_script`` template filter.
    """
    csv_path = Path(results_dir) / f"{safe_name(repository_name)}_complexity.csv"

    if not csv_path.exists():
        return []

    try:
        frame = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return []

    required = {"date", "total_complexity"}
    if not required.issubset(frame.columns):
        return []

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["total_complexity"] = pd.to_numeric(
        frame["total_complexity"], errors="coerce"
    )
    frame = frame.dropna(subset=["date", "total_complexity"])
    frame = frame.sort_values("date").drop_duplicates(
        subset=["date", "commit"] if "commit" in frame.columns else ["date"],
        keep="last",
    )

    optional_numeric = [
        "avg_complexity",
        "function_count",
        "nloc",
        "file_count",
    ]
    for column in optional_numeric:
        if column not in frame.columns:
            frame[column] = 0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

    if "commit" not in frame.columns:
        frame["commit"] = ""

    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        records.append(
            {
                "date": row.date.strftime("%Y-%m-%d"),
                "commit": str(row.commit),
                "total_complexity": float(row.total_complexity),
                "avg_complexity": float(row.avg_complexity),
                "function_count": int(row.function_count),
                "nloc": int(row.nloc),
                "file_count": int(row.file_count),
            }
        )

    return records