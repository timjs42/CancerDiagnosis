"""Loading patient marker records from disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_PATIENTS_PATH = Path(__file__).resolve().parents[2] / "data" / "example_patients.json"


def load_patients(path: Path | str = DEFAULT_PATIENTS_PATH) -> List[Dict[str, Any]]:
    """
    Load a list of patient records from a JSON file.

    Each record is expected to be an object with a "name" field and
    numeric values for each marker in cancer_dx.model.ALL_MARKERS.

    Args:
        path: path to a JSON file containing a list of patient records.
            Defaults to the bundled example dataset.

    Returns:
        List of patient dicts, e.g. {"name": "Alice", "HE4": 180.0, ...}

    Raises:
        FileNotFoundError: if the given path does not exist.
        ValueError: if the file does not contain a JSON list.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No patient data file found at {path}")

    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list of patient records in {path}")

    return records