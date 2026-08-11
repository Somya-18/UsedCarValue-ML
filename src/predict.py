"""Command-line inference for one record or a JSON array of records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .preprocessing import FEATURE_COLUMNS


def load_records(path: Path) -> pd.DataFrame:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else [payload]
    frame = pd.DataFrame(records)
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required fields: {missing}")
    return frame.loc[:, FEATURE_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict used-car prices from JSON.")
    parser.add_argument("--model", type=Path, default=Path("models/best_model.joblib"))
    parser.add_argument("--input", type=Path, required=True, help="JSON object or array of objects")
    args = parser.parse_args()
    artifact = joblib.load(args.model)
    frame = load_records(args.input)
    predictions = np.maximum(artifact["model"].predict(frame), 0.0)
    output = [
        {"predicted_price": round(float(value), 2), "currency": artifact.get("currency", "EUR")}
        for value in predictions
    ]
    print(json.dumps(output[0] if len(output) == 1 else output, indent=2))


if __name__ == "__main__":
    main()

