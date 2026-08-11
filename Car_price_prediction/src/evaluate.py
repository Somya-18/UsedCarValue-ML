"""Evaluation utilities and a CLI for a saved model artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

from .preprocessing import FEATURE_COLUMNS, TARGET_COLUMN


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return interpretable regression metrics on the original euro scale."""
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    return {
        "mae_eur": float(mean_absolute_error(y_true_array, y_pred_array)),
        "rmse_eur": float(mean_squared_error(y_true_array, y_pred_array) ** 0.5),
        "mape": float(mean_absolute_percentage_error(y_true_array, y_pred_array)),
        "r2": float(r2_score(y_true_array, y_pred_array)),
    }


def evaluate_model(model: Any, frame: pd.DataFrame) -> tuple[dict[str, float], np.ndarray]:
    predictions = np.maximum(model.predict(frame.loc[:, FEATURE_COLUMNS]), 0.0)
    return regression_metrics(frame[TARGET_COLUMN], predictions), predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved car-price model.")
    parser.add_argument("--model", type=Path, default=Path("models/best_model.joblib"))
    parser.add_argument("--data", type=Path, default=Path("data/autoscout_car_sales.csv"))
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Evaluate all rows. By default, reuse the artifact's untouched test rows.",
    )
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    frame = pd.read_csv(args.data)
    if not args.full_data:
        test_indices = artifact.get("test_indices")
        if test_indices is None:
            raise ValueError("Artifact has no stored test indices; pass --full-data to continue.")
        frame = frame.loc[test_indices]

    metrics, _ = evaluate_model(artifact["model"], frame)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

