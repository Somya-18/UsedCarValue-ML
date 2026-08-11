"""Train, compare, and persist regularized linear car-price models."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline

from .evaluate import regression_metrics
from .preprocessing import CarFeatureEngineer, FEATURE_COLUMNS, TARGET_COLUMN, build_preprocessor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42


def _pipeline(regressor: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", CarFeatureEngineer()),
            ("preprocess", build_preprocessor()),
            (
                "regressor",
                TransformedTargetRegressor(
                    regressor=regressor,
                    func=np.log1p,
                    inverse_func=np.expm1,
                    check_inverse=True,
                ),
            ),
        ]
    )


def candidate_models() -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    """Models and compact, logarithmic hyperparameter search spaces."""
    return {
        "linear_regression": (_pipeline(LinearRegression()), {}),
        "ridge": (
            _pipeline(Ridge()),
            {"regressor__regressor__alpha": list(np.logspace(-2, 3, 9))},
        ),
        "lasso": (
            _pipeline(Lasso(max_iter=20_000, selection="cyclic")),
            {"regressor__regressor__alpha": list(np.logspace(-5, -1, 9))},
        ),
        "elastic_net": (
            _pipeline(ElasticNet(max_iter=20_000, selection="cyclic")),
            {
                "regressor__regressor__alpha": list(np.logspace(-5, -1, 7)),
                "regressor__regressor__l1_ratio": [0.2, 0.5, 0.8],
            },
        ),
    }


def _validate_training_data(frame: pd.DataFrame) -> None:
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Dataset is empty.")
    if frame[TARGET_COLUMN].isna().any() or (frame[TARGET_COLUMN] <= 0).any():
        raise ValueError("Target prices must be present and strictly positive.")


def _save_coefficients(model: Pipeline, output_path: Path) -> None:
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    fitted_regressor = model.named_steps["regressor"].regressor_
    coefficients = np.asarray(fitted_regressor.coef_).ravel()
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient_log_price": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False)
    table.to_csv(output_path, index=False)


def _save_plots(results: pd.DataFrame, y_test: pd.Series, predictions: np.ndarray, report_dir: Path) -> None:
    ordered = results.sort_values("test_mae_eur")
    fig, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(ordered["model"], ordered["test_mae_eur"], color="#2563eb")
    axis.bar_label(
        bars,
        labels=[f"EUR {value:,.0f}" for value in ordered["test_mae_eur"]],
        padding=3,
    )
    axis.set_ylim(0, ordered["test_mae_eur"].max() * 1.12)
    axis.set_ylabel("Test MAE (EUR; lower is better)")
    axis.set_title("Model comparison on the untouched test set")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(report_dir / "model_comparison.png", dpi=160)
    plt.close(fig)

    residuals = np.asarray(y_test) - predictions
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(predictions, residuals, alpha=0.25, s=12)
    axis.axhline(0, color="#dc2626", linewidth=1.5)
    axis.set_xlabel("Predicted price (EUR)")
    axis.set_ylabel("Residual: actual - predicted (EUR)")
    axis.set_title("Best-model residuals")
    fig.tight_layout()
    fig.savefig(report_dir / "residuals.png", dpi=160)
    plt.close(fig)


def train(data_path: Path, model_dir: Path, report_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(data_path)
    _validate_training_data(frame)
    raw_row_count = len(frame)
    frame = frame.drop_duplicates()
    X = frame.loc[:, FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN]
    # Identical feature rows must stay in one partition, even when their prices
    # differ, or the same listing specification can leak into validation/test.
    groups = pd.util.hash_pandas_object(X, index=False).astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_positions, test_positions = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_positions], X.iloc[test_positions]
    y_train, y_test = y.iloc[train_positions], y.iloc[test_positions]
    train_groups = groups.iloc[train_positions]
    cv = GroupKFold(n_splits=5)
    rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}

    for name, (pipeline, grid) in candidate_models().items():
        started = perf_counter()
        search = GridSearchCV(
            pipeline,
            param_grid=grid or [{}],
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
        )
        search.fit(X_train, y_train, groups=train_groups)
        predictions = np.maximum(search.best_estimator_.predict(X_test), 0.0)
        train_predictions = np.maximum(search.best_estimator_.predict(X_train), 0.0)
        test_metrics = regression_metrics(y_test, predictions)
        train_metrics = regression_metrics(y_train, train_predictions)
        rows.append(
            {
                "model": name,
                "cv_mae_eur": float(-search.best_score_),
                "train_mae_eur": train_metrics["mae_eur"],
                **{f"test_{key}": value for key, value in test_metrics.items()},
                "generalization_gap_eur": test_metrics["mae_eur"] - train_metrics["mae_eur"],
                "fit_seconds": perf_counter() - started,
                "best_params": json.dumps(search.best_params_, sort_keys=True),
            }
        )
        fitted[name] = search.best_estimator_
        print(f"Finished {name}: test MAE EUR {test_metrics['mae_eur']:,.0f}")

    results = pd.DataFrame(rows).sort_values("test_mae_eur").reset_index(drop=True)
    selected_name = min(rows, key=lambda row: row["cv_mae_eur"])["model"]
    selected_model = fitted[selected_name]
    selected_predictions = np.maximum(selected_model.predict(X_test), 0.0)

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": selected_model,
        "model_name": selected_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "currency": "EUR",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_indices": X_test.index.tolist(),
        "random_state": RANDOM_STATE,
        "raw_row_count": raw_row_count,
        "deduplicated_row_count": len(frame),
    }
    joblib.dump(artifact, model_dir / "best_model.joblib")
    results.to_csv(report_dir / "model_comparison.csv", index=False)
    (model_dir / "metrics.json").write_text(
        json.dumps(
            {
                "selected_model": selected_name,
                "selection_rule": "lowest 5-fold cross-validation MAE on training data",
                "split_strategy": "grouped by identical predictor rows after exact-row deduplication",
                "dataset": {
                    "raw_rows": raw_row_count,
                    "deduplicated_rows": len(frame),
                    "training_rows": len(X_train),
                    "test_rows": len(X_test),
                },
                "models": results.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _save_coefficients(selected_model, report_dir / "feature_coefficients.csv")
    _save_plots(results, y_test, selected_predictions, report_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare used-car price models.")
    parser.add_argument("--data", type=Path, default=Path("data/autoscout_car_sales.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    results = train(args.data, args.model_dir, args.report_dir)
    print("\n", results.to_string(index=False))


if __name__ == "__main__":
    main()
