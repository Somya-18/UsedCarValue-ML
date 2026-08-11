"""Feature engineering and leakage-safe preprocessing for car listings."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "price"

LIST_COLUMNS = [
    "Comfort_Convenience",
    "Entertainment_Media",
    "Extras",
    "Safety_Security",
]

RAW_NUMERIC_COLUMNS = [
    "km",
    "Gears",
    "age",
    "Previous_Owners",
    "hp_kW",
    "Inspection_new",
    "Displacement_cc",
    "Weight_kg",
    "cons_comb",
]

ENGINEERED_NUMERIC_COLUMNS = [
    "num_comfort_convenience",
    "num_entertainment_media",
    "num_extras",
    "num_safety_security",
    "km_per_year",
    "power_to_weight",
]

CATEGORICAL_COLUMNS = [
    "make_model",
    "body_type",
    "vat",
    "Type",
    "Fuel",
    "Paint_Type",
    "Upholstery_type",
    "Gearing_Type",
    "Drive_chain",
]

FEATURE_COLUMNS = RAW_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + LIST_COLUMNS
MODEL_NUMERIC_COLUMNS = RAW_NUMERIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS


def _item_count(value: object) -> int:
    """Count distinct comma-separated equipment items."""
    if pd.isna(value):
        return 0
    items = {item.strip() for item in str(value).split(",") if item.strip()}
    return len(items)


class CarFeatureEngineer(BaseEstimator, TransformerMixin):
    """Convert raw listing fields into stable model-ready columns.

    The high-cardinality equipment strings are represented by counts. This keeps
    inference robust when a listing contains an accessory not seen in training.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "CarFeatureEngineer":
        self._validate(X)
        self.feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate(X)
        frame = X.loc[:, FEATURE_COLUMNS].copy()

        for column in LIST_COLUMNS:
            output_name = f"num_{column.lower()}"
            frame[output_name] = frame[column].map(_item_count).astype(float)

        age_denominator = frame["age"].astype(float).clip(lower=0) + 1.0
        frame["km_per_year"] = frame["km"].astype(float) / age_denominator
        weight = frame["Weight_kg"].astype(float).replace(0, np.nan)
        frame["power_to_weight"] = 1000.0 * frame["hp_kW"].astype(float) / weight
        return frame.drop(columns=LIST_COLUMNS)

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        return np.asarray(MODEL_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS, dtype=object)

    @staticmethod
    def _validate(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CarFeatureEngineer expects a pandas DataFrame.")
        missing = sorted(set(FEATURE_COLUMNS) - set(X.columns))
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")


class IQRClipper(BaseEstimator, TransformerMixin):
    """Winsorize numeric values using bounds learned only from training data."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X: np.ndarray, y: object = None) -> "IQRClipper":
        values = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(values, 25, axis=0)
        q3 = np.nanpercentile(values, 75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.factor * iqr
        self.upper_bounds_ = q3 + self.factor * iqr
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        return np.asarray(list(input_features) if input_features is not None else [], dtype=object)


def build_preprocessor(min_category_frequency: int = 20) -> ColumnTransformer:
    """Build the numeric and categorical preprocessing graph."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", IQRClipper()),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_category_frequency,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, MODEL_NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

