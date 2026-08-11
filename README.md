# Used-Car Price Prediction

An end-to-end regression project that estimates the advertised price of a used
car in euros. It turns the original notebook exercise into a reproducible
training and inference pipeline with leakage-safe preprocessing, cross-validated
model selection, saved artifacts, tests, and container support.

## Problem

A reseller needs a consistent first-pass valuation from the vehicle attributes
available in a listing. This is a regression problem, so the primary measure is
mean absolute error (MAE) in euros—not "accuracy." R-squared is reported as a
secondary measure of explained variance.

The prediction is an asking-price estimate, not a guaranteed transaction price.
The data is from one marketplace in 2019 and covers only nine models, so the
model should not be used outside that population without fresh validation.

## Dataset

The included [AutoScout24 dataset](data/README.md) has 15,915 rows and 23 raw
columns. It combines numerical specifications (mileage, age, engine power,
weight and fuel consumption), categorical listing attributes, four comma-separated
equipment fields, and the target `price` in euros.

## EDA

The analysis in [notebooks/EDA.ipynb](notebooks/EDA.ipynb) focuses on the checks
that affect modelling decisions:

- prices are right-skewed, motivating a `log1p` target transform;
- age and mileage have strong negative relationships with price;
- power, model and transmission contain substantial pricing signal;
- equipment fields are lists, not ordinary single-label categories;
- numeric variables contain plausible extremes that should be capped using
  bounds learned from training data only.

The source package is the authoritative implementation; the notebook remains
focused on exploration and modelling decisions.

## Feature engineering and preprocessing

`src/preprocessing.py` performs the same transformations during cross-validation,
final training and inference:

1. Count distinct items in each equipment list. Counts are robust to previously
   unseen accessories and avoid hundreds of sparse, unstable indicators.
2. Add mileage per year and power-to-weight ratio.
3. Median-impute numeric values, cap extremes with training-fold IQR bounds, and
   standardize them.
4. Most-frequent-impute and one-hot encode categoricals. Categories appearing
   fewer than 20 times are grouped by the encoder; unseen categories remain safe.
5. Fit the regressor to `log1p(price)` and automatically convert predictions back
   to euros.

All learned preprocessing lives inside a scikit-learn `Pipeline`. This prevents
information from the test set or validation folds from influencing imputation,
outlier bounds, scaling, encoding, or hyperparameter choice.

## Models and evaluation

The comparison includes Linear Regression, Ridge (L2), Lasso (L1), and Elastic
Net (L1 + L2). Ridge, Lasso and Elastic Net hyperparameters are selected with
five-fold cross-validation on the 80% training partition. The 20% test partition
is evaluated once after selection (`random_state=42`). Before splitting, 1,673
exact duplicate rows are removed. Listings with identical predictor values are
kept in the same train, validation, or test group to prevent duplicate leakage.

| Model | CV MAE | Test MAE | Test RMSE | Test MAPE | Test R2 |
|---|---:|---:|---:|---:|---:|
| Lasso | **EUR 1,587** | EUR 1,594 | EUR 2,500 | 8.35% | 0.890 |
| Elastic Net | EUR 1,587 | EUR 1,594 | EUR 2,501 | 8.35% | 0.890 |
| Ridge | EUR 1,588 | **EUR 1,592** | **EUR 2,497** | **8.34%** | **0.891** |
| Linear Regression | EUR 1,589 | EUR 1,592 | EUR 2,500 | 8.35% | 0.890 |

Exact unrounded results and chosen hyperparameters are generated in
`reports/model_comparison.csv` and `models/metrics.json`. Model selection uses CV
MAE rather than test MAE; Lasso wins by a very small CV margin and sets 17 of 53
encoded coefficients to zero.

## What regularization changed

Regularization helped only marginally: Lasso improves cross-validated MAE by
about EUR 2 over ordinary least squares, while all test MAEs are within EUR 3.
That is not a practically meaningful performance gap.

The reason is informative. After rare-category handling and equipment-count
compression, the feature space is moderate relative to the training sample.
Train and test MAE are also nearly identical, so there is little evidence of
high-variance overfitting for regularization to correct. Ridge stabilizes
correlated effects such as age/mileage and power/weight, while Lasso and Elastic
Net shrink weak one-hot coefficients and provide limited feature selection.
Their main benefit here is coefficient stability and a small complexity penalty,
not a dramatic accuracy gain.

The roughly 0.890 test R2 shows that a linear model is a strong, explainable
baseline. Residual structure and the EUR 2.50k RMSE still suggest nonlinear
interactions and price-segment effects; tree boosting would be a sensible next
experiment, evaluated with the same split and pipeline discipline.

## Project structure

```text
used-car-price-prediction/
|-- data/
|   |-- autoscout_car_sales.csv
|   `-- README.md
|-- notebooks/
|   `-- EDA.ipynb
|-- src/
|   |-- preprocessing.py
|   |-- train.py
|   |-- evaluate.py
|   `-- predict.py
|-- examples/sample_car.json
|-- models/                  # generated model and metrics
|-- reports/                 # generated tables and diagnostics
|-- tests/
|-- requirements.txt
|-- requirements-dev.txt
|-- Dockerfile
`-- README.md
```

## How to run

Create an environment and install dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
```

Train all models and create the reports:

```bash
python -m src.train
```

Re-evaluate the saved model on the untouched test rows:

```bash
python -m src.evaluate
```

Predict one listing from JSON:

```bash
python -m src.predict --input examples/sample_car.json
```

Run the tests:

```bash
python -m pytest -q
```

Or run training in Docker:

```bash
docker build -t used-car-price-prediction .
docker run --rm used-car-price-prediction
```

To keep generated artifacts on the host, mount `/app/models` and `/app/reports`
when running the container.

## Reproducibility and next steps

The split and cross-validation seed are fixed. The saved artifact contains the
entire preprocessing/model graph plus held-out row indices. For production use,
add data-drift monitoring, prediction intervals, schema validation, experiment
tracking, and time-based validation on newer marketplace data.
