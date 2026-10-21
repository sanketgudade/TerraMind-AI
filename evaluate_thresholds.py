
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "high_frp_classifier.joblib"
DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "firms"
    / "fire_archive_SV-C2_820222.csv"
)

CUTOFF_DATE = pd.Timestamp("2026-04-01")

print("Loading saved model...")
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]

print("Loading test data...")
df = pd.read_csv(DATA_PATH)
df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")

numeric_columns = [
    "latitude",
    "longitude",
    "brightness",
    "bright_t31",
    "scan",
    "track",
    "acq_time",
    "frp",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(
    subset=[
        "acq_date",
        "latitude",
        "longitude",
        "brightness",
        "bright_t31",
        "scan",
        "track",
        "acq_time",
        "frp",
        "confidence",
        "daynight",
    ]
)

df = df[
    df["latitude"].between(6, 38)
    & df["longitude"].between(65, 100)
].copy()

df = df[df["acq_date"] >= CUTOFF_DATE].copy()

df["month"] = df["acq_date"].dt.month
df["day_of_year"] = df["acq_date"].dt.dayofyear

df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)
df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)
df["day_sin"] = np.sin(
    2 * np.pi * df["day_of_year"] / 365.25
)
df["day_cos"] = np.cos(
    2 * np.pi * df["day_of_year"] / 365.25
)

df["brightness_difference"] = (
    df["brightness"] - df["bright_t31"]
)

hour = (df["acq_time"] // 100) + (
    (df["acq_time"] % 100) / 60
)

df["time_sin"] = np.sin(2 * np.pi * hour / 24)
df["time_cos"] = np.cos(2 * np.pi * hour / 24)

threshold_frp = artifact["frp_threshold_mw"]
y_true = (df["frp"] >= threshold_frp).astype(int)

X_test = df[artifact["feature_columns"]]

print(f"Test records: {len(df):,}")
print(f"FRP threshold: {threshold_frp:.2f} MW")

probabilities = model.predict_proba(X_test)[:, 1]

print(f"ROC-AUC: {roc_auc_score(y_true, probabilities):.4f}")
print(
    "PR-AUC:",
    round(average_precision_score(y_true, probabilities), 4),
)

results = []

for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    predictions = (probabilities >= threshold).astype(int)

    results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
    })

results_df = pd.DataFrame(results)

print("\nThreshold comparison:")
print(results_df.round(4).to_string(index=False))

best = results_df.loc[results_df["f1"].idxmax()]

print("\nBest threshold by F1:")
print(best.to_string())
