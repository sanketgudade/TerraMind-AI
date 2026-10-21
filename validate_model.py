
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "high_frp_classifier.joblib"
DATA_PATH = BASE_DIR / "data" / "raw" / "firms" / "fire_archive_SV-C2_820222.csv"

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
features = artifact["feature_columns"]
frp_threshold = float(artifact["frp_threshold_mw"])

print("Model:", type(model.named_steps["classifier"]).__name__)
print("FRP threshold:", frp_threshold, "MW")

# Read a manageable sample from the real archive.
df = pd.read_csv(DATA_PATH)

required = [
    "latitude", "longitude", "brightness", "bright_t31",
    "scan", "track", "confidence", "daynight",
    "acq_date", "acq_time", "frp",
]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing dataset columns: {missing}")

df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
df["frp"] = pd.to_numeric(df["frp"], errors="coerce")

# Use India observations and keep only records with required values.
df = df[
    df["latitude"].between(6, 38)
    & df["longitude"].between(65, 100)
].copy()

df = df.dropna(subset=required).copy()

# Use a reproducible sample from the actual archive.
sample = df.sample(n=min(5000, len(df)), random_state=42).copy()

sample["brightness"] = pd.to_numeric(sample["brightness"], errors="coerce")
sample["bright_t31"] = pd.to_numeric(sample["bright_t31"], errors="coerce")
sample["scan"] = pd.to_numeric(sample["scan"], errors="coerce")
sample["track"] = pd.to_numeric(sample["track"], errors="coerce")
sample["acq_time"] = pd.to_numeric(sample["acq_time"], errors="coerce")

sample = sample.dropna(
    subset=["brightness", "bright_t31", "scan", "track", "acq_time"]
).copy()

dates = sample["acq_date"]
day_of_year = dates.dt.dayofyear
month = dates.dt.month
hours = (sample["acq_time"] // 100).astype(int)
minutes = (sample["acq_time"] % 100).astype(int)

valid_time = (
    hours.between(0, 23)
    & minutes.between(0, 59)
)
sample = sample[valid_time].copy()

dates = sample["acq_date"]
day_of_year = dates.dt.dayofyear
month = dates.dt.month
hours = (sample["acq_time"] // 100).astype(int)
minutes = (sample["acq_time"] % 100).astype(int)
decimal_hour = hours + minutes / 60.0

X = pd.DataFrame(index=sample.index)
X["latitude"] = sample["latitude"]
X["longitude"] = sample["longitude"]
X["brightness"] = sample["brightness"]
X["bright_t31"] = sample["bright_t31"]
X["brightness_difference"] = (
    sample["brightness"] - sample["bright_t31"]
)
X["scan"] = sample["scan"]
X["track"] = sample["track"]
X["month_sin"] = np.sin(2 * np.pi * month / 12)
X["month_cos"] = np.cos(2 * np.pi * month / 12)
X["day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
X["day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
X["time_sin"] = np.sin(2 * np.pi * decimal_hour / 24)
X["time_cos"] = np.cos(2 * np.pi * decimal_hour / 24)
X["confidence"] = sample["confidence"].astype(str).str.lower()
X["daynight"] = sample["daynight"].astype(str).str.upper()

X = X[features]
y_true = (sample["frp"] >= frp_threshold).astype(int)

# These are in-sample-era archive observations, not an independent test set.
probabilities = model.predict_proba(X)[:, 1]
y_pred = (probabilities >= 0.5).astype(int)

print("\nRecords evaluated:", len(sample))
print("Actual high-FRP rate:", round(y_true.mean() * 100, 2), "%")
print("Predicted high-FRP rate:", round(y_pred.mean() * 100, 2), "%")
print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["Other detections", "High-FRP detections"],
    zero_division=0,
))

if y_true.nunique() == 2:
    print("ROC-AUC:", round(roc_auc_score(y_true, probabilities), 4))
    print("PR-AUC:", round(average_precision_score(y_true, probabilities), 4))
else:
    print("ROC-AUC and PR-AUC unavailable: sample contains only one class.")
