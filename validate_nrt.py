from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "raw" / "firms" / "fire_nrt_SV-C2_820222.csv"
MODEL_PATH = BASE_DIR / "models" / "high_frp_classifier.joblib"

# ------------------------------------------------------------------
# Load the saved model and metadata
# ------------------------------------------------------------------
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
features = artifact["feature_columns"]
frp_threshold = float(artifact["frp_threshold_mw"])

# ------------------------------------------------------------------
# Load the later-period satellite detections (NRT file)
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# ------------------------------------------------------------------
# Feature engineering — must mirror the training definitions exactly
# ------------------------------------------------------------------
df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")

df["acq_time"] = (
    pd.to_numeric(df["acq_time"], errors="coerce")
    .fillna(0)
    .astype(int)
    .astype(str)
    .str.zfill(4)
)

df["brightness_difference"] = df["brightness"] - df["bright_t31"]

df["month"] = df["acq_date"].dt.month
df["day_of_year"] = df["acq_date"].dt.dayofyear

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

# Convert HHMM to minutes after midnight
time_minutes = (
    (df["acq_time"].str[:2].astype(int) * 60)
    + df["acq_time"].str[2:].astype(int)
)

df["time_sin"] = np.sin(2 * np.pi * time_minutes / 1440)
df["time_cos"] = np.cos(2 * np.pi * time_minutes / 1440)

# ------------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------------
missing = [column for column in features if column not in df.columns]

if missing:
    raise ValueError(f"Missing model features: {missing}")

df["frp"] = pd.to_numeric(df["frp"], errors="coerce")
df = df.dropna(subset=["acq_date", "frp"]).copy()

if df.empty:
    raise ValueError("No valid NRT records remain after cleaning.")

# ------------------------------------------------------------------
# Ground-truth label and prediction
# ------------------------------------------------------------------
y_true = (df["frp"] >= frp_threshold).astype(int)

X = df[features]
y_probability = model.predict_proba(X)[:, 1]

# Default decision threshold
y_pred = (y_probability >= 0.5).astype(int)

# ------------------------------------------------------------------
# Threshold comparison table
# ------------------------------------------------------------------
print("\n===== PROBABILITY THRESHOLD COMPARISON =====")
print(f"{'Threshold':>10} {'Precision':>11} {'Recall':>9} {'F1':>9} {'Alerts':>9}")

for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    predictions = (y_probability >= threshold).astype(int)

    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    alerts = int(predictions.sum())

    print(
        f"{threshold:>10.2f} "
        f"{precision:>11.3f} "
        f"{recall:>9.3f} "
        f"{f1:>9.3f} "
        f"{alerts:>9}"
    )

# ------------------------------------------------------------------
# Main evaluation at 0.50
# ------------------------------------------------------------------
print("\n===== TERRAMIND AI: LATER-PERIOD VALIDATION =====")
print("Model:", artifact.get("selected_model", type(model).__name__))
print("Dataset:", DATA_PATH.name)
print(
    "Date range:",
    df["acq_date"].min().date(),
    "to",
    df["acq_date"].max().date(),
)
print("Records evaluated:", len(df))
print(f"High-FRP definition: FRP >= {frp_threshold:.2f} MW")
print(f"Actual high-FRP rate: {y_true.mean():.2%}")
print(f"Predicted high-FRP rate: {y_pred.mean():.2%}")

print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Other detections", "High-FRP detections"],
        zero_division=0,
    )
)

if len(np.unique(y_true)) == 2:
    print(f"ROC-AUC: {roc_auc_score(y_true, y_probability):.4f}")
    print(f"PR-AUC: {average_precision_score(y_true, y_probability):.4f}")
else:
    print(
        "ROC-AUC and PR-AUC cannot be calculated: "
        "only one class is present."
    )

print("\nValidation completed. The model was not retrained.")
print(
    "This evaluates high-FRP detection classification, "
    "not future wildfire risk."
)