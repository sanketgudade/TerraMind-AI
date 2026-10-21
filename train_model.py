
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ==================================================
# 1. Configuration
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw" / "firms"
MODEL_DIR = BASE_DIR / "models"

DATA_FILE = DATA_DIR / "fire_archive_SV-C2_820222.csv"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = pd.Timestamp("2026-01-01")
VALIDATION_END = pd.Timestamp("2026-04-01")

FRP_PERCENTILE = 0.90
RANDOM_STATE = 42


# ==================================================
# 2. Load and clean data
# ==================================================

print("Loading archive...")

df = pd.read_csv(DATA_FILE)
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

df = df.sort_values("acq_date").reset_index(drop=True)

print(f"India detections: {len(df):,}")
print(
    f"Dates: {df['acq_date'].min().date()} "
    f"to {df['acq_date'].max().date()}"
)


# ==================================================
# 3. Define chronological splits and target
# ==================================================

train_mask = df["acq_date"] < TRAIN_END

validation_mask = (
    (df["acq_date"] >= TRAIN_END)
    & (df["acq_date"] < VALIDATION_END)
)

test_mask = df["acq_date"] >= VALIDATION_END

if not train_mask.any() or not validation_mask.any() or not test_mask.any():
    raise ValueError("One chronological data split is empty.")

frp_threshold = df.loc[train_mask, "frp"].quantile(
    FRP_PERCENTILE
)

df["high_frp"] = (
    df["frp"] >= frp_threshold
).astype("int8")

print(f"\nHigh-FRP label threshold: {frp_threshold:.2f} MW")

for name, mask in [
    ("Train", train_mask),
    ("Validation", validation_mask),
    ("Final test", test_mask),
]:
    print(
        f"{name:12s}: {mask.sum():>9,} rows | "
        f"positive rate = {df.loc[mask, 'high_frp'].mean():.2%}"
    )


# ==================================================
# 4. Feature engineering
# ==================================================

df["brightness_difference"] = (
    df["brightness"] - df["bright_t31"]
)

df["month"] = df["acq_date"].dt.month
df["day_of_year"] = df["acq_date"].dt.dayofyear

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

df["day_sin"] = np.sin(
    2 * np.pi * df["day_of_year"] / 365.25
)
df["day_cos"] = np.cos(
    2 * np.pi * df["day_of_year"] / 365.25
)

hour = (
    (df["acq_time"] // 100)
    + (df["acq_time"] % 100) / 60
)

df["time_sin"] = np.sin(2 * np.pi * hour / 24)
df["time_cos"] = np.cos(2 * np.pi * hour / 24)

numeric_features = [
    "latitude",
    "longitude",
    "brightness",
    "bright_t31",
    "brightness_difference",
    "scan",
    "track",
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
    "time_sin",
    "time_cos",
]

categorical_features = ["confidence", "daynight"]
feature_columns = numeric_features + categorical_features

# Normalize categorical missing-value handling.
for column in categorical_features:
    df[column] = df[column].astype(str)

X_train = df.loc[train_mask, feature_columns].copy()
y_train = df.loc[train_mask, "high_frp"].copy()

X_val = df.loc[validation_mask, feature_columns].copy()
y_val = df.loc[validation_mask, "high_frp"].copy()

X_test = df.loc[test_mask, feature_columns].copy()
y_test = df.loc[test_mask, "high_frp"].copy()

# HistGradientBoosting requires numeric encoded features.
# One-hot encoding is learned from training data only.
X_train_hist = pd.get_dummies(
    X_train, columns=categorical_features, dtype=float
)
hist_columns = X_train_hist.columns.tolist()

X_val_hist = pd.get_dummies(
    X_val, columns=categorical_features, dtype=float
).reindex(columns=hist_columns, fill_value=0)

X_test_hist = pd.get_dummies(
    X_test, columns=categorical_features, dtype=float
).reindex(columns=hist_columns, fill_value=0)

print("\nFeature matrices prepared.")


# ==================================================
# 5. Model definitions
# ==================================================

models = {
    "CatBoost": CatBoostClassifier(
        iterations=700,
        depth=8,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="PRAUC",
        verbose=100,
        random_seed=RANDOM_STATE,
        allow_writing_files=False,
        thread_count=4,
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=2.0,
        early_stopping=True,
        random_state=RANDOM_STATE,
    ),
}


# ==================================================
# 6. Train and select by validation PR-AUC
# ==================================================

results = []
trained_models = {}
best_name = None
best_ap = -1.0
best_model = None
best_kind = None

for name, model in models.items():
    print("\n" + "=" * 60)
    print("TRAINING:", name)
    print("=" * 60)

    if name == "CatBoost":
        model.fit(
            X_train,
            y_train,
            cat_features=categorical_features,
            eval_set=(X_val, y_val),
            early_stopping_rounds=80,
            verbose=100,
        )
        val_prob = model.predict_proba(X_val)[:, 1]
        kind = "catboost"
    else:
        model.fit(X_train_hist, y_train)
        val_prob = model.predict_proba(X_val_hist)[:, 1]
        kind = "hist"

    val_ap = average_precision_score(y_val, val_prob)
    val_roc = roc_auc_score(y_val, val_prob)

    # Select threshold using validation data only.
    threshold_rows = []

    for threshold in np.arange(0.10, 0.901, 0.01):
        val_pred = (val_prob >= threshold).astype(int)

        threshold_rows.append({
            "threshold": float(threshold),
            "f1": f1_score(y_val, val_pred, zero_division=0),
            "precision": precision_score(
                y_val, val_pred, zero_division=0
            ),
            "recall": recall_score(
                y_val, val_pred, zero_division=0
            ),
            "accuracy": accuracy_score(y_val, val_pred),
            "balanced_accuracy": balanced_accuracy_score(
                y_val, val_pred
            ),
        })

    threshold_df = pd.DataFrame(threshold_rows)
    best_threshold_row = threshold_df.loc[
        threshold_df["f1"].idxmax()
    ]

    print(f"Validation PR-AUC: {val_ap:.4f}")
    print(f"Validation ROC-AUC: {val_roc:.4f}")
    print(
        "Best validation threshold:",
        f"{best_threshold_row['threshold']:.2f}",
    )
    print(
        "Validation F1:",
        f"{best_threshold_row['f1']:.4f}",
    )
    print(
        "Validation precision:",
        f"{best_threshold_row['precision']:.4f}",
    )
    print(
        "Validation recall:",
        f"{best_threshold_row['recall']:.4f}",
    )

    results.append({
        "model": name,
        "validation_pr_auc": val_ap,
        "validation_roc_auc": val_roc,
        "validation_f1": best_threshold_row["f1"],
        "threshold": best_threshold_row["threshold"],
    })

    trained_models[name] = {
        "model": model,
        "kind": kind,
        "threshold": float(best_threshold_row["threshold"]),
        "validation_pr_auc": val_ap,
    }

    # Choose model by validation PR-AUC, not final test scores.
    if val_ap > best_ap:
        best_ap = val_ap
        best_name = name
        best_model = model
        best_kind = kind
        best_threshold = float(best_threshold_row["threshold"])


# ==================================================
# 7. Final test evaluation - one time
# ==================================================

print("\n" + "=" * 60)
print("VALIDATION MODEL COMPARISON")
print("=" * 60)
print(
    pd.DataFrame(results)
    .sort_values("validation_pr_auc", ascending=False)
    .to_string(index=False)
)

print("\nSelected model:", best_name)
print(f"Selected threshold: {best_threshold:.2f}")

if best_kind == "catboost":
    test_prob = best_model.predict_proba(X_test)[:, 1]
else:
    test_prob = best_model.predict_proba(X_test_hist)[:, 1]

test_pred = (test_prob >= best_threshold).astype(int)

print("\n" + "=" * 60)
print("FINAL CHRONOLOGICAL TEST RESULTS")
print("=" * 60)

print(f"Accuracy:          {accuracy_score(y_test, test_pred):.4f}")
print(
    "Balanced accuracy:",
    f"{balanced_accuracy_score(y_test, test_pred):.4f}",
)
print(f"Precision:         {precision_score(y_test, test_pred, zero_division=0):.4f}")
print(f"Recall:            {recall_score(y_test, test_pred, zero_division=0):.4f}")
print(f"F1-score:          {f1_score(y_test, test_pred, zero_division=0):.4f}")
print(f"ROC-AUC:           {roc_auc_score(y_test, test_prob):.4f}")
print(f"PR-AUC:            {average_precision_score(y_test, test_prob):.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_pred,
        target_names=["Other detections", "High-FRP detections"],
        zero_division=0,
    )
)

print("Confusion matrix:")
print(confusion_matrix(y_test, test_pred))


# ==================================================
# 8. Save model and metadata
# ==================================================

artifact = {
    "model": best_model,
    "model_type": best_kind,
    "feature_columns": feature_columns,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "hist_columns": hist_columns,
    "threshold": best_threshold,
    "frp_threshold_mw": float(frp_threshold),
    "target_definition": (
        "FRP >= 90th percentile calculated from training-period detections"
    ),
    "train_end": str(TRAIN_END.date()),
    "validation_end": str(VALIDATION_END.date()),
    "selected_model": best_name,
    "selection_metric": "Validation PR-AUC",
}

model_path = MODEL_DIR / "high_frp_classifier.joblib"
joblib.dump(artifact, model_path)

print(f"\nSaved model to: {model_path}")
print("Done.")


