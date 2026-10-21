
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


# ==================================================
# 1. Paths and configuration
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw" / "firms"
MODEL_DIR = BASE_DIR / "models"

ARCHIVE_FILE = DATA_DIR / "fire_archive_SV-C2_820222.csv"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF_DATE = pd.Timestamp("2026-04-01")
FRP_PERCENTILE = 0.90


# ==================================================
# 2. Load historical archive
# ==================================================

print("Loading archive dataset...")

df = pd.read_csv(ARCHIVE_FILE)
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

print(f"Archive records: {len(df):,}")
print(
    f"Date range: {df['acq_date'].min().date()} "
    f"to {df['acq_date'].max().date()}"
)


# ==================================================
# 3. Define the target using training data only
# ==================================================

train_mask = df["acq_date"] < CUTOFF_DATE
test_mask = df["acq_date"] >= CUTOFF_DATE

if not train_mask.any() or not test_mask.any():
    raise ValueError("Training or testing period is empty.")

frp_threshold = df.loc[train_mask, "frp"].quantile(
    FRP_PERCENTILE
)

df["high_frp"] = (
    df["frp"] >= frp_threshold
).astype("int8")

print(f"\nHigh-FRP threshold: {frp_threshold:.2f} MW")
print(
    "Training positive rate:",
    round(df.loc[train_mask, "high_frp"].mean(), 4),
)
print(
    "Testing positive rate:",
    round(df.loc[test_mask, "high_frp"].mean(), 4),
)


# ==================================================
# 4. Engineer features
# ==================================================

# Location and seasonal features
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

# Difference between the two satellite brightness measurements
df["brightness_difference"] = (
    df["brightness"] - df["bright_t31"]
)

# Cyclical acquisition time: minutes after midnight
hour = (df["acq_time"] // 100) + (
    (df["acq_time"] % 100) / 60
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

categorical_features = [
    "confidence",
    "daynight",
]

feature_columns = numeric_features + categorical_features

X_train = df.loc[train_mask, feature_columns]
y_train = df.loc[train_mask, "high_frp"]

X_test = df.loc[test_mask, feature_columns]
y_test = df.loc[test_mask, "high_frp"]

print(f"\nTraining records: {len(X_train):,}")
print(f"Testing records: {len(X_test):,}")


# ==================================================
# 5. Build preprocessing pipeline
# ==================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent"),
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        ),
    ),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    (
        "categorical",
        categorical_pipeline,
        categorical_features,
    ),
])


# ==================================================
# 6. Establish the majority-class baseline
# ==================================================

majority_class = int(y_train.mode().iloc[0])

baseline_predictions = np.full(
    len(y_test),
    majority_class,
)

print("\n" + "=" * 55)
print("MAJORITY-CLASS BASELINE")
print("=" * 55)

print(
    "Accuracy:",
    round(accuracy_score(y_test, baseline_predictions), 4),
)
print(
    "Balanced accuracy:",
    round(
        balanced_accuracy_score(y_test, baseline_predictions),
        4,
    ),
)


# ==================================================
# 7. Compare two models
# ==================================================

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=42,
    ),
}

results = []
best_model = None
best_name = None
best_ap = -1.0

for name, classifier in models.items():
    print("\n" + "=" * 55)
    print(f"TRAINING: {name}")
    print("=" * 55)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    balanced_acc = balanced_accuracy_score(
        y_test, predictions
    )
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(
        y_test, probabilities
    )

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"PR-AUC:            {pr_auc:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Other detections",
                "High-FRP detections",
            ],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    results.append({
        "model": name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    })

    # Select using PR-AUC, not raw accuracy.
    if pr_auc > best_ap:
        best_ap = pr_auc
        best_model = pipeline
        best_name = name


# ==================================================
# 8. Print comparison
# ==================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)
print(
    results_df.sort_values(
        "pr_auc",
        ascending=False,
    ).to_string(index=False)
)

print(f"\nBest model by PR-AUC: {best_name}")


# ==================================================
# 9. Save the selected model
# ==================================================

artifact = {
    "model": best_model,
    "feature_columns": feature_columns,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "frp_threshold_mw": float(frp_threshold),
    "target_definition": (
        "FRP at or above the 90th percentile of "
        "training-period archive detections"
    ),
    "cutoff_date": str(CUTOFF_DATE.date()),
    "selected_model": best_name,
    "selection_metric": "PR-AUC",
}

model_path = MODEL_DIR / "high_frp_classifier.joblib"

joblib.dump(artifact, model_path)

print(f"\nSaved selected model to: {model_path}")
print("Training and comparison completed.")



"""
============================================================
MODEL TRAINING RESULTS — TERRAMIND AI
============================================================

Dataset:
- Source: NASA FIRMS Suomi-NPP VIIRS active-fire archive
- Total records: 1,065,317
- Date range: 2025-01-01 to 2026-06-30
- Training records: 883,779
- Testing records: 181,538

Target Definition:
- High-FRP detection: FRP >= 10.81 MW
- Training positive rate: 10.01%
- Testing positive rate: 14.73%

------------------------------------------------------------
1. MAJORITY-CLASS BASELINE
------------------------------------------------------------
Accuracy:          85.27%
Balanced Accuracy: 50.00%

------------------------------------------------------------
2. LOGISTIC REGRESSION
------------------------------------------------------------
Accuracy:          68.05%
Balanced Accuracy: 76.19%
ROC-AUC:            0.8499
PR-AUC:             0.5085

High-FRP Class:
- Precision:        30%
- Recall:           88%
- F1-score:         45%

Confusion Matrix:
[[100075, 54729],
 [  3281, 23453]]

Observation:
High recall, but many non-high-FRP detections were incorrectly
classified as high-FRP. The model also produced a convergence
warning after 1,000 iterations.

------------------------------------------------------------
3. HISTOGRAM-BASED GRADIENT BOOSTING
------------------------------------------------------------
Accuracy:          89.24%
Balanced Accuracy: 72.04%
ROC-AUC:            0.9156
PR-AUC:             0.6645

High-FRP Class:
- Precision:        70%
- Recall:           48%
- F1-score:         57%

Confusion Matrix:
[[149260,  5544],
 [ 13994, 12740]]

Observation:
Achieved better accuracy, ROC-AUC, PR-AUC, and high-FRP precision
than Logistic Regression. However, recall for the high-FRP class
remains limited at the default classification threshold.

------------------------------------------------------------
4. MODEL COMPARISON
------------------------------------------------------------

Selected Model: HistGradientBoosting

Selection Criterion: Highest PR-AUC

Logistic Regression:
- Accuracy:          68.05%
- Balanced Accuracy: 76.19%
- ROC-AUC:            0.8499
- PR-AUC:             0.5085

HistGradientBoosting:
- Accuracy:          89.24%
- Balanced Accuracy: 72.04%
- ROC-AUC:            0.9156
- PR-AUC:             0.6645

------------------------------------------------------------
5. CONCLUSION
------------------------------------------------------------

HistGradientBoosting was selected because it achieved the highest
PR-AUC (0.6645) among the two tested models.

Its accuracy of 89.24% exceeds the majority-class baseline
accuracy of 85.27%.

Further improvements should focus on feature engineering,
classification threshold tuning, and validation on a separate
chronological period.

IMPORTANT:
This model classifies existing satellite detections according
to their FRP values. It does not directly predict future wildfire
occurrence. The reported metrics are specific to the evaluated
dataset and test period.

============================================================
"""
