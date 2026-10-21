from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.cluster import DBSCAN

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw" / "firms"

ARCHIVE_FILE = DATA_DIR / "fire_archive_SV-C2_820222.csv"
NRT_FILE = DATA_DIR / "fire_nrt_SV-C2_820222.csv"

# DBSCAN configuration
MAX_CLUSTER_POINTS = 20_000
MAX_NOISE_POINTS_RETURNED = 1_000
EARTH_RADIUS_KM = 6371.0088
CLUSTER_RADIUS_KM = 5
MIN_CLUSTER_POINTS = 10


# ============================================================
# TERRAMIND AI — HIGH-FRP CLASSIFIER
# ============================================================

MODEL_PATH = BASE_DIR / "models" / "high_frp_classifier.joblib"

# Decision threshold used by /api/predict.
# 0.15 is provisional pending independent validation.
# Do not describe this as a fully validated threshold until
# it has been confirmed on a dataset that was not used to
# choose it.
DECISION_THRESHOLD = 0.15

try:
    model_artifact = joblib.load(MODEL_PATH)
    fire_model = model_artifact["model"]
    model_features = model_artifact["feature_columns"]
    frp_threshold_mw = float(model_artifact["frp_threshold_mw"])

    print("TerraMind AI model loaded successfully.")
    print("Model:", type(fire_model.named_steps["classifier"]).__name__)
    print("High-FRP threshold:", frp_threshold_mw, "MW")
    print("Decision threshold:", DECISION_THRESHOLD)

except Exception as exc:
    model_artifact = None
    fire_model = None
    model_features = []
    frp_threshold_mw = None
    print("TerraMind AI model loading failed:", exc)


def load_fire_data():
    frames = []

    for file_path, source in [
        (ARCHIVE_FILE, "Archive"),
        (NRT_FILE, "Near-real-time"),
    ]:
        if not file_path.exists():
            continue

        df = pd.read_csv(file_path, low_memory=False)

        required = {
            "latitude",
            "longitude",
            "acq_date",
            "confidence",
            "frp",
            "daynight",
        }

        missing = required.difference(df.columns)

        if missing:
            raise ValueError(
                f"{file_path.name} is missing columns: "
                f"{sorted(missing)}"
            )

        df["acq_date"] = pd.to_datetime(
            df["acq_date"], errors="coerce"
        )

        for column in ["latitude", "longitude", "frp"]:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

        df["source_file"] = source
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No FIRMS CSV files found in {DATA_DIR}"
        )

    data = pd.concat(frames, ignore_index=True)

    data = data.dropna(
        subset=[
            "latitude",
            "longitude",
            "acq_date",
            "frp",
            "confidence",
            "daynight",
        ]
    )

    # Approximate geographic bounds of India.
    data = data[
        data["latitude"].between(6, 38)
        & data["longitude"].between(65, 100)
    ].copy()

    data["confidence"] = (
        data["confidence"].astype(str).str.lower()
    )

    data["daynight"] = (
        data["daynight"].astype(str).str.upper()
    )

    data["date_str"] = data["acq_date"].dt.strftime(
        "%Y-%m-%d"
    )

    return data.reset_index(drop=True)


try:
    FIRE_DATA = load_fire_data()
    DATA_ERROR = None
except Exception as error:
    FIRE_DATA = pd.DataFrame()
    DATA_ERROR = str(error)


def get_filtered_data():
    """Apply common filters to the selected FIRMS records."""

    if DATA_ERROR:
        return pd.DataFrame()

    data = FIRE_DATA

    start = request.args.get("start", "")
    end = request.args.get("end", "")
    confidence = request.args.get(
        "confidence", "all"
    ).lower()
    daynight = request.args.get(
        "daynight", "all"
    ).upper()
    source = request.args.get(
        "source", "all"
    ).lower()

    if start:
        data = data[data["date_str"] >= start]

    if end:
        data = data[data["date_str"] <= end]

    if confidence in {"h", "n", "l"}:
        data = data[data["confidence"] == confidence]

    if daynight in {"D", "N"}:
        data = data[data["daynight"] == daynight]

    if source == "archive":
        data = data[data["source_file"] == "Archive"]
    elif source == "nrt":
        data = data[
            data["source_file"] == "Near-real-time"
        ]

    return data


def date_context():
    """Return date limits for the HTML templates."""

    if DATA_ERROR or FIRE_DATA.empty:
        return {"min_date": "", "max_date": ""}

    return {
        "min_date": FIRE_DATA["date_str"].min(),
        "max_date": FIRE_DATA["date_str"].max(),
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        data_error=DATA_ERROR,
        **date_context(),
    )


@app.route("/clustering")
def clustering_page():
    return render_template(
        "clustering.html",
        data_error=DATA_ERROR,
        **date_context(),
    )


@app.route("/classifier")
def classifier_page():
    return render_template(
        "classifier.html",
        data_error=DATA_ERROR,
        **date_context(),
    )


@app.route("/activity")
def activity_page():
    return render_template(
        "activity.html",
        data_error=DATA_ERROR,
        **date_context(),
    )


@app.route("/api/summary")
def summary():
    if DATA_ERROR:
        return jsonify({"error": DATA_ERROR}), 500

    data = get_filtered_data()

    if data.empty:
        return jsonify({
            "detections": 0,
            "high_confidence": 0,
            "mean_frp": 0,
            "median_frp": 0,
            "daily": [],
            "cells": [],
            "confidence": [],
            "daynight": [],
            "high_frp_detections": 0,
            "frp_threshold_mw": frp_threshold_mw,
        })

    daily = data.groupby("date_str").size().sort_index()

    grid = data.assign(
        grid_lat=data["latitude"].round(1),
        grid_lon=data["longitude"].round(1),
    )

    cells = (
        grid.groupby(["grid_lat", "grid_lon"])
        .agg(
            count=("latitude", "size"),
            mean_frp=("frp", "mean"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
        .head(3000)
    )

    confidence_counts = (
        data["confidence"]
        .value_counts()
        .reindex(["h", "n", "l"], fill_value=0)
    )

    daynight_counts = (
        data["daynight"]
        .value_counts()
        .reindex(["D", "N"], fill_value=0)
    )

    return jsonify({
        "detections": int(len(data)),
        "high_confidence": int(
            (data["confidence"] == "h").sum()
        ),
        "mean_frp": round(float(data["frp"].mean()), 2),
        "median_frp": round(float(data["frp"].median()), 2),

        "daily": [
            {"date": date, "count": int(count)}
            for date, count in daily.items()
        ],

        "cells": [
            {
                "lat": round(float(row.grid_lat), 1),
                "lon": round(float(row.grid_lon), 1),
                "count": int(row.count),
                "mean_frp": round(float(row.mean_frp), 2),
            }
            for row in cells.itertuples(index=False)
        ],

        "confidence": [
            {"label": label, "count": int(count)}
            for label, count in confidence_counts.items()
        ],

        "daynight": [
            {"label": label, "count": int(count)}
            for label, count in daynight_counts.items()
        ],

        # AI Classifier dashboard metrics
        "high_frp_detections": (
            int((data["frp"] >= frp_threshold_mw).sum())
            if frp_threshold_mw is not None
            else 0
        ),
        "frp_threshold_mw": frp_threshold_mw,
    })


@app.route("/api/clusters")
def clusters():
    """Run DBSCAN on a reproducible sample of filtered records."""

    if DATA_ERROR:
        return jsonify({"error": DATA_ERROR}), 500

    data = get_filtered_data()

    if data.empty:
        return jsonify({
            "total_filtered_detections": 0,
            "sampled_detections": 0,
            "cluster_count": 0,
            "noise_points": 0,
            "noise_points_returned": 0,
            "parameters": {
                "radius_km": CLUSTER_RADIUS_KM,
                "min_samples": MIN_CLUSTER_POINTS,
            },
            "clusters": [],
            "noise_detections": [],
            "note": "No detections match the selected filters.",
        })

    total_filtered = len(data)

    # Keep clustering within a manageable memory limit.
    if total_filtered > MAX_CLUSTER_POINTS:
        sample = data.sample(
            n=MAX_CLUSTER_POINTS,
            random_state=42,
        ).copy()
    else:
        sample = data.copy()

    coordinates = np.radians(
        sample[["latitude", "longitude"]].to_numpy()
    )

    model = DBSCAN(
        eps=CLUSTER_RADIUS_KM / EARTH_RADIUS_KM,
        min_samples=MIN_CLUSTER_POINTS,
        metric="haversine",
        algorithm="ball_tree",
        n_jobs=-1,
    )

    labels = model.fit_predict(coordinates)
    sample["cluster_id"] = labels

    noise_mask = labels == -1
    noise_points = int(noise_mask.sum())

    # Return a limited number of noise records to the map.
    noise_sample = sample.loc[
        noise_mask,
        [
            "latitude",
            "longitude",
            "confidence",
            "frp",
            "date_str",
        ],
    ].head(MAX_NOISE_POINTS_RETURNED)

    noise_detections = [
        {
            "latitude": round(float(row.latitude), 5),
            "longitude": round(float(row.longitude), 5),
            "confidence": str(row.confidence),
            "frp": round(float(row.frp), 2),
            "date": str(row.date_str),
        }
        for row in noise_sample.itertuples(index=False)
    ]

    cluster_ids = sorted(
        int(label)
        for label in set(labels)
        if label != -1
    )

    results = []

    for cluster_id in cluster_ids:
        group = sample[sample["cluster_id"] == cluster_id]

        results.append({
            "cluster_id": cluster_id,
            "detections_in_sample": int(len(group)),
            "center_lat": round(
                float(group["latitude"].mean()), 5
            ),
            "center_lon": round(
                float(group["longitude"].mean()), 5
            ),
            "mean_frp": round(
                float(group["frp"].mean()), 2
            ),
            "high_confidence_detections": int(
                (group["confidence"] == "h").sum()
            ),
        })

    results.sort(
        key=lambda item: item["detections_in_sample"],
        reverse=True,
    )

    return jsonify({
        "total_filtered_detections": int(total_filtered),
        "sampled_detections": int(len(sample)),
        "cluster_count": len(results),
        "noise_points": noise_points,
        "noise_points_returned": len(noise_detections),

        "parameters": {
            "radius_km": CLUSTER_RADIUS_KM,
            "min_samples": MIN_CLUSTER_POINTS,
        },

        "clusters": results,
        "noise_detections": noise_detections,

        "note": (
            "Cluster sizes and noise counts describe the analysed "
            "sample, not necessarily the complete dataset. Up to "
            f"{MAX_NOISE_POINTS_RETURNED} noise coordinates are "
            "returned for map display. Clusters indicate spatially "
            "dense detections, not confirmed individual wildfires "
            "or future fire risk."
        ),
    })


@app.route("/api/activity")
def wildfire_activity():
    """Rank geographic cells by relative historical fire activity."""

    if DATA_ERROR:
        return jsonify({"error": DATA_ERROR}), 500

    data = get_filtered_data()

    if data.empty:
        return jsonify({
            "total_detections": 0,
            "cells_returned": 0,
            "activity_cells": [],
            "score_definition": (
                "50% detection frequency, "
                "30% mean FRP, and 20% high-confidence percentage."
            ),
            "note": "No detections match the selected filters."
        })

    data = data.copy()

    # Create geographic cells approximately 0.25 degrees apart.
    data["grid_lat"] = np.floor(data["latitude"] * 4) / 4
    data["grid_lon"] = np.floor(data["longitude"] * 4) / 4

    # Summarize detections within each cell.
    grouped = (
        data.groupby(["grid_lat", "grid_lon"])
        .agg(
            detections=("latitude", "size"),
            mean_frp=("frp", "mean"),
            max_frp=("frp", "max"),
            high_confidence=(
                "confidence",
                lambda values: (values == "h").sum()
            ),
            latest_detection=("date_str", "max"),
        )
        .reset_index()
    )

    # Convert high-confidence detections to a percentage.
    grouped["confidence_pct"] = (
        grouped["high_confidence"]
        / grouped["detections"].clip(lower=1)
        * 100
    )

    # Use percentile-based scaling to reduce the influence
    # of unusually large values.
    def percentile_score(series):
        upper = float(series.quantile(0.95))

        if upper <= 0:
            return pd.Series(0.0, index=series.index)

        return (
            series.clip(lower=0, upper=upper) / upper * 100
        )

    grouped["frequency_score"] = percentile_score(
        grouped["detections"]
    )

    grouped["frp_score"] = percentile_score(
        grouped["mean_frp"]
    )

    grouped["confidence_score"] = (
        grouped["confidence_pct"].clip(0, 100)
    )

    # Relative historical activity score, not a fire probability.
    grouped["activity_score"] = (
        0.50 * grouped["frequency_score"]
        + 0.30 * grouped["frp_score"]
        + 0.20 * grouped["confidence_score"]
    ).clip(0, 100)

    def activity_level(score):
        if score >= 70:
            return "High"
        elif score >= 35:
            return "Moderate"
        return "Low"

    grouped["activity_level"] = (
        grouped["activity_score"].apply(activity_level)
    )

    # Sort the highest-scoring cells first.
    grouped = grouped.sort_values(
        ["activity_score", "detections"],
        ascending=[False, False]
    ).head(1000)

    activity_cells = []

    for row in grouped.itertuples(index=False):
        activity_cells.append({
            "lat": round(float(row.grid_lat), 4),
            "lon": round(float(row.grid_lon), 4),
            "detections": int(row.detections),
            "mean_frp": round(float(row.mean_frp), 2),
            "max_frp": round(float(row.max_frp), 2),
            "high_confidence": int(row.high_confidence),
            "confidence_pct": round(float(row.confidence_pct), 2),
            "frequency_score": round(float(row.frequency_score), 2),
            "frp_score": round(float(row.frp_score), 2),
            "confidence_score": round(float(row.confidence_score), 2),
            "latest_detection": str(row.latest_detection),
            "activity_score": round(float(row.activity_score), 2),
            "activity_level": str(row.activity_level),
        })

    return jsonify({
        "total_detections": int(len(data)),
        "cells_returned": len(activity_cells),
        "high_activity_cells": int(
            (grouped["activity_level"] == "High").sum()
        ),
        "score_definition": (
            "50% percentile-scaled detection frequency, "
            "30% percentile-scaled mean FRP, and "
            "20% high-confidence detection percentage."
        ),
        "note": (
            "This is a relative historical activity score, not a "
            "validated wildfire probability or future-risk prediction. "
            "Scores depend on the selected filters and observation period. "
            "Only the top 1,000 cells are returned."
        ),
        "activity_cells": activity_cells,
    })


@app.route("/api/hotspots")
def wildfire_hotspots():
    """
    Return ranked wildfire detection hotspots across India.

    Score components:
    - 60%: Log-scaled detection frequency
    - 25%: Mean FRP, capped at the 95th percentile
    - 15%: Percentage of high-confidence detections

    Important:
    This is a relative hotspot activity score, NOT a probability
    of a future wildfire.
    """

    try:
        # 1. Load data using the existing dashboard filters
        data = get_filtered_data()

        if data.empty:
            return jsonify({
                "total_detections": 0,
                "hotspots": [],
                "score_definition": (
                    "Relative activity score based on detection "
                    "frequency, FRP, and confidence."
                ),
                "message": "No detections found for the selected filters."
            })

        # 2. Ensure required columns are numeric
        data = data.copy()

        data["latitude"] = pd.to_numeric(
            data["latitude"], errors="coerce"
        )

        data["longitude"] = pd.to_numeric(
            data["longitude"], errors="coerce"
        )

        data["frp"] = pd.to_numeric(
            data["frp"], errors="coerce"
        ).fillna(0).clip(lower=0)

        data = data.dropna(subset=["latitude", "longitude"])

        if data.empty:
            return jsonify({
                "total_detections": 0,
                "hotspots": [],
                "message": "No valid geographic detections found."
            })

        # 3. Create geographic grid cells of 0.25 x 0.25 degrees
        # Nearby detections fall into the same cell.
        data["lat_cell"] = (
            np.floor(data["latitude"] * 4) / 4
        ).round(2)

        data["lon_cell"] = (
            np.floor(data["longitude"] * 4) / 4
        ).round(2)

        # 4. Identify high-confidence detections
        # NASA FIRMS confidence can be numeric or categorical.
        confidence = (
            data["confidence"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        numeric_confidence = pd.to_numeric(
            confidence, errors="coerce"
        )

        high_confidence = (
            numeric_confidence.ge(80)
            | confidence.isin(["high", "h"])
        )

        data["is_high_confidence"] = high_confidence.astype(int)

        # 5. Aggregate detections in each geographic cell
        hotspots = (
            data.groupby(["lat_cell", "lon_cell"])
            .agg(
                detections=("frp", "size"),
                mean_frp=("frp", "mean"),
                max_frp=("frp", "max"),
                total_frp=("frp", "sum"),
                high_confidence=("is_high_confidence", "sum")
            )
            .reset_index()
        )

        hotspots = hotspots.rename(columns={
            "lat_cell": "latitude",
            "lon_cell": "longitude"
        })

        # 6. Ignore cells with fewer than five detections
        # This reduces the effect of isolated detections.
        hotspots = hotspots[
            hotspots["detections"] >= 5
        ].copy()

        if hotspots.empty:
            return jsonify({
                "total_detections": int(len(data)),
                "hotspots": [],
                "score_definition": (
                    "Cells require at least 5 detections."
                ),
                "message": (
                    "No geographic cells have at least "
                    "5 detections for the selected filters."
                )
            })

        # 7. Calculate frequency score using logarithmic scaling
        # This prevents the largest cell from dominating excessively.
        max_detections = max(
            int(hotspots["detections"].max()), 1
        )

        hotspots["frequency_score"] = (
            np.log1p(hotspots["detections"])
            / np.log1p(max_detections)
            * 100
        )

        # 8. Cap mean FRP at its 95th percentile
        # This limits the influence of extreme FRP outliers.
        frp_cap = max(
            float(hotspots["mean_frp"].quantile(0.95)),
            1.0
        )

        capped_frp = hotspots["mean_frp"].clip(
            lower=0,
            upper=frp_cap
        )

        hotspots["frp_score"] = (
            capped_frp / frp_cap * 100
        )

        # 9. Calculate high-confidence percentage
        hotspots["confidence_score"] = (
            hotspots["high_confidence"]
            / hotspots["detections"].clip(lower=1)
            * 100
        )

        # 10. Calculate final relative hotspot activity score
        hotspots["hotspot_score"] = (
            0.60 * hotspots["frequency_score"]
            + 0.25 * hotspots["frp_score"]
            + 0.15 * hotspots["confidence_score"]
        ).clip(0, 100)

        # 11. Assign descriptive activity levels
        def get_activity_level(score):
            if score >= 70:
                return "Very High"
            elif score >= 50:
                return "High"
            elif score >= 30:
                return "Moderate"
            else:
                return "Low"

        hotspots["activity_level"] = (
            hotspots["hotspot_score"]
            .apply(get_activity_level)
        )

        # 12. Rank hotspots and keep the top 100
        hotspots = hotspots.sort_values(
            by=["hotspot_score", "detections"],
            ascending=[False, False]
        ).head(100)

        # 13. Round numerical values for readable JSON
        numeric_columns = [
            "latitude",
            "longitude",
            "mean_frp",
            "max_frp",
            "total_frp",
            "frequency_score",
            "frp_score",
            "confidence_score",
            "hotspot_score"
        ]

        for column in numeric_columns:
            hotspots[column] = hotspots[column].round(2)

        # 14. Convert results into JSON-safe records
        hotspot_records = []

        for rank, (_, row) in enumerate(
            hotspots.iterrows(), start=1
        ):
            hotspot_records.append({
                "rank": rank,
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "detections": int(row["detections"]),
                "mean_frp": float(row["mean_frp"]),
                "max_frp": float(row["max_frp"]),
                "total_frp": float(row["total_frp"]),
                "high_confidence": int(row["high_confidence"]),
                "confidence_score": float(row["confidence_score"]),
                "frequency_score": float(row["frequency_score"]),
                "frp_score": float(row["frp_score"]),
                "hotspot_score": float(row["hotspot_score"]),
                "activity_level": str(row["activity_level"])
            })

        # 15. Return complete API response
        return jsonify({
            "total_detections": int(len(data)),
            "cells_meeting_minimum": int(
                (data.groupby(["lat_cell", "lon_cell"])
                 .size() >= 5).sum()
            ),
            "returned_hotspots": len(hotspot_records),
            "grid_size_degrees": 0.25,
            "minimum_detections_per_cell": 5,
            "frp_cap_95th_percentile": round(frp_cap, 2),
            "score_definition": (
                "60% log-scaled detection frequency, "
                "25% mean FRP capped at the 95th percentile, "
                "15% high-confidence detection percentage. "
                "Cells require at least 5 detections."
            ),
            "score_note": (
                "The score ranks relative historical or filtered "
                "satellite detection activity. It is not a calibrated "
                "wildfire probability or a forecast of future fires."
            ),
            "hotspots": hotspot_records
        })

    except Exception as e:
        app.logger.exception("Failed to calculate wildfire hotspots")

        return jsonify({
            "error": "Failed to calculate wildfire hotspots.",
            "details": str(e),
            "hotspots": []
        }), 500


@app.route("/api/predict", methods=["POST"])
def predict_fire_intensity():
    if fire_model is None:
        return jsonify({
            "success": False,
            "error": "The fire prediction model is not available."
        }), 503

    try:
        payload = request.get_json(silent=True)

        if not isinstance(payload, dict):
            return jsonify({
                "success": False,
                "error": "Please provide valid JSON input."
            }), 400

        required_fields = [
            "latitude",
            "longitude",
            "brightness",
            "bright_t31",
            "scan",
            "track",
            "confidence",
            "daynight",
            "acq_date",
            "acq_time",
        ]

        missing_fields = [
            field for field in required_fields
            if field not in payload
        ]

        if missing_fields:
            return jsonify({
                "success": False,
                "error": (
                    "Missing required fields: "
                    + ", ".join(missing_fields)
                )
            }), 400

        # Parse and validate numeric fields.
        latitude = float(payload["latitude"])
        longitude = float(payload["longitude"])
        brightness = float(payload["brightness"])
        bright_t31 = float(payload["bright_t31"])
        scan = float(payload["scan"])
        track = float(payload["track"])
        acq_time = float(payload["acq_time"])

        if not (6 <= latitude <= 38):
            return jsonify({
                "success": False,
                "error": "Latitude must be between 6 and 38."
            }), 400

        if not (65 <= longitude <= 100):
            return jsonify({
                "success": False,
                "error": "Longitude must be between 65 and 100."
            }), 400

        # Parse the observation date.
        acq_date_raw = str(payload["acq_date"])
        acq_date = pd.to_datetime(acq_date_raw, errors="coerce")

        if pd.isna(acq_date):
            return jsonify({
                "success": False,
                "error": "Observation date could not be parsed."
            }), 400

        # Normalize categorical fields the same way as the dataset.
        confidence = str(payload["confidence"]).strip().lower()
        daynight = str(payload["daynight"]).strip().upper()

        # Build the engineered features that the model expects.
        month = int(acq_date.month)
        day_of_year = int(acq_date.dayofyear)

        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        day_sin = np.sin(2 * np.pi * day_of_year / 365.25)
        day_cos = np.cos(2 * np.pi * day_of_year / 365.25)

        brightness_difference = brightness - bright_t31

        # Validate acquisition time in HHMM format.
        time_value = int(acq_time)
        hours = time_value // 100
        minutes = time_value % 100

        if hours > 23 or minutes > 59:
            return jsonify({
                "success": False,
                "error": (
                    "Acquisition time must be a valid HHMM time."
                )
            }), 400

        hour = hours + minutes / 60.0
        time_sin = np.sin(2 * np.pi * hour / 24)
        time_cos = np.cos(2 * np.pi * hour / 24)

        feature_row = {
            "latitude": latitude,
            "longitude": longitude,
            "brightness": brightness,
            "bright_t31": bright_t31,
            "scan": scan,
            "track": track,
            "confidence": confidence,
            "daynight": daynight,
            "acq_time": acq_time,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "day_sin": day_sin,
            "day_cos": day_cos,
            "brightness_difference": brightness_difference,
            "time_sin": time_sin,
            "time_cos": time_cos,
        }

        # Only keep the features the saved model actually uses.
        feature_frame = pd.DataFrame([feature_row])

        missing_features = [
            name for name in model_features
            if name not in feature_frame.columns
        ]

        if missing_features:
            return jsonify({
                "success": False,
                "error": (
                    "Model expects features that could not be built: "
                    + ", ".join(missing_features)
                )
            }), 400

        feature_frame = feature_frame[model_features]

        # Predict.
        probabilities = fire_model.predict_proba(feature_frame)[0]
        high_frp_probability = float(probabilities[1])

        prediction = int(
            high_frp_probability >= DECISION_THRESHOLD
        )

        return jsonify({
            "success": True,
            "prediction": prediction,
            "probability": high_frp_probability,
            "high_frp_probability": high_frp_probability,
            "threshold_used": DECISION_THRESHOLD,
            "frp_threshold_mw": frp_threshold_mw,
            "label": (
                "High FRP" if prediction == 1 else "Other Detection"
            ),
            "note": (
                "This classifier estimates whether an existing satellite "
                "detection belongs to the high-FRP category. It is not a "
                "forecast of future wildfire risk."
            ),
        })

    except (TypeError, ValueError) as error:
        return jsonify({
            "success": False,
            "error": f"Invalid input values: {error}"
        }), 400

    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {error}"
        }), 500


if __name__ == "__main__":
    print("Starting TerraMind AI...")

    if DATA_ERROR:
        print("DATA ERROR:", DATA_ERROR)
    else:
        print(f"Loaded {len(FIRE_DATA):,} detections")
        print(
            "Date range:",
            FIRE_DATA["date_str"].min(),
            "to",
            FIRE_DATA["date_str"].max(),
        )

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000,
    )