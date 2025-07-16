import datetime
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import numpy as np
from autogluon.timeseries.predictor import TimeSeriesPredictor
from pymongo.database import Database
from scipy.stats import ks_2samp
from src import dbConnect

def log_model_run(
    predictor: TimeSeriesPredictor,
    collection_name: str,
    performance_metrics: dict,
    data_snapshot_info: dict | None = None,
    trigger_source: str = "auto",
):
    load_dotenv()
    client = dbConnect.client
    db = dbConnect.db

    try:
        feature_importances = predictor.feature_importance().to_dict()
    except Exception:
        feature_importances = {}

    run_doc = {
        "collection_name": collection_name,
        "trained_date": datetime.datetime.now().isoformat(),
        "performance_metrics": performance_metrics,
        "trigger_source": trigger_source,
        "data_snapshot_info": data_snapshot_info or {},
        "feature_importances": feature_importances,
    }

    db.model_runs.insert_one(run_doc)
    print(f"Logged model run metadata in 'model_runs' collection.")


def check_for_drift(
    previous_data: pd.DataFrame,
    new_data: pd.DataFrame,
    mean_threshold: float = 0.1,
    p_value_threshold: float = 0.05
) -> bool:
    if previous_data.empty or new_data.empty:
        return False

    common_cols = previous_data.columns.intersection(new_data.columns)
    if len(common_cols) == 0:
        return False

    mean_diffs = []
    p_values = []

    for col in common_cols:
        if pd.api.types.is_numeric_dtype(previous_data[col]):
            old_mean = previous_data[col].mean()
            new_mean = new_data[col].mean()
            if old_mean != 0:
                diff_ratio = abs((new_mean - old_mean) / old_mean)
                mean_diffs.append(diff_ratio)
            stat, p_value = ks_2samp(
                previous_data[col].dropna(),
                new_data[col].dropna()
            )
            p_values.append(p_value)

    if not mean_diffs or not p_values:
        return False

    avg_diff = np.mean(mean_diffs)
    min_p_value = np.min(p_values)

    drift_detected = (avg_diff > mean_threshold) or (min_p_value < p_value_threshold)
    return bool(drift_detected)


def rollback_to_previous_version(
    limit_version_count: int = 1
) -> str:
    load_dotenv()
    client = dbConnect.client
    db = dbConnect.db

    versioned = [coll for coll in db.list_collection_names() if coll.startswith("inventory_recommendations_")]
    versioned.sort()

    if len(versioned) < 1:
        print("No rollback possible because only one or zero versioned collections exist.")
        return ""

    newest = versioned[-1]
    second_newest = versioned[-2] if len(versioned) >= 2 else ""

    if len(versioned) > limit_version_count:
        db.drop_collection(newest)
        print(f"Rolled back and dropped the newest collection: {newest}")
        return second_newest
    else:
        print("Rollback not performed because limit_version_count is reached.")
        return ""


def compare_model_performance(
    threshold_improvement: float = 0.02
) -> bool:
    load_dotenv()
    client = dbConnect.client
    db = dbConnect.db

    runs = list(db.model_runs.find().sort("trained_date", 1))
    if len(runs) < 2:
        print("No historical runs to compare.")
        return True

    previous_run = runs[-2]
    latest_run = runs[-1]

    prev_mase = previous_run["performance_metrics"].get("MASE")
    new_mase = latest_run["performance_metrics"].get("MASE")

    if prev_mase is None or new_mase is None:
        print("Cannot compare MASE; missing metrics. Defaulting to acceptance.")
        return True

    improvement = (prev_mase - new_mase) / prev_mase
    print(f"Model improvement over previous: {improvement:.2%}")

    better = improvement >= threshold_improvement
    return better
