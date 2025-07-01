#### filepath: c:\Users\User\Desktop\sales_autoML\model_monitor.py
import datetime
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import numpy as np
from autogluon.timeseries.predictor import TimeSeriesPredictor

def log_model_run(
    predictor: TimeSeriesPredictor,
    collection_name: str,
    performance_metrics: dict,
    data_snapshot_info: dict = None,
    trigger_source: str = "auto",
):
    """
    Logs essential information about a model run in 'model_runs' collection.

    :param predictor: The trained TimeSeriesPredictor (for feature importance, etc.).
    :param collection_name: The name of the MongoDB collection where recommendations are stored.
    :param performance_metrics: Dictionary of your run's evaluation metrics, e.g. MASE, RMSE, etc.
    :param data_snapshot_info: Optional info about the data used (time range, version, etc.).
    :param trigger_source: Short description of what triggered this run (manual, pipeline, etc.).
    """
    load_dotenv()
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017/"))
    db = client[os.getenv("MONGO_DB", "sales_automl")]

    # (Optional) Retrieve feature importances for analysis
    try:
        feature_importances = predictor.feature_importance().to_dict()
    except Exception:
        feature_importances = {}

    # Log metadata
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
    threshold: float = 0.1
) -> bool:
    """
    Basic example of data drift detection using mean difference.
    Real-world drift checks can be more advanced (KS tests, distribution checks, etc.).

    :param previous_data: A reference (older) dataset to compare against.
    :param new_data: The new or current dataset to check.
    :param threshold: A simple threshold for average difference. Adjust for your domain.
    :return: True if drift is detected, otherwise False.
    """
    if previous_data.empty or new_data.empty:
        return False

    # Ensure the two dataframes have the same columns
    common_cols = previous_data.columns.intersection(new_data.columns)
    if len(common_cols) == 0:
        return False

    # Simple example: compute avg difference in means across the columns
    diffs = []
    for col in common_cols:
        if pd.api.types.is_numeric_dtype(previous_data[col]):
            old_mean = previous_data[col].mean()
            new_mean = new_data[col].mean()
            if old_mean != 0:  # avoid divide-by-zero
                diff_ratio = abs((new_mean - old_mean) / old_mean)
                diffs.append(diff_ratio)

    if not diffs:
        return False

    # If the average difference ratio across columns exceeds a threshold => drift
    avg_diff = np.mean(diffs)
    drift_detected = avg_diff > threshold
    return drift_detected

def rollback_to_previous_version(
    db: MongoClient,
    limit_version_count: int = 2
) -> str:
    """
    Rolls back to the previous versioned recommendations if the new one underperforms.
    This is a simplistic approach that just drops the newest collection.

    :param db: A connected MongoClient database handle.
    :param limit_version_count: Ensure you keep at least N latest versions, e.g., 2.
    :return: The name of the active version after rollback, or empty if no rollback performed.
    """
    # Filter only versioned collections
    versioned = [coll for coll in db.list_collection_names() if coll.startswith("inventory_recommendations_")]
    versioned.sort()  # oldest -> newest (YYYYMMDD_HHMMSS sorts effectively in ascending lexicographical order)

    if len(versioned) <= 1:
        print("No major rollback possible because only one or zero versioned collections exist.")
        return ""

    # Drop the newest collection to revert
    newest = versioned[-1]
    second_newest = versioned[-2] if len(versioned) >= 2 else None

    # But ensure we keep at least 'limit_version_count' versions
    if len(versioned) > limit_version_count:
        db.drop_collection(newest)
        print(f"Rolled back and dropped the newest collection: {newest}")
        return second_newest
    else:
        print("Rollback not performed because limit_version_count is reached.")
        return ""

def compare_model_performance(
    db: MongoClient,
    current_run_collection: str
) -> bool:
    """
    This function compares the current run's performance to the previous run.
    In practice, you might look at MASE, RMSE, or custom business KPIs.
    Returns True if current run is better than the previous run; otherwise False.
    """
    runs = list(db.model_runs.find().sort("trained_date", 1))
    if len(runs) < 2:
        # Not enough data to compare
        print("No historical runs to compare.")
        return True

    # last run before current
    previous_run = runs[-2]
    latest_run = runs[-1]

    # Example comparison by MASE
    prev_mase = previous_run["performance_metrics"].get("MASE", None)
    new_mase = latest_run["performance_metrics"].get("MASE", None)

    if prev_mase is None or new_mase is None:
        print("Cannot compare MASE; missing metrics. Defaulting to acceptance.")
        return True

    # If the new MASE is lower, performance is better
    better = new_mase < prev_mase
    return better
