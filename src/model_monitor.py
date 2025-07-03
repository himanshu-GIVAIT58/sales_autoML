
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
    data_snapshot_info: 'dict | None' = None,
    trigger_source: str = "auto",
):
    load_dotenv()
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017/"))
    db = client[os.getenv("MONGO_DB", "sales_automl")]

    
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

    
    common_cols = previous_data.columns.intersection(new_data.columns)
    if len(common_cols) == 0:
        return False

    
    diffs = []
    for col in common_cols:
        if pd.api.types.is_numeric_dtype(previous_data[col]):
            old_mean = previous_data[col].mean()
            new_mean = new_data[col].mean()
            if old_mean != 0:  
                diff_ratio = abs((new_mean - old_mean) / old_mean)
                diffs.append(diff_ratio)

    if not diffs:
        return False

    
    avg_diff = np.mean(diffs)
    drift_detected = avg_diff > threshold
    return bool(drift_detected)

from pymongo.database import Database

def rollback_to_previous_version(
    db: Database,
    limit_version_count: int = 2
) -> str:
    """
    Rolls back to the previous versioned recommendations if the new one underperforms.
    This is a simplistic approach that just drops the newest collection.

    :param db: A connected MongoClient database handle.
    :param limit_version_count: Ensure you keep at least N latest versions, e.g., 2.
    :return: The name of the active version after rollback, or empty if no rollback performed.
    """
    
    versioned = [coll for coll in db.list_collection_names() if coll.startswith("inventory_recommendations_")]
    versioned.sort()  

    if len(versioned) <= 1:
        print("No major rollback possible because only one or zero versioned collections exist.")
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
    db: MongoClient,
    current_run_collection: str
) -> bool:
    runs = list(db.model_runs.find().sort("trained_date", 1))
    if len(runs) < 2:
        print("No historical runs to compare.")
        return True

    
    previous_run = runs[-2]
    latest_run = runs[-1]

    
    prev_mase = previous_run["performance_metrics"].get("MASE", None)
    new_mase = latest_run["performance_metrics"].get("MASE", None)

    if prev_mase is None or new_mase is None:
        print("Cannot compare MASE; missing metrics. Defaulting to acceptance.")
        return True

    
    better = new_mase < prev_mase
    return better
