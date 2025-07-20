import datetime
import pandas as pd
from pymongo import MongoClient
import os
import numpy as np
from autogluon.timeseries.predictor import TimeSeriesPredictor
from pymongo.database import Database
from dataclasses import dataclass
from typing import Any, Dict, List

# Assuming dbConnect.py handles the MongoDB connection
from src import dbConnect

# --- Dataclass for Structured Output ---
@dataclass
class ModelValidationResult:
    """A structured result of the model validation pipeline."""
    decision: str  # "promote" or "reject"
    reasons: List[str]
    performance_comparison: Dict[str, Any]
    new_model_metrics: Dict[str, Any]
    champion_model_id: Any


# --- Enhanced Monitoring Functions ---

def log_model_run(
    predictor: TimeSeriesPredictor,
    collection_name: str,
    performance_metrics: dict,
    validation_result: ModelValidationResult,
    data_snapshot_info: dict,
    trigger_source: str = "auto",
):
    """Logs a comprehensive record of the model training and validation run."""
    db = dbConnect.db
    
    try:
        model_details = {
            "best_model": predictor.model_best,
            "model_names": predictor.model_names(),
            "leaderboard": predictor.leaderboard().to_dict("records"),
            "path": predictor.path
        }
        feature_importances = predictor.feature_importance().to_dict()
    except Exception as e:
        print(f"Warning: Could not capture full model details. Error: {e}")
        model_details = {}
        feature_importances = {}

    run_doc = {
        "run_id": f"{collection_name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        "trained_date": datetime.datetime.now(datetime.timezone.utc),
        "model_promotion_decision": validation_result.decision,
        "validation_reasons": validation_result.reasons,
        "performance_metrics": performance_metrics,
        "champion_model_id": validation_result.champion_model_id,
        "data_snapshot_info": data_snapshot_info,
        "trigger_source": trigger_source,
        "model_details": model_details,
        "feature_importances": feature_importances,
    }

    db.model_runs.insert_one(run_doc)
    print("✅ Logged model run and validation results to 'model_runs' collection.")


def compare_model_performance(
    new_metrics: Dict[str, float],
    champion_metrics: Dict[str, float],
    threshold_improvement: float = 0.02
) -> Dict[str, Any]:
    """Compares new model performance against the champion model."""
    # This function's logic remains the same to provide a performance report
    # but its output will no longer be used for the final decision.
    comparison = {"is_better": False, "reasons": []}
    
    if not champion_metrics:
        comparison["is_better"] = True
        comparison["reasons"].append("No champion model to compare against. New model promoted by default.")
        return comparison

    prev_metric, new_metric, metric_name = None, None, None

    if "MASE" in champion_metrics and "MASE" in new_metrics:
        prev_metric = champion_metrics.get("MASE")
        new_metric = new_metrics.get("MASE")
        metric_name = "MASE"
    elif "RMSE" in champion_metrics and "RMSE" in new_metrics:
        prev_metric = champion_metrics.get("RMSE")
        new_metric = new_metrics.get("RMSE")
        metric_name = "RMSE"
    else:
        comparison["is_better"] = True
        comparison["reasons"].append("Metric type mismatch or missing. Promoting by default.")
        return comparison

    if new_metric < prev_metric:
        improvement = abs((new_metric - prev_metric) / prev_metric) if prev_metric != 0 else float('inf')
        comparison["reasons"].append(f"New model improved {metric_name} by {improvement:.2%}.")
        if improvement >= threshold_improvement:
            comparison["is_better"] = True
            comparison["reasons"].append(f"Improvement meets or exceeds the {threshold_improvement:.0%} threshold.")
        else:
            comparison["reasons"].append(f"Improvement is below the {threshold_improvement:.0%} threshold.")
    else:
        comparison["reasons"].append(f"New model {metric_name} is worse than or equal to the champion model.")

    comparison["champion_metric"] = prev_metric
    comparison["new_metric"] = new_metric
    comparison["metric_name"] = metric_name
    
    return comparison


def manage_recommendation_versions(
    decision: str, 
    new_collection_name: str, 
    versions_to_keep: int = 5
):
    """
    Archives old versions of promoted recommendations.
    This version will NOT delete rejected recommendations.
    """
    db = dbConnect.db
    
    if decision == "promote":
        versioned_collections = sorted([
            coll for coll in db.list_collection_names() 
            if coll.startswith("inventory_recommendations_")
        ])
        if len(versioned_collections) > versions_to_keep:
            collections_to_drop = versioned_collections[:-versions_to_keep]
            for coll in collections_to_drop:
                db.drop_collection(coll)
                print(f"🧹 Archived old recommendations: {coll}")
    # The 'elif decision == "reject"' block has been completely removed.


# --- Main Orchestration Pipeline ---

def run_model_validation_pipeline(
    new_predictor: TimeSeriesPredictor,
    new_training_data: pd.DataFrame,
    new_performance_metrics: Dict[str, float],
) -> ModelValidationResult:
    """
    Orchestrates the model validation process.
    This version ALWAYS promotes the new model.
    """
    print("🚀 Starting Model Validation Pipeline...")
    db = dbConnect.db
    reasons = []

    # 1. Get the current champion model for logging purposes
    champion_run = db.model_runs.find_one(
        {"model_promotion_decision": "promote"},
        sort=[("trained_date", -1)]
    )
    
    if not champion_run:
        champion_id = "None"
        perf_comparison = {}
        reasons.append("No champion model found to compare against.")
    else:
        champion_id = champion_run.get("_id")
        print(f"Found champion model from run ID: {champion_id}")
        
        # We still run the comparison to log the results, but it won't affect the decision
        champion_metrics = champion_run.get("performance_metrics", {})
        perf_comparison = compare_model_performance(new_performance_metrics, champion_metrics)
        reasons.extend(perf_comparison.get("reasons", []))
        print(f"Performance Check (for logging only): {'New model is better' if perf_comparison.get('is_better') else 'New model is not better'}")

    # 2. Make Promotion Decision: ALWAYS PROMOTE
    decision = "promote"
    reasons.append("Policy: Always promote the latest model.")

    result = ModelValidationResult(
        decision=decision,
        reasons=reasons,
        performance_comparison=perf_comparison,
        new_model_metrics=new_performance_metrics,
        champion_model_id=str(champion_id) if champion_id else "None"
    )

    print(f"🏁 Validation Pipeline Decision: **{result.decision.upper()}**")
    return result
