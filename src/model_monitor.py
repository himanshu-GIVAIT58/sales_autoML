import datetime
import pandas as pd
from pymongo import MongoClient
import os
import numpy as np
from autogluon.timeseries.predictor import TimeSeriesPredictor
from pymongo.database import Database
from scipy.stats import ks_2samp, chi2_contingency
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

# Assuming dbConnect.py handles the MongoDB connection
from src import dbConnect

# --- Dataclass for Structured Output ---
@dataclass
class ModelValidationResult:
    """A structured result of the model validation pipeline."""
    decision: str  # "promote" or "reject"
    reasons: List[str]
    data_drift_report: Dict[str, Any]
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
        # Capture more detailed model information from AutoGluon
        model_details = {
            "best_model": predictor.model_best,
            "model_names": predictor.model_names(),
            "leaderboard": predictor.leaderboard().to_dict("records"),
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
        "data_drift_report": validation_result.data_drift_report,
    }

    db.model_runs.insert_one(run_doc)
    print(f"✅ Logged model run and validation results to 'model_runs' collection.")


def check_for_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    p_value_threshold: float = 0.05,
    max_categories: int = 20,
) -> Dict[str, Any]:
    """
    Performs comprehensive drift detection for both numeric and categorical features.
    - Numeric drift: Kolmogorov-Smirnov (KS) test.
    - Categorical drift: Chi-squared test.
    """
    drift_report = {"drift_detected": False, "details": {}}
    if reference_data.empty or current_data.empty:
        drift_report["reasons"] = ["Reference or current data is empty."]
        return drift_report

    common_cols = reference_data.columns.intersection(current_data.columns)
    drifted_features = []

    for col in common_cols:
        # 1. Numeric Feature Drift (KS Test)
        if pd.api.types.is_numeric_dtype(reference_data[col]):
            stat, p_value = ks_2samp(reference_data[col].dropna(), current_data[col].dropna())
            if p_value < p_value_threshold:
                drifted_features.append(col)
                drift_report["details"][col] = {"type": "numeric", "p_value": p_value, "drifted": True}
        
        # 2. Categorical Feature Drift (Chi-Squared Test)
        elif pd.api.types.is_object_dtype(reference_data[col]) or pd.api.types.is_categorical_dtype(reference_data[col]):
            # Skip high-cardinality features for chi2 test to be meaningful
            if reference_data[col].nunique() > max_categories:
                continue
            
            contingency_table = pd.crosstab(
                reference_data[col].astype(str), 
                current_data[col].astype(str)
            )
            try:
                chi2, p_value, _, _ = chi2_contingency(contingency_table)
                if p_value < p_value_threshold:
                    drifted_features.append(col)
                    drift_report["details"][col] = {"type": "categorical", "p_value": p_value, "drifted": True}
            except ValueError: # Happens if a category exists in one but not the other
                drift_report["details"][col] = {"type": "categorical", "error": "Category mismatch"}


    if drifted_features:
        drift_report["drift_detected"] = True
        drift_report["reasons"] = [f"Significant drift detected in features: {', '.join(drifted_features)}"]
    
    return drift_report


def compare_model_performance(
    new_metrics: Dict[str, float],
    champion_metrics: Dict[str, float],
    threshold_improvement: float = 0.02
) -> Dict[str, Any]:
    """Compares new model performance against the champion model."""
    comparison = {"is_better": False, "reasons": []}
    
    if not champion_metrics:
        comparison["is_better"] = True
        comparison["reasons"].append("No champion model to compare against. New model promoted by default.")
        return comparison

    prev_mase = champion_metrics.get("MASE")
    new_mase = new_metrics.get("MASE")

    if prev_mase is None or new_mase is None:
        comparison["is_better"] = True # Default to accept if metric is missing
        comparison["reasons"].append("MASE metric missing, cannot compare. Promoting by default.")
        return comparison

    # For MASE, lower is better.
    if new_mase < prev_mase:
        improvement = abs((new_mase - prev_mase) / prev_mase) if prev_mase != 0 else float('inf')
        comparison["reasons"].append(f"New model improved MASE by {improvement:.2%}.")
        if improvement >= threshold_improvement:
            comparison["is_better"] = True
            comparison["reasons"].append(f"Improvement meets or exceeds the {threshold_improvement:.0%} threshold.")
        else:
            comparison["reasons"].append(f"Improvement is below the {threshold_improvement:.0%} threshold.")
    else:
        comparison["reasons"].append("New model MASE is worse than or equal to the champion model.")

    comparison["champion_mase"] = prev_mase
    comparison["new_mase"] = new_mase
    
    return comparison


def manage_recommendation_versions(
    decision: str, 
    new_collection_name: str, 
    versions_to_keep: int = 5
):
    """Archives old versions or rolls back a rejected model's output."""
    db = dbConnect.db
    
    if decision == "promote":
        # On promotion, clean up old versions beyond the keep limit
        versioned_collections = sorted([
            coll for coll in db.list_collection_names() 
            if coll.startswith("inventory_recommendations_")
        ])
        
        if len(versioned_collections) > versions_to_keep:
            collections_to_drop = versioned_collections[:-versions_to_keep]
            for coll in collections_to_drop:
                db.drop_collection(coll)
                print(f"🧹 Archived old recommendations: {coll}")
                
    elif decision == "reject":
        # On rejection, drop the newly created (but rejected) recommendations
        db.drop_collection(new_collection_name)
        print(f"🗑️ Rolled back and dropped rejected recommendations: {new_collection_name}")


# --- Main Orchestration Pipeline ---

def run_model_validation_pipeline(
    new_predictor: TimeSeriesPredictor,
    new_training_data: pd.DataFrame,
    new_performance_metrics: Dict[str, float],
) -> ModelValidationResult:
    """Orchestrates the model validation process."""
    print("🚀 Starting Model Validation Pipeline...")
    db = dbConnect.db
    reasons = []

    # 1. Get the current champion model and its data snapshot
    champion_run = db.model_runs.find_one(
        {"model_promotion_decision": "promote"},
        sort=[("trained_date", -1)]
    )
    
    if not champion_run:
        print("No champion model found. Promoting new model by default.")
        decision = "promote"
        reasons.append("No champion model exists for comparison.")
        drift_report = {}
        perf_comparison = {}
        champion_id = "None"
    else:
        champion_id = champion_run.get("_id")
        print(f"Found champion model from run ID: {champion_id}")
        
        # 2. Check for Data Drift
        champion_snapshot_info = champion_run.get("data_snapshot_info", {})
        champion_data_collection = champion_snapshot_info.get("collection_name")
        
        if champion_data_collection:
            champion_data = pd.DataFrame(list(db[champion_data_collection].find()))
            drift_report = check_for_drift(champion_data, new_training_data)
            reasons.extend(drift_report.get("reasons", []))
            print(f"Data Drift Check: {'Drift Detected' if drift_report['drift_detected'] else 'No Significant Drift'}")
        else:
            drift_report = {"drift_detected": False, "reasons": ["Champion data snapshot not found."]}

        # 3. Compare Model Performance
        champion_metrics = champion_run.get("performance_metrics", {})
        perf_comparison = compare_model_performance(new_performance_metrics, champion_metrics)
        reasons.extend(perf_comparison.get("reasons", []))
        print(f"Performance Check: {'New model is better' if perf_comparison['is_better'] else 'New model is not better'}")

        # 4. Make Promotion Decision
        is_promotable = perf_comparison["is_better"] and not drift_report["drift_detected"]
        decision = "promote" if is_promotable else "reject"

    result = ModelValidationResult(
        decision=decision,
        reasons=reasons,
        data_drift_report=drift_report,
        performance_comparison=perf_comparison,
        new_model_metrics=new_performance_metrics,
        champion_model_id=str(champion_id) if champion_id else "None"
    )

    print(f"🏁 Validation Pipeline Decision: **{result.decision.upper()}**")
    return result
