
"""
This module handles all interactions with the AutoGluon TimeSeriesPredictor.
It now includes both a full-pipeline prediction method for batch runs and a
new, fast prediction method for on-demand forecasts from user uploads.
"""

import pandas as pd
import os
import joblib  
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame


from src import config
from src.feature_engineering import create_seasonal_features, prepare_data, generate_static_features, create_inventory_features, create_price_elasticity_features, create_trend_features

def train_predictor(ts_data, config):
    """
    Trains a TimeSeriesPredictor. AutoGluon will automatically fine-tune
    an existing model if one is found at config.MODEL_SAVE_PATH.
    """
    try:
        print("\nTraining or fine-tuning forecasting models...")
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)

        
        
        predictor = TimeSeriesPredictor(
            prediction_length=config.PREDICTION_LENGTH,
            path=config.MODEL_SAVE_PATH,
            target=config.TARGET_COLUMN,
            known_covariates_names=config.KNOWN_COVARIATES_NAMES,
            freq=config.FREQ,
            eval_metric=config.EVAL_METRIC,
            quantile_levels=config.QUANTILE_LEVELS
        )
        predictor.fit(
            ts_data,
            presets=config.AUTOGLUON_PRESETS,
            time_limit=config.TIME_LIMIT,
            num_val_windows=config.NUM_VAL_WINDOWS,

        )

        print("Model training completed.")
        return predictor
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        raise

def evaluate_predictor(predictor, ts_data):
    try:
        print("\n--- Model Leaderboard ---")
        leaderboard = predictor.leaderboard(ts_data)
        print(leaderboard)

        print("\n--- Detailed Error Metrics ---")
        
        
        evaluation_summary = predictor.evaluate(ts_data, display=False)
        print(evaluation_summary)

        
        return evaluation_summary
    except Exception as e:
        print(f"❌ Error during model evaluation: {e}")
        raise

from typing import Optional

def load_latest_predictor(model_path: Optional[str] = None):
    try:
        model_dir = model_path if model_path else config.MODEL_SAVE_PATH
        if not os.path.exists(model_dir):
            print(f"Model directory not found: {model_dir}")
            return None
        print(f"Loading predictor from: {model_dir}")
        return TimeSeriesPredictor.load(model_dir)
    except Exception as e:
        print(f"Error loading predictor: {e}")
        return None

def make_predictions(predictor, ts_data=None, holidays_df=None, user_uploaded_data=None):
    if predictor is None:
        raise ValueError("A trained predictor object must be provided.")

    try:
        print("\n🐌 Running SLOW prediction process (full pipeline)...")

        
        if user_uploaded_data is not None:
            print("  -> Adapting uploaded data to the main pipeline format...")
            source_df = user_uploaded_data.copy()
            source_df.rename(columns={"timestamp": "created_at", "target": "qty"}, inplace=True)
            source_df.update({'category': 'unknown', 'gender': 'unknown', 'Channel': 'Online'})

            inventory_df = pd.DataFrame(columns=['date', 'sku', 'wh'])
            if holidays_df is None:
                holidays_df = pd.DataFrame(columns=['Date'])

            print("  -> Running the full feature engineering pipeline...")
            ts_upload, static_features_upload = prepare_data(
                source_data=source_df,
                inventory_data=inventory_df,
                holidays_data=holidays_df
            )
            
            print("  -> Generating future covariates...")
            future_known_covariates = predictor.make_future_data_frame(ts_upload)
            
            print("  -> Generating forecasts...")
            predictions = predictor.predict(
                ts_upload,
                known_covariates=future_known_covariates,
                static_features=static_features_upload
            )
            print("  -> ✅ Forecasts generated successfully!")
            return predictions
            
        
        elif ts_data is not None:
            print("  -> Prediction Mode: Main Pipeline")
            future_known_covariates = predictor.make_future_data_frame(data=ts_data)
            future_known_covariates.reset_index(inplace=True)
            future_known_covariates = create_seasonal_features(future_known_covariates)
            if holidays_df is not None:
                future_known_covariates = pd.merge(
                    future_known_covariates, holidays_df[['timestamp', 'is_holiday']],
                    on="timestamp", how="left"
                ).fillna(0)
            for col in config.KNOWN_COVARIATES_NAMES:
                if col not in future_known_covariates.columns:
                    future_known_covariates[col] = 0
            predictions = predictor.predict(ts_data, known_covariates=future_known_covariates)
            print("  -> Forecasts for main pipeline generated successfully.")
            return predictions
        else:
            raise ValueError("Either 'ts_data' or 'user_uploaded_data' must be provided.")

    except Exception as e:
        print(f"❌ Error during prediction generation: {e}")
        raise

def load_prediction_artifacts():
    """Loads the artifacts needed for making predictions."""
    try:
        static_columns_path = os.path.join(config.ARTIFACTS_PATH, 'static_feature_columns.joblib')
        holidays_path = os.path.join(config.ARTIFACTS_PATH, 'holidays.csv')
        static_feature_columns = joblib.load(static_columns_path)
        holidays_df = pd.read_csv(holidays_path, parse_dates=['timestamp'])
        return static_feature_columns, holidays_df
    except FileNotFoundError as e:
        print(f"❌ Error: Artifact not found at '{config.ARTIFACTS_PATH}'.")
        raise e

def prepare_prediction_data(user_data_raw, holidays_df):
    """Enriches the raw user-uploaded data with all necessary features."""
    user_data = user_data_raw.copy()
    user_data['timestamp'] = pd.to_datetime(user_data['timestamp'])
    user_data = create_seasonal_features(user_data)
    user_data = pd.merge(user_data, holidays_df[['timestamp', 'is_holiday']], on='timestamp', how='left').fillna({'is_holiday': 0})
    user_data['warehouse_qty'] = 1
    user_data = create_inventory_features(user_data)
    user_data = create_price_elasticity_features(user_data)
    user_data['sku'] = user_data['sku'].astype(str)
    user_data = create_trend_features(user_data)
    for col in config.KNOWN_COVARIATES_NAMES:
        if col not in user_data.columns:
            user_data[col] = 0
    return user_data

def generate_future_covariates(predictor, ts_data, holidays_df):
    """Generates the known covariates for the future prediction window."""
    future_covariates = predictor.make_future_data_frame(ts_data)
    future_covariates = create_seasonal_features(future_covariates)
    future_covariates = pd.merge(future_covariates, holidays_df[['timestamp', 'is_holiday']], on='timestamp', how='left').fillna(0)
    for col in config.KNOWN_COVARIATES_NAMES:
        if col not in future_covariates.columns:
            future_covariates[col] = 0
    return future_covariates


def make_fast_predictions(predictor, user_uploaded_data):
    """
    Orchestrates the fast prediction process, from loading artifacts
    to safely evaluating and forecasting. This function always returns
    a tuple of (predictions, metrics).
    """
    if predictor is None:
        raise ValueError("A trained predictor object must be provided.")

    # 1. Load artifacts and prepare data
    static_feature_columns, holidays_df = load_prediction_artifacts()
    enriched_data = prepare_prediction_data(user_uploaded_data, holidays_df)

    # 2. Create TimeSeriesDataFrame
    enriched_data['channel'] = 'Online'
    enriched_data['item_id'] = enriched_data['sku'].astype(str) + "_" + enriched_data['channel']
    static_features = generate_static_features(enriched_data, all_training_columns=static_feature_columns)
    static_features.reset_index(inplace=True)
    ts_upload = TimeSeriesDataFrame.from_data_frame(
        enriched_data,
        id_column='item_id',
        timestamp_column='timestamp',
        static_features_df=static_features
    )

    # 3. Evaluate performance (with safety check)
    metrics = None
    min_series_length = ts_upload.index.get_level_values('item_id').value_counts().min()
    
    if min_series_length > predictor.prediction_length:
        print("  -> Data is long enough. Evaluating model performance...")
        # --- KEY FIX ---
        # Assign the direct output of evaluate() to metrics, without unpacking.
        metrics = predictor.evaluate(ts_upload, display=False)
    else:
        print(f"  -> Data is too short to evaluate. Skipping evaluation.")
        reason = f"Uploaded data history ({min_series_length} points) is not longer than the model's prediction length ({predictor.prediction_length} points)."
        metrics = pd.DataFrame([{"info": "Evaluation skipped", "reason": reason}])

    # 4. Generate future forecast
    future_known_covariates = generate_future_covariates(predictor, ts_upload, holidays_df)
    predictions = predictor.predict(
        ts_upload,
        known_covariates=future_known_covariates
    )
    
    print("✅ Forecasts and metrics generated successfully!")
    return predictions, metrics
