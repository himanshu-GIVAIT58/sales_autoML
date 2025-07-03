# model_handler.py
"""
This module handles all interactions with the AutoGluon TimeSeriesPredictor.
It now includes both a full-pipeline prediction method for batch runs and a
new, fast prediction method for on-demand forecasts from user uploads.
"""

import pandas as pd
import os
import joblib  # <-- Add joblib to load artifacts
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# Import necessary components from your project
import config
from feature_engineering import create_seasonal_features, prepare_data, generate_static_features

# --- Core Model Functions (Unchanged) ---

def train_predictor(ts_data, config):
    """
    Trains the TimeSeriesPredictor with the given data and configuration.
    """
    try:
        print("\nTraining forecasting models with enhanced configuration...")
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)

        predictor = TimeSeriesPredictor(
            prediction_length=config.PREDICTION_LENGTH,
            path=config.MODEL_SAVE_PATH,
            target=config.TARGET_COLUMN,
            known_covariates_names=config.KNOWN_COVARIATES_NAMES,
            freq=config.FREQ,
            eval_metric=config.EVAL_METRIC,
            quantile_levels=config.QUANTILE_LEVELS
        ).fit(
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
    """
    Prints the model leaderboard and detailed evaluation metrics.
    """
    try:
        print("\n--- Model Leaderboard ---")
        leaderboard = predictor.leaderboard()
        print(leaderboard)

        print("\n--- Detailed Error Metrics ---")
        evaluation_summary = predictor.evaluate(ts_data)
        print(evaluation_summary)
        return evaluation_summary
    except Exception as e:
        print(f"❌ Error during model evaluation: {e}")
        raise

def load_latest_predictor(model_path: str = None):
    """
    Loads the latest saved AutoGluon TimeSeriesPredictor from the artifacts path.
    """
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

# --- Prediction Functions ---

def make_predictions(predictor, ts_data=None, holidays_df=None, user_uploaded_data=None):
    """
    (Slow Method) Generates predictions by running the full feature engineering pipeline.
    Suitable for the main batch training run where data consistency is paramount.
    """
    if predictor is None:
        raise ValueError("A trained predictor object must be provided.")

    try:
        print("\n🐌 Running SLOW prediction process (full pipeline)...")

        # --- Mode 2: User Uploaded Data (Full Re-processing) ---
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
            
        # --- Mode 1: Main Pipeline (Batch Prediction) ---
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

def make_fast_predictions(predictor, user_uploaded_data):
    """
    (Fast Method) Generates forecasts from a user-uploaded CSV using a lightweight
    process that relies on saved training artifacts for speed and consistency.
    """
    if predictor is None:
        raise ValueError("A trained predictor object must be provided.")

    print("\n🚀 Starting FAST prediction process using artifacts...")

    # 1. Load the saved training artifacts
    try:
        static_feature_columns = joblib.load('artifacts/static_feature_columns.joblib')
        holidays_df = pd.read_csv('artifacts/holidays.csv', parse_dates=['timestamp'])
    except FileNotFoundError as e:
        print(f"❌ Error: Artifact not found. Ensure 'main.py' has been run to save artifacts.")
        raise e

    # 2. Prepare user data and create item_id
    user_data = user_uploaded_data.copy()
    user_data['timestamp'] = pd.to_datetime(user_data['timestamp'])
    user_data['channel'] = 'Online'
    user_data['item_id'] = user_data['sku'].astype(str) + "_" + user_data['channel']

    # 3. Generate static features aligned with the training set
    static_features = generate_static_features(user_data, all_training_columns=static_feature_columns)
    static_features.reset_index(inplace=True) # Ensure item_id is a column

    # 4. Create TimeSeriesDataFrame
    ts_upload = TimeSeriesDataFrame.from_data_frame(
        user_data, id_column='item_id', timestamp_column='timestamp'
    )

    # 5. Generate future known covariates
    future_known_covariates = predictor.make_future_data_frame(ts_upload)
    future_known_covariates = create_seasonal_features(future_known_covariates)
    future_known_covariates = pd.merge(
        future_known_covariates, holidays_df[['timestamp', 'is_holiday']],
        on='timestamp', how='left'
    ).fillna(0)

    # 6. Make the prediction
    print("  -> Generating forecasts...")
    predictions = predictor.predict(
        ts_upload,
        static_features=static_features,
        known_covariates=future_known_covariates
    )
    print("✅ Forecasts generated successfully!")
    return predictions
