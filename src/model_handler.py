import pandas as pd
import os
from typing import Optional, Dict, List
import joblib  
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from src import config
from src.feature_engineering import create_seasonal_features, prepare_data, generate_static_features, create_inventory_features, create_price_elasticity_features, create_trend_features

def train_predictor(ts_data, config):
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
            random_seed=config.RANDOM_SEED
        )
        print("Model training completed.")
        return predictor
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        raise


# Add this new function to src/model_handler.py

def refit_final_ensemble(predictors: Dict[str, TimeSeriesPredictor], ts_data: TimeSeriesDataFrame) -> TimeSeriesPredictor:
    """
    Takes the best models from multiple predictors and refits them into a single, final ensemble.
    """
    print("\nCreating final ensemble from all presets...")
    
    # 1. Collect the names of the best models from each preset's leaderboard
    best_models_from_each_preset = []
    for preset, predictor in predictors.items():
        best_model_name = predictor.model_best
        # Ensure we don't add the top-level ensemble itself, only base models
        if "WeightedEnsemble" not in best_model_name:
             best_models_from_each_preset.append(best_model_name)
    
    # Remove duplicates
    best_models_from_each_preset = sorted(list(set(best_models_from_each_preset)))
    
    print(f"  -> Found {len(best_models_from_each_preset)} unique best models to ensemble: {best_models_from_each_preset}")

    # 2. Use the highest quality predictor as the base for refitting the final ensemble
    # Assumes presets are ordered from lowest to highest quality in config
    final_predictor = list(predictors.values())[-1] 
    
    # 3. Refit a new WeightedEnsemble using only the best models from all runs
    final_predictor.fit(
        train_data=ts_data,
        hyperparameters={
            "WeightedEnsemble": {
                "models_to_ensemble": best_models_from_each_preset
            }
        },
        time_limit=300 # Add a short time limit just for ensembling
    )
    
    print("✅ Final ensembled predictor created successfully.")
    return final_predictor

    
def train_multiple_predictors(ts_data: TimeSeriesDataFrame, config) -> Dict[str, TimeSeriesPredictor]:
    """
    Trains multiple TimeSeriesPredictor models, one for each preset specified in the config.
    Returns a dictionary of trained predictors.
    """
    predictors = {}
    presets = config.TRAINING['PRESETS']  # Expects a list, e.g., ["medium_quality", "best_quality"]

    print(f"\nTraining models for presets: {presets}")
    for preset in presets:
        print("-" * 40)
        print(f"🚀 Starting training for preset: '{preset}'")
        
        # Create a unique path for each predictor to avoid overwriting
        model_path = os.path.join(config.MODEL_SAVE_PATH, preset)
        os.makedirs(model_path, exist_ok=True)

        try:
            predictor = TimeSeriesPredictor(
                prediction_length=config.TRAINING['PREDICTION_LENGTH'],
                path=model_path,
                target=config.DATA_COLUMNS['TARGET'],
                known_covariates_names=config.KNOWN_COVARIATES_NAMES,
                freq=config.TRAINING['FREQ'],
                eval_metric=config.TRAINING['EVAL_METRIC'],
                quantile_levels=config.TRAINING['QUANTILE_LEVELS']
            )
            predictor.fit(
                ts_data,
                presets=preset,
                time_limit=config.TRAINING['TIME_LIMIT'],
                num_val_windows=config.TRAINING['NUM_VAL_WINDOWS'],
                # random_seed=config.TRAINING['RANDOM_SEED']
            )
            predictors[preset] = predictor
            print(f"✅ Model training completed for preset: '{preset}'")
        except Exception as e:
            print(f"❌ Error during model training for preset '{preset}': {e}")
            # Continue to the next preset even if one fails
            continue
            
    if not predictors:
        raise RuntimeError("All model training presets failed. No predictors were trained.")
        
    return predictors

def make_ensembled_predictions(
    predictors: Dict[str, TimeSeriesPredictor], 
    ts_data: TimeSeriesDataFrame, 
    holidays_df: pd.DataFrame
) -> TimeSeriesDataFrame:
    all_predictions = []
    
    first_predictor = next(iter(predictors.values()))
    future_known_covariates = generate_future_covariates(first_predictor, ts_data, holidays_df)

    print("\nGenerating forecasts from each trained model preset...")
    for preset_name, predictor in predictors.items():
        print(f"   -> Predicting with '{preset_name}' model...")
        try:
            predictions = predictor.predict(ts_data, known_covariates=future_known_covariates)
            predictions["model_preset"] = preset_name
            all_predictions.append(predictions.reset_index())
        except Exception as e:
            print(f"   -> ⚠️ Could not generate predictions for preset '{preset_name}': {e}")
            continue

    if not all_predictions:
        raise RuntimeError("Failed to generate predictions from any model.")

    combined_predictions_df = pd.concat(all_predictions, ignore_index=True)

    print("\nEnsembling predictions by averaging...")
    quantile_cols = [str(q) for q in first_predictor.quantile_levels]
    cols_to_average = ["mean"] + quantile_cols
    
    ensembled_df = combined_predictions_df.groupby(["item_id", "timestamp"])[cols_to_average].mean()
    
    # Convert the ensembled result back to a TimeSeriesDataFrame
    # This sets the index correctly before returning
    ensembled_ts_df = TimeSeriesDataFrame(ensembled_df)
    
    print("✅ Ensembled forecast generated successfully.")
    return ensembled_ts_df
def evaluate_predictors(predictors: Dict[str, TimeSeriesPredictor], ts_data: TimeSeriesDataFrame) -> pd.DataFrame:
    """Evaluates all trained predictors and returns a combined leaderboard."""
    all_leaderboards = []
    print("\n--- Model Leaderboards ---")
    for preset_name, predictor in predictors.items():
        try:
            print(f"\n--- Leaderboard for preset: '{preset_name}' ---")
            leaderboard = predictor.leaderboard(ts_data)
            leaderboard['preset'] = preset_name
            print(leaderboard)
            all_leaderboards.append(leaderboard)
        except Exception as e:
             print(f"  -> ⚠️ Could not generate leaderboard for preset '{preset_name}': {e}")
    
    if not all_leaderboards:
        return pd.DataFrame()
        
    return pd.concat(all_leaderboards, ignore_index=True)

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

    
    holidays_df = holidays_df.copy()
    
    holidays_df.columns = holidays_df.columns.str.lower()
    
    if 'timestamp' not in holidays_df.columns:
        if 'date' in holidays_df.columns:
            holidays_df['timestamp'] = pd.to_datetime(holidays_df['date'], errors='coerce')
        elif 'date' in [c.lower() for c in holidays_df.columns]:
            
            date_col = [c for c in holidays_df.columns if c.lower() == 'date'][0]
            holidays_df['timestamp'] = pd.to_datetime(holidays_df[date_col], errors='coerce')
        else:
            holidays_df['timestamp'] = pd.NaT
    if 'is_holiday' not in holidays_df.columns:
        holidays_df['is_holiday'] = 1

    print('user_data_before',user_data.columns.tolist())

    user_data = create_seasonal_features(user_data)

    print('user_data_after',user_data.columns.tolist())

    user_data = pd.merge(
        user_data,
        holidays_df[['timestamp', 'is_holiday']],
        on='timestamp',
        how='left'
    ).fillna({'is_holiday': 0})

    print('user_data_after_merge',user_data.columns.tolist())

    user_data['warehouse_qty'] = 1
    user_data = create_inventory_features(user_data)
    print('user_data_create_inventory',user_data.columns.to_list())

    
    

    user_data['sku'] = user_data['sku'].astype(str)
    user_data = create_trend_features(user_data)
    print('user_data_after_trend',user_data)

    for col in config.KNOWN_COVARIATES_NAMES:
        if col not in user_data.columns:
            user_data[col] = 0
    return user_data

def generate_future_covariates(predictor, ts_data, holidays_df):
    """Generates the known covariates for the future prediction window."""
    future_covariates = predictor.make_future_data_frame(data=ts_data)
    future_covariates.reset_index(inplace=True)
    future_covariates = create_seasonal_features(future_covariates)
    
    if holidays_df is not None:
        future_covariates = pd.merge(
            future_covariates, holidays_df[['timestamp', 'is_holiday']], 
            on="timestamp", how="left"
        ).fillna(0)
        
    # Ensure all required columns exist
    for col in config.KNOWN_COVARIATES_NAMES:
        if col not in future_covariates.columns:
            future_covariates[col] = 0
            
    return future_covariates

def make_fast_predictions(predictor, user_uploaded_data):
    if predictor is None:
        raise ValueError("A trained predictor object must be provided.")

    
    static_feature_columns, holidays_df = load_prediction_artifacts()
    enriched_data = prepare_prediction_data(user_uploaded_data, holidays_df)

    
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

    
    metrics = None
    min_series_length = ts_upload.index.get_level_values('item_id').value_counts().min()
    
    if min_series_length > predictor.prediction_length:
        print("  -> Data is long enough. Evaluating model performance...")
        metrics = predictor.evaluate(ts_upload, display=False)
    else:
        print(f"  -> Data is too short to evaluate. Skipping evaluation.")
        reason = f"Uploaded data history ({min_series_length} points) is not longer than the model's prediction length ({predictor.prediction_length} points)."
        metrics = pd.DataFrame([{"info": "Evaluation skipped", "reason": reason}])
    future_known_covariates = generate_future_covariates(predictor, ts_upload, holidays_df)
    predictions = predictor.predict(
        ts_upload,
        known_covariates=future_known_covariates
    )

    print("✅ Forecasts and metrics generated successfully!")
    return predictions, metrics
