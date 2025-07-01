import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor
from src.feature_engineering import create_jewelry_features

def train_predictor(ts_data, config):
    """
    Trains the TimeSeriesPredictor with the given data and configuration.
    """
    try:
        print("\nTraining forecasting models with enhanced configuration...")
        predictor = TimeSeriesPredictor(
            prediction_length=config.PREDICTION_LENGTH,
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
        print(f"Error during model training: {e}")
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
        print(f"Error during model evaluation: {e}")
        raise

def make_predictions(predictor, ts_data, holidays_df):
    """
    Generates future predictions using the trained model.
    """
    try:
        print("\nMaking predictions for the future...")

        # Create the future dataframe structure
        future_known_covariates = predictor.make_future_data_frame(data=ts_data)
        future_known_covariates.reset_index(inplace=True)

        # --- Engineer features for the future ---
        # 1. Seasonal and holiday features
        future_known_covariates = create_jewelry_features(future_known_covariates)
        future_known_covariates = pd.merge(
            future_known_covariates,
            holidays_df[['timestamp', 'is_holiday']],
            on="timestamp",
            how="left"
        )

        # 2. Assumed future values (can be replaced with actuals if known)
        assumed_values = {
            'was_stocked_out': 0,
            'is_on_promotion': 0,
            'is_high_discount': 0,
            'gold_price_change': 0,
            'gold_price_ma_7': 0
        }
        for col, value in assumed_values.items():
            future_known_covariates[col] = value

        # 3. Use last known values for rolling features
        rolling_features = ['sales_ma_7', 'sales_ma_14', 'sales_ma_30', 'stockout_days_last_7', 'potential_lost_sales']
        for feature in rolling_features:
            last_values = ts_data.groupby('item_id')[feature].last()
            future_known_covariates = pd.merge(
                future_known_covariates,
                last_values.rename(feature),
                on='item_id',
                how='left'
            )

        # Fill any remaining NaNs
        future_known_covariates.fillna(0, inplace=True)

        # Generate predictions
        predictions = predictor.predict(ts_data, known_covariates=future_known_covariates)
        print("Forecasts generated successfully.")
        return predictions
    except Exception as e:
        print(f"Error during prediction generation: {e}")
        raise

