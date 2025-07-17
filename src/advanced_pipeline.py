import os
import sys
import logging
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from itertools import product
from xgboost import XGBRegressor
from keras_tuner import RandomSearch
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import (
    LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten,
    GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization
)
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm
from statsmodels.tsa.stattools import acf
import concurrent.futures
import threading
import math
import re
import random
import warnings
import joblib
import optuna
import shap
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax
import time


from src import config_advanced as CONFIG
from src import data_loader


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


os.makedirs(os.path.dirname(CONFIG.LOG_FILE_PATH), exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
file_handler = logging.FileHandler(CONFIG.LOG_FILE_PATH, mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(console_handler)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)



def verify_gpu():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"Detected {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            return True
        else:
            logging.info("No GPUs detected, using CPU.")
            return False
    except Exception as e:
        logging.error(f"Error verifying GPU: {str(e)}")
        return False


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

def tune_transformer_model(X_train, y_train, features, n_steps, model_dir, sku_id, keras_lock):
    def build_model(hp):
        inputs = Input(shape=(n_steps, len(features)))
        x = inputs
        for _ in range(hp.Int('num_transformer_blocks', 1, 3, step=1)):
            x = transformer_encoder(
                x,
                head_size=hp.Int('head_size', 64, 256, step=64),
                num_heads=hp.Int('num_heads', 2, 4, step=1),
                ff_dim=hp.Int('ff_dim', 64, 256, step=64),
                dropout=hp.Float('dropout', 0.1, 0.4, step=0.1)
            )
        x = GlobalAveragePooling1D(data_format="channels_last")(x)
        x = Dense(hp.Int('dense_units', 20, 100, step=20), activation="relu")(x)
        x = Dropout(hp.Float('dropout_final', 0.1, 0.4, step=0.1))(x)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log')), loss='mse')
        return model

    logging.info(f"SKU {sku_id}: Starting Transformer model hyperparameter tuning ({CONFIG.TRANSFORMER_CONFIG['max_trials']} trials)...")
    tuner = RandomSearch(build_model, objective='val_loss', max_trials=CONFIG.TRANSFORMER_CONFIG['max_trials'], executions_per_trial=1, directory=model_dir, project_name=f'transformer_tune_{sku_id}', overwrite=True)
    best_model = None
    try:
        with keras_lock:
            tuner.search(X_train, y_train, epochs=CONFIG.TRANSFORMER_CONFIG['epochs'], batch_size=CONFIG.TRANSFORMER_CONFIG['batch_size'], validation_split=0.2, verbose=1, callbacks=[EarlyStopping('val_loss', patience=5)])
            best_models = tuner.get_best_models(num_models=1)
            if best_models:
                best_model = best_models[0]
                best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
                logging.info(f"SKU {sku_id}: Transformer tuning complete. Best hyperparameters found: {best_hp.values}")
            else:
                logging.warning(f"SKU {sku_id}: Transformer tuning found no best models.")
    except Exception as e:
        logging.error(f"SKU {sku_id}: Error during Transformer tuning: {e}")
        best_model = None
    return best_model

def tune_cnn_lstm_model(X_train, y_train, features, n_steps, model_dir, sku_id, keras_lock):
    def build_model(hp):
        model = Sequential([
            Input(shape=(n_steps, len(features))),
            Conv1D(filters=hp.Int('filters', 32, 128, step=32), kernel_size=hp.Int('kernel_size', 3, 7, step=2), activation='relu', padding='same'),
            MaxPooling1D(pool_size=2, padding='same'),
            LSTM(units=hp.Int('units_1', 50, 150, step=50), activation='tanh', return_sequences=True),
            LSTM(units=hp.Int('units_2', 50, 150, step=50), activation='tanh'),
            Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log')), loss='mse')
        return model

    logging.info(f"SKU {sku_id}: Starting CNN_LSTM model hyperparameter tuning ({CONFIG.CNN_LSTM_CONFIG['max_trials']} trials)...")
    tuner = RandomSearch(build_model, objective='val_loss', max_trials=CONFIG.CNN_LSTM_CONFIG['max_trials'], executions_per_trial=2, directory=model_dir, project_name=f'cnn_lstm_tune_{sku_id}', overwrite=True)
    best_model = None
    try:
        with keras_lock:
            tuner.search(X_train, y_train, epochs=CONFIG.CNN_LSTM_CONFIG['epochs'], batch_size=CONFIG.CNN_LSTM_CONFIG['batch_size'], validation_split=0.2, verbose=1, callbacks=[EarlyStopping('val_loss', patience=5)])
            best_models = tuner.get_best_models(num_models=1)
            if best_models:
                best_model = best_models[0]
                best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
                logging.info(f"SKU {sku_id}: CNN_LSTM tuning complete. Best hyperparameters found: {best_hp.values}")
            else:
                logging.warning(f"SKU {sku_id}: CNN_LSTM tuning found no best models.")
    except Exception as e:
        logging.error(f"SKU {sku_id}: Error during CNN_LSTM tuning: {e}")
        best_model = None
    return best_model

def tune_lstm_model(X_train, y_train, lstm_features, n_steps, model_dir, sku_id, keras_lock):
    def build_model(hp):
        model = Sequential([
            Input(shape=(n_steps, len(lstm_features))),
            LSTM(units=hp.Int('units_1', 50, 200, step=50), activation='tanh', return_sequences=True),
            LSTM(units=hp.Int('units_2', 50, 200, step=50), activation='tanh'),
            Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log')), loss='mse')
        return model

    logging.info(f"SKU {sku_id}: Starting LSTM model hyperparameter tuning ({CONFIG.LSTM_CONFIG['max_trials']} trials)...")
    tuner = RandomSearch(build_model, objective='val_loss', max_trials=CONFIG.LSTM_CONFIG['max_trials'], executions_per_trial=2, directory=model_dir, project_name=f'lstm_tune_{sku_id}', overwrite=True)
    best_model = None
    try:
        with keras_lock:
            tuner.search(X_train, y_train, epochs=CONFIG.LSTM_CONFIG['epochs'], batch_size=CONFIG.LSTM_CONFIG['batch_size'], validation_split=0.2, verbose=1, callbacks=[EarlyStopping('val_loss', patience=5)])
            best_models = tuner.get_best_models(num_models=1)
            if best_models:
                best_model = best_models[0]
                best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
                logging.info(f"SKU {sku_id}: LSTM tuning complete. Best hyperparameters found: {best_hp.values}")
            else:
                logging.warning(f"SKU {sku_id}: LSTM tuning found no best models.")
    except Exception as e:
        logging.error(f"SKU {sku_id}: Error during LSTM tuning: {e}")
        best_model = None
    return best_model


def clean_numeric(value):
    if pd.isna(value): return np.nan
    if isinstance(value, (int, float)): return float(value)
    value = str(value).strip()
    value = re.sub(r'[^\d\.-]', '', value)
    try: return float(value)
    except (ValueError, TypeError): return np.nan

def create_multivariate_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps, 0])
    return np.array(X), np.array(y)

def predict_with_mc_dropout(model, X, n_samples=50):
    preds = np.array([model(X, training=True).numpy() for _ in range(n_samples)])
    return preds.squeeze()

def evaluate_forecast(actual, predicted):
    actual, predicted = np.array(actual).flatten(), np.array(predicted).flatten()
    metrics = {'rmse': np.sqrt(mean_squared_error(actual, predicted)), 'mae': mean_absolute_error(actual, predicted), 'r2': r2_score(actual, predicted)}
    non_zero_mask = actual != 0
    metrics['mape'] = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100 if np.any(non_zero_mask) else 0.0
    metrics['wape'] = np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100 if np.sum(np.abs(actual)) > 0 else 0.0
    metrics['smape'] = 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual) + 1e-10)) if len(actual) > 0 else 0.0
    metrics['accuracy'] = 100 - metrics['wape']
    return metrics

def calculate_inventory_metrics(forecast_df, product_price, demand_std, config_obj):
    inv_conf = config_obj.INVENTORY_CONFIG
    avg_fcst = forecast_df['yhat'].mean()
    avg_lead_time, moq = inv_conf['avg_lead_time_days'], inv_conf['moq']
    lead_time_demand = avg_fcst * avg_lead_time
    z_score = norm.ppf(inv_conf['service_level'])
    safety_stock = z_score * demand_std * np.sqrt(avg_lead_time)
    reorder_point = lead_time_demand + safety_stock
    H = inv_conf['holding_cost_percentage'] * product_price if product_price and not pd.isna(product_price) else 0
    S = inv_conf['ordering_cost_percentage'] * product_price if product_price and not pd.isna(product_price) else 0
    D_annual = avg_fcst * 365
    eoq = math.sqrt((2 * D_annual * S) / H) if H > 0 else 0
    validated_eoq = max(eoq, moq)
    return {'Avg Lead Time': avg_lead_time, 'Lead Time Demand': f"{lead_time_demand:,.0f}", 'Safety Stock': f"{safety_stock:,.0f}", 'Reorder Point': f"{reorder_point:,.0f}", 'Validated EOQ': f"{validated_eoq:,.0f}", 'Holding Cost': f"{H * D_annual:,.2f}", 'Ordering Cost': f"{S:,.2f}"}

def is_sporadic(series, threshold=0.4):
    if len(series) == 0: return False
    return (series == 0).mean() > threshold

def tune_prophet_params(train_df, param_grid):
    all_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    rmses = []
    tuning_df = train_df.tail(365)
    for params in all_params:
        try:
            m = Prophet(**params).fit(tuning_df)
            df_cv = cross_validation(m, initial='180 days', period='30 days', horizon='30 days', parallel="threads", disable_diagnostics=True)
            df_p = performance_metrics(df_cv, rolling_window=1)
            if df_p is not None and not df_p.empty and 'rmse' in df_p.columns:
                rmses.append(df_p['rmse'].values[0])
            else:
                rmses.append(float('inf'))
        except Exception:
            rmses.append(float('inf'))
    if not rmses or all(r == float('inf') for r in rmses): return {}
    return all_params[np.argmin(rmses)]

def tune_xgb_params(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {'n_estimators': trial.suggest_int('n_estimators', 50, 200), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True), 'max_depth': trial.suggest_int('max_depth', 3, 10), 'min_child_weight': trial.suggest_int('min_child_weight', 1, 10), 'subsample': trial.suggest_float('subsample', 0.5, 1.0), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), 'random_state': 42}
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
        y_pred = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, y_pred))
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20, n_jobs=1)
    return study.best_params

def run_sma_forecast(model_df, config_obj):
    sma_window = config_obj.SMA_WINDOW
    test_pred = model_df['y'].shift(1).rolling(window=sma_window).mean().fillna(0)
    performance_metrics = evaluate_forecast(model_df['y'], test_pred)
    last_known_avg = model_df['y'].tail(sma_window).mean()
    future_dates = pd.date_range(start=model_df['ds'].max() + pd.Timedelta(days=1), periods=183)
    final_forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': last_known_avg, 'yhat_lower': last_known_avg, 'yhat_upper': last_known_avg})
    return final_forecast_df, performance_metrics

def generate_dl_forecast(model, scaler, hist_data, future_steps, n_steps, dl_features, store_growth_df):
    history_df = hist_data.copy()
    mc_simulations = []
    for i in range(future_steps):
        input_features_scaled = scaler.transform(history_df[dl_features].tail(n_steps))
        input_seq = input_features_scaled.reshape(1, n_steps, len(dl_features))
        mc_preds_scaled = predict_with_mc_dropout(model, input_seq, CONFIG.MC_DROPOUT_SAMPLES)
        mc_simulations.append(mc_preds_scaled)
        next_y_pred_scaled = np.median(mc_preds_scaled)
        dummy_row = np.zeros((1, len(dl_features)))
        dummy_row[0, 0] = next_y_pred_scaled
        next_y_pred = scaler.inverse_transform(dummy_row)[0, 0]
        next_date = history_df['ds'].iloc[-1] + pd.Timedelta(days=1)
        next_row = pd.DataFrame({'ds': [next_date], 'y': [next_y_pred]})
        next_row = next_row.merge(store_growth_df[['created_at', 'store_count']], left_on='ds', right_on='created_at', how='left').drop(columns='created_at')
        for col in ['was_stocked_out', 'disc', 'revenue', 'is_on_promotion', 'discount_percentage']:
            next_row[col] = 0
        next_row['is_month_start'] = next_row['ds'].dt.is_month_start.astype(int)
        next_row['is_month_end'] = next_row['ds'].dt.is_month_end.astype(int)
        temp_history = pd.concat([history_df, next_row], ignore_index=True)
        temp_history.sort_values('ds', inplace=True)
        for lag in [1, 7, 14, 30]: temp_history[f'sales_lag_{lag}d'] = temp_history['y'].shift(lag)
        temp_history['rolling_avg_sales_7d'] = temp_history['y'].shift(1).rolling(window=7, min_periods=1).mean()
        temp_history['rolling_std_sales_7d'] = temp_history['y'].shift(1).rolling(window=7, min_periods=1).std()
        temp_history['day_of_week_sin'] = np.sin(2 * np.pi * temp_history['ds'].dt.dayofweek / 7)
        temp_history['day_of_week_cos'] = np.cos(2 * np.pi * temp_history['ds'].dt.dayofweek / 7)
        temp_history['month_sin'] = np.sin(2 * np.pi * temp_history['ds'].dt.month / 12)
        temp_history['month_cos'] = np.cos(2 * np.pi * temp_history['ds'].dt.month / 12)
        history_df = temp_history.fillna(0)
    mc_simulations = np.array(mc_simulations).squeeze().T
    if mc_simulations.ndim == 1:
        mc_simulations = mc_simulations.reshape(CONFIG.MC_DROPOUT_SAMPLES, -1)
    rescaled_sims = np.zeros_like(mc_simulations)
    for i in range(mc_simulations.shape[0]):
        dummy_row = np.zeros((mc_simulations.shape[1], len(dl_features)))
        dummy_row[:, 0] = mc_simulations[i, :]
        rescaled_sims[i, :] = scaler.inverse_transform(dummy_row)[:, 0]
    yhat = np.median(rescaled_sims, axis=0)
    yhat_lower = np.percentile(rescaled_sims, CONFIG.QUANTILES[0] * 100, axis=0)
    yhat_upper = np.percentile(rescaled_sims, CONFIG.QUANTILES[2] * 100, axis=0)
    return yhat, yhat_lower, yhat_upper


def _prepare_sku_data(sku_df, store_growth_df):
    sku_df = sku_df.merge(store_growth_df, on='created_at', how='left')
    sku_df['store_count'] = sku_df['store_count'].ffill().bfill()
    agg_rules = {'qty': 'sum', 'store_count': 'first', 'was_stocked_out': 'max', 'disc': 'sum', 'revenue': 'sum'}
    model_df = sku_df.groupby('created_at').agg(agg_rules).reset_index()
    model_df.rename(columns={'created_at': 'ds', 'qty': 'y'}, inplace=True)
    q1, q3 = model_df['y'].quantile([0.25, 0.75])
    iqr = q3 - q1
    model_df['y'] = model_df['y'].clip(lower=0, upper=q3 + 1.5 * iqr)
    full_date_range = pd.date_range(start=model_df['ds'].min(), end=model_df['ds'].max(), freq='D')
    model_df = model_df.set_index('ds').reindex(full_date_range).reset_index().rename(columns={'index': 'ds'})
    model_df['y'] = model_df['y'].interpolate(method='linear').fillna(0)
    for col in ['store_count', 'was_stocked_out', 'disc', 'revenue']:
        model_df[col] = model_df[col].fillna(0)
    return model_df

def _calculate_sku_attributes(model_df, sku_id, sku_to_cluster_map):
    sku_results = {'sku_id': sku_id, 'historical_avg_sales': model_df['y'].mean(), 'historical_std_sales': model_df['y'].std(), 'historical_max_sales': model_df['y'].max(), 'demand_cluster_id': sku_to_cluster_map.get(sku_id, -1)}
    sku_results['coefficient_of_variation'] = sku_results['historical_std_sales'] / sku_results['historical_avg_sales'] if sku_results['historical_avg_sales'] > 0 else 0
    sku_results['zero_sales_ratio'] = (model_df['y'] == 0).mean()
    try:
        sku_results['acf_lag_7'] = acf(model_df['y'], nlags=7, fft=True)[-1]
    except Exception:
        sku_results['acf_lag_7'] = None
    return sku_results

def _engineer_time_series_features(model_df):
    model_df['discount_percentage'] = (model_df['disc'] / (model_df['revenue'] + model_df['disc'])).fillna(0) if (model_df['revenue'] + model_df['disc']).sum() > 0 else 0
    model_df['is_on_promotion'] = (model_df['disc'] > 0).astype(int)
    model_df['is_month_start'] = model_df['ds'].dt.is_month_start.astype(int)
    model_df['is_month_end'] = model_df['ds'].dt.is_month_end.astype(int)
    for lag in [1, 7, 14, 30]:
        model_df[f'sales_lag_{lag}d'] = model_df['y'].shift(lag).fillna(0)
    model_df['rolling_avg_sales_7d'] = model_df['y'].shift(1).rolling(window=7, min_periods=1).mean().fillna(0)
    model_df['rolling_std_sales_7d'] = model_df['y'].shift(1).rolling(window=7, min_periods=1).std().fillna(0)
    model_df['day_of_week_sin'] = np.sin(2 * np.pi * model_df['ds'].dt.dayofweek / 7)
    model_df['day_of_week_cos'] = np.cos(2 * np.pi * model_df['ds'].dt.dayofweek / 7)
    model_df['month_sin'] = np.sin(2 * np.pi * model_df['ds'].dt.month / 12)
    model_df['month_cos'] = np.cos(2 * np.pi * model_df['ds'].dt.month / 12)
    model_df.sort_values('ds', inplace=True)
    return model_df

def _evaluate_prophet_model(train_df, test_df, holidays_df, sku_id):
    logging.info(f"SKU {sku_id}: Evaluating Prophet...")
    best_prophet_params = tune_prophet_params(train_df, CONFIG.PROPHET_TUNING_PARAMS)
    prophet_regressors = ['store_count', 'was_stocked_out', 'is_on_promotion', 'discount_percentage', 'sales_lag_7d', 'rolling_avg_sales_7d']
    prophet_eval_model = Prophet(holidays=holidays_df, **best_prophet_params)
    prophet_eval_model.add_seasonality(name='yearly', period=365.25, fourier_order=10)
    prophet_eval_model.add_seasonality(name='daily', period=1, fourier_order=3)
    prophet_eval_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    for r in prophet_regressors:
        if r in train_df.columns:
            prophet_eval_model.add_regressor(r)
    prophet_eval_model.fit(train_df)
    prophet_test_forecast = prophet_eval_model.predict(test_df.drop(columns='y', errors='ignore'))
    prophet_metrics = evaluate_forecast(test_df['y'], prophet_test_forecast['yhat'])
    return prophet_metrics, prophet_test_forecast, best_prophet_params, prophet_regressors

def _evaluate_deep_learning_models(train_df, test_df, dl_features, n_steps, model_dir, sku_id, keras_lock, store_growth_df):
    lstm_metrics, cnn_lstm_metrics, transformer_metrics = None, None, None
    lstm_yhat, cnn_lstm_yhat, transformer_yhat = None, None, None
    scaler_eval = RobustScaler()
    scaled_train_data = scaler_eval.fit_transform(train_df[dl_features])
    X_train_eval, y_train_eval = create_multivariate_sequences(scaled_train_data, n_steps)
    if X_train_eval.shape[0] > 0:
        for model_type in ['LSTM', 'CNN_LSTM', 'Transformer']:
            try:
                logging.info(f"SKU {sku_id}: Evaluating {model_type}...")
                tune_func = {'LSTM': tune_lstm_model, 'CNN_LSTM': tune_cnn_lstm_model, 'Transformer': tune_transformer_model}[model_type]
                eval_model = tune_func(X_train_eval, y_train_eval, dl_features, n_steps, model_dir, sku_id, keras_lock)
                if eval_model:
                    yhat, _, _ = generate_dl_forecast(eval_model, scaler_eval, train_df, len(test_df), n_steps, dl_features, store_growth_df)
                    if model_type == 'LSTM':
                        lstm_yhat, lstm_metrics = yhat, evaluate_forecast(test_df['y'], yhat)
                    elif model_type == 'CNN_LSTM':
                        cnn_lstm_yhat, cnn_lstm_metrics = yhat, evaluate_forecast(test_df['y'], yhat)
                    elif model_type == 'Transformer':
                        transformer_yhat, transformer_metrics = yhat, evaluate_forecast(test_df['y'], yhat)
                else:
                    logging.warning(f"SKU {sku_id}: {model_type} model tuning returned no valid model.")
            except Exception as e:
                logging.warning(f"SKU {sku_id}: {model_type} evaluation failed: {e}.")
    return lstm_metrics, cnn_lstm_metrics, transformer_metrics, lstm_yhat, cnn_lstm_yhat, transformer_yhat

def _evaluate_xgboost_model(train_df, test_df, xgb_features, sku_id):
    logging.info(f"SKU {sku_id}: Evaluating XGBoost...")
    xgb_metrics, xgb_yhat, best_xgb_params = None, None, None
    try:
        train_size = int(0.8 * len(train_df))
        X_train_xgb, X_val_xgb = train_df[xgb_features].iloc[:train_size], train_df[xgb_features].iloc[train_size:]
        y_train_xgb, y_val_xgb = train_df['y'].iloc[:train_size], train_df['y'].iloc[train_size:]
        best_xgb_params = tune_xgb_params(X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb)
        xgb_model = XGBRegressor(**best_xgb_params, objective='reg:squarederror')
        xgb_model.fit(train_df[xgb_features], train_df['y'])
        xgb_yhat = xgb_model.predict(test_df[xgb_features])
        xgb_metrics = evaluate_forecast(test_df['y'], xgb_yhat)
    except Exception as e:
        logging.warning(f"SKU {sku_id}: XGBoost evaluation failed: {e}.")
    return xgb_metrics, xgb_yhat, best_xgb_params

def _select_and_forecast_champion_model(sku_id, models, test_df, prophet_test_forecast, xgb_yhat, lstm_yhat, cnn_lstm_yhat, transformer_yhat):
    valid_models = {k: v for k, v in models.items() if v is not None and not np.isnan(v.get('wape', np.nan))}
    if not valid_models:
        raise ValueError("No models produced valid evaluation metrics.")
    scores = {k: v['wape'] for k, v in valid_models.items()}
    final_model_type = None
    champion_metrics = None
    weights = {}
    if len(valid_models) > 1 and all(v.get('wape', 101) < CONFIG.ENSEMBLE_WAPE_THRESHOLD for v in valid_models.values()):
        final_model_type = 'Ensemble'
        weights = {k: 1 / (s + 1e-10) for k, s in scores.items()}
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        ensemble_yhat = np.zeros(len(test_df))
        if 'Tuned_Prophet' in valid_models and prophet_test_forecast is not None:
            ensemble_yhat += weights.get('Tuned_Prophet', 0) * prophet_test_forecast['yhat'].values
        if 'XGBoost' in valid_models and xgb_yhat is not None:
            ensemble_yhat += weights.get('XGBoost', 0) * xgb_yhat
        if 'LSTM' in valid_models and lstm_yhat is not None:
            ensemble_yhat += weights.get('LSTM', 0) * lstm_yhat
        if 'CNN_LSTM' in valid_models and cnn_lstm_yhat is not None:
            ensemble_yhat += weights.get('CNN_LSTM', 0) * cnn_lstm_yhat
        if 'Transformer' in valid_models and transformer_yhat is not None:
            ensemble_yhat += weights.get('Transformer', 0) * transformer_yhat
        champion_metrics = evaluate_forecast(test_df['y'], ensemble_yhat)
        logging.info(f"SKU {sku_id}: Ensemble selected. Weights: { {k: round(v, 3) for k, v in weights.items()} }")
    else:
        final_model_type = min(scores, key=lambda k: scores[k])
        champion_metrics = valid_models[final_model_type]
        weights = {final_model_type: 1.0}
    logging.info(f"SKU {sku_id}: Champion model is {final_model_type} with WAPE: {champion_metrics.get('wape', 'N/A'):.2f}%")
    return final_model_type, champion_metrics, weights

def _generate_future_forecasts(model_df, holidays_df, store_growth_df, final_model_type, weights, best_prophet_params, prophet_regressors, best_xgb_params, dl_features, n_steps, model_dir, sku_id, keras_lock, pre_trained_model=None):
    future_df_base = pd.DataFrame({'ds': pd.date_range(start=model_df['ds'].max() + pd.Timedelta(days=1), periods=183)})
    future_predictions, future_lower_predictions, future_upper_predictions = {}, {}, {}
    final_model_to_save = {}
    if 'Category_Transfer' in final_model_type:
        model_list_for_final_run = [final_model_type.replace('Category_Transfer_', '')]
    else:
        model_list_for_final_run = [final_model_type] if final_model_type != 'Ensemble' else list(weights.keys())
    for model_name in model_list_for_final_run:
        if model_name == 'Tuned_Prophet':
            if pre_trained_model:
                logging.info(f"SKU {sku_id}: Using pre-trained Prophet model for forecasting.")
                final_prophet_model = pre_trained_model
                final_model_to_save['prophet'] = None
            else:
                logging.info(f"SKU {sku_id}: Training a new Prophet model.")
                final_prophet_model = Prophet(holidays=holidays_df, **best_prophet_params)
                final_prophet_model.add_seasonality(name='yearly', period=365.25, fourier_order=10)
                final_prophet_model.add_seasonality(name='daily', period=1, fourier_order=3)
                for r in prophet_regressors:
                    final_prophet_model.add_regressor(r)
                final_prophet_model.fit(model_df)
                final_model_to_save['prophet'] = final_prophet_model
            future_df_prophet = future_df_base.copy()
            future_df_prophet = future_df_prophet.merge(store_growth_df[['created_at', 'store_count']], left_on='ds', right_on='created_at', how='left').drop(columns='created_at').ffill()
            future_df_prophet['was_stocked_out'] = 0
            future_df_prophet['is_on_promotion'] = 0
            future_df_prophet['discount_percentage'] = 0
            future_df_prophet['sales_lag_7d'] = model_df['y'].iloc[-7] if len(model_df) >= 7 else 0
            future_df_prophet['rolling_avg_sales_7d'] = model_df['y'].tail(7).mean() if len(model_df) >= 7 else model_df['y'].mean()
            prophet_future_forecast = final_prophet_model.predict(future_df_prophet)
            future_predictions['Tuned_Prophet'] = prophet_future_forecast['yhat'].values
            future_lower_predictions['Tuned_Prophet'] = prophet_future_forecast['yhat_lower'].values
            future_upper_predictions['Tuned_Prophet'] = prophet_future_forecast['yhat_upper'].values
        elif model_name == 'XGBoost':
            xgb_features_for_forecast = [col for col in dl_features if col != 'y']
            future_df_xgb = future_df_base.copy()
            future_df_xgb = future_df_xgb.merge(store_growth_df[['created_at', 'store_count']], left_on='ds', right_on='created_at', how='left').drop(columns='created_at').ffill()
            for col in ['was_stocked_out', 'is_on_promotion', 'discount_percentage', 'is_month_start', 'is_month_end']:
                future_df_xgb[col] = 0
            future_df_xgb['day_of_week_sin'] = np.sin(2 * np.pi * future_df_xgb['ds'].dt.dayofweek / 7)
            future_df_xgb['day_of_week_cos'] = np.cos(2 * np.pi * future_df_xgb['ds'].dt.dayofweek / 7)
            future_df_xgb['month_sin'] = np.sin(2 * np.pi * future_df_xgb['ds'].dt.month / 12)
            future_df_xgb['month_cos'] = np.cos(2 * np.pi * future_df_xgb['ds'].dt.month / 12)
            temp_hist_df = model_df.copy()
            future_lag_features = []
            for i in range(len(future_df_base)):
                new_row = future_df_xgb.iloc[[i]]
                new_row['sales_lag_1d'] = temp_hist_df['y'].iloc[-1]
                new_row['sales_lag_7d'] = temp_hist_df['y'].iloc[-7] if len(temp_hist_df) >= 7 else 0
                new_row['sales_lag_14d'] = temp_hist_df['y'].iloc[-14] if len(temp_hist_df) >= 14 else 0
                new_row['sales_lag_30d'] = temp_hist_df['y'].iloc[-30] if len(temp_hist_df) >= 30 else 0
                new_row['rolling_avg_sales_7d'] = temp_hist_df['y'].tail(7).mean()
                new_row['rolling_std_sales_7d'] = temp_hist_df['y'].tail(7).std() if not np.isnan(temp_hist_df['y'].tail(7).std()) else 0
                if pre_trained_model:
                    xgb_model_for_pred = pre_trained_model.get(0.5)
                else:
                    xgb_model_for_pred = XGBRegressor(**best_xgb_params, objective='reg:squarederror')
                    xgb_model_for_pred.fit(model_df[xgb_features_for_forecast], model_df['y'])
                y_pred = xgb_model_for_pred.predict(new_row[xgb_features_for_forecast])[0]
                future_lag_features.append(new_row)
                pred_df_row = pd.DataFrame({'ds': [new_row['ds'].iloc[0]], 'y': [y_pred]})
                temp_hist_df = pd.concat([temp_hist_df, pred_df_row], ignore_index=True)
            future_df_xgb_final = pd.concat(future_lag_features, ignore_index=True)
            xgb_future_preds = {}
            if pre_trained_model:
                logging.info(f"SKU {sku_id}: Using pre-trained XGBoost models for forecasting.")
                final_xgb_models = pre_trained_model
                final_model_to_save['xgb'] = None
            else:
                logging.info(f"SKU {sku_id}: Training new XGBoost models.")
                final_xgb_models = {}
                for q in CONFIG.QUANTILES:
                    model = XGBRegressor(**best_xgb_params, objective='reg:squarederror', quantile_alpha=q)
                    model.fit(model_df[xgb_features_for_forecast], model_df['y'])
                    final_xgb_models[q] = model
                final_model_to_save['xgb'] = final_xgb_models
            for q, model in final_xgb_models.items():
                xgb_future_preds[q] = model.predict(future_df_xgb_final[xgb_features_for_forecast])
            future_predictions['XGBoost'] = xgb_future_preds[CONFIG.QUANTILES[1]]
            future_lower_predictions['XGBoost'] = xgb_future_preds[CONFIG.QUANTILES[0]]
            future_upper_predictions['XGBoost'] = xgb_future_preds[CONFIG.QUANTILES[2]]
        elif model_name in ['LSTM', 'CNN_LSTM', 'Transformer']:
            final_dl_model = None
            scaler_for_forecast = None
            if pre_trained_model:
                logging.info(f"SKU {sku_id}: Using pre-trained DL model ({model_name}) for forecasting.")
                final_dl_model, scaler_for_forecast = pre_trained_model
                final_model_to_save[model_name.lower()] = None
            else:
                logging.info(f"SKU {sku_id}: Tuning a new DL model ({model_name}).")
                scaler_final = RobustScaler()
                scaled_data = scaler_final.fit_transform(model_df[dl_features])
                X_train_full, y_train_full = create_multivariate_sequences(scaled_data, n_steps)
                if X_train_full.shape[0] > 0:
                    tune_func = {'LSTM': tune_lstm_model, 'CNN_LSTM': tune_cnn_lstm_model, 'Transformer': tune_transformer_model}[model_name]
                    final_dl_model = tune_func(X_train_full, y_train_full, dl_features, n_steps, model_dir, sku_id, keras_lock)
                    scaler_for_forecast = scaler_final
                    final_model_to_save[model_name.lower()] = final_dl_model
                    final_model_to_save['scaler'] = scaler_for_forecast
                else:
                    final_dl_model = None
            if final_dl_model:
                yhat, yhat_lower, yhat_upper = generate_dl_forecast(final_dl_model, scaler_for_forecast, model_df, 183, n_steps, dl_features, store_growth_df)
                future_predictions[model_name] = yhat
                future_lower_predictions[model_name] = yhat_lower
                future_upper_predictions[model_name] = yhat_upper
            else:
                logging.warning(f"SKU {sku_id}: DL model '{model_name}' was not available or failed to train. Skipping future forecast generation.")
    if 'Category_Transfer' in final_model_type:
        model_name = final_model_type.replace('Category_Transfer_', '')
        if model_name in future_predictions:
            final_forecast_df = pd.DataFrame({'ds': future_df_base['ds'], 'yhat': future_predictions[model_name], 'yhat_lower': future_lower_predictions[model_name], 'yhat_upper': future_upper_predictions[model_name]})
        else:
            final_forecast_df = pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper'])
    elif final_model_type == 'Ensemble':
        ensemble_yhat = np.zeros(183)
        ensemble_yhat_lower = np.zeros(183)
        ensemble_yhat_upper = np.zeros(183)
        for model, weight in weights.items():
            if model in future_predictions:
                ensemble_yhat += weight * future_predictions.get(model, 0)
                ensemble_yhat_lower += weight * future_lower_predictions.get(model, 0)
                ensemble_yhat_upper += weight * future_upper_predictions.get(model, 0)
        final_forecast_df = pd.DataFrame({'ds': future_df_base['ds'], 'yhat': ensemble_yhat, 'yhat_lower': ensemble_yhat_lower, 'yhat_upper': ensemble_yhat_upper})
    else:
        if final_model_type in future_predictions:
            final_forecast_df = pd.DataFrame({'ds': future_df_base['ds'], 'yhat': future_predictions[final_model_type], 'yhat_lower': future_lower_predictions[final_model_type], 'yhat_upper': future_upper_predictions[final_model_type]})
        else:
            final_forecast_df = pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper'])
            logging.warning(f"SKU {sku_id}: Champion model '{final_model_type}' did not generate future predictions. Returning empty forecast.")
    return final_forecast_df, final_model_to_save

def _calculate_and_store_shap_values(final_model_type, final_model_to_save, model_df, xgb_features, sku_results):
    if final_model_type == 'XGBoost' or ('Ensemble' in final_model_type and 'xgb' in final_model_to_save):
        if 'xgb' in final_model_to_save and 0.5 in final_model_to_save['xgb']:
            xgb_champion_model = final_model_to_save['xgb'][0.5]
            explainer = shap.TreeExplainer(xgb_champion_model)
            shap_values = explainer.shap_values(model_df[xgb_features])
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_instance_values = pd.DataFrame(shap_values, columns=xgb_features)
            mean_shap_values = shap_instance_values.mean(axis=0).sort_values()
            sku_results['top_5_positive_features'] = str(list(mean_shap_values.tail(5).index))
            sku_results['top_5_negative_features'] = str(list(mean_shap_values.head(5).index))
            feature_imp = pd.DataFrame({'feature': xgb_champion_model.feature_names_in_, 'importance': xgb_champion_model.feature_importances_}).sort_values('importance', ascending=False).head(5)
            sku_results['top_features'] = list(feature_imp['feature'])
            sku_results['top_importances'] = list(feature_imp['importance'])
        else:
            logging.warning("XGBoost model not found in final_model_to_save for SHAP calculation.")
            sku_results.update({'top_features': None, 'top_importances': None, 'top_5_positive_features': None, 'top_5_negative_features': None})
    else:
        sku_results.update({'top_features': None, 'top_importances': None, 'top_5_positive_features': None, 'top_5_negative_features': None})
    return sku_results

def process_sku(args):
    """
    Processes a single SKU for demand forecasting, returning results instead of writing to file.
    """
    (sku_id, sku_df, holidays_df, store_growth_df, model_dir, keras_lock, sku_number, total_skus, sku_to_cluster_map, category_models) = args
    logging.info(f"\n--- 🔄 Processing SKU {sku_number} of {total_skus}: {sku_id} ---")
    sku_results = {'sku_id': sku_id}
    try:
        model_df = _prepare_sku_data(sku_df, store_growth_df)
        sku_results.update(_calculate_sku_attributes(model_df, sku_id, sku_to_cluster_map))
        if len(model_df) < CONFIG.MIN_DATA_POINTS_DEEP_LEARNING or is_sporadic(model_df['y'], CONFIG.SPORADIC_THRESHOLD):
            logging.info(f"SKU {sku_id}: Insufficient or sporadic data. Attempting to use a pre-trained category model.")
            sku_cat = _get_sku_category(sku_id)
            if sku_cat and sku_cat in category_models:
                model_type, model_object = category_models[sku_cat]
                logging.info(f"SKU {sku_id}: Found champion model '{model_type}' for category '{sku_cat}'. Applying transfer learning.")
                final_forecast_df, _ = _generate_future_forecasts(
                    model_df, holidays_df, store_growth_df, model_type, {model_type: 1.0},
                    best_prophet_params={}, prophet_regressors=[], best_xgb_params={},
                    dl_features=[], n_steps=CONFIG.LSTM_CONFIG['n_steps'], model_dir=model_dir, sku_id=sku_id, keras_lock=keras_lock,
                    pre_trained_model=model_object
                )
                champion_metrics = {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'wape': np.nan, 'accuracy': np.nan, 'r2': np.nan}
                final_model_type = f"Category_Transfer_{model_type}"
                final_model_to_save = {}
            else:
                logging.warning(f"SKU {sku_id}: No category model found for '{sku_cat}'. Falling back to SMA.")
                final_forecast_df, champion_metrics = run_sma_forecast(model_df, CONFIG)
                final_model_type = 'SMA'
                final_model_to_save = {}
        else:
            model_df = _engineer_time_series_features(model_df)
            sku_results['last_processed_date'] = model_df['ds'].max().strftime('%Y-%m-%d')
            test_days = CONFIG.TEST_DAYS
            train_df, test_df = model_df.iloc[:-test_days], model_df.iloc[-test_days:].copy()
            dl_features = ['y', 'store_count', 'was_stocked_out', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'is_on_promotion', 'discount_percentage', 'is_month_start', 'is_month_end', 'sales_lag_1d', 'sales_lag_7d', 'sales_lag_14d', 'sales_lag_30d', 'rolling_avg_sales_7d', 'rolling_std_sales_7d']
            xgb_features = [col for col in dl_features if col != 'y']
            
            prophet_metrics, prophet_test_forecast, best_prophet_params, prophet_regressors = _evaluate_prophet_model(train_df, test_df, holidays_df, sku_id)
            lstm_metrics, cnn_lstm_metrics, transformer_metrics, lstm_yhat, cnn_lstm_yhat, transformer_yhat = _evaluate_deep_learning_models(train_df, test_df, dl_features, CONFIG.LSTM_CONFIG['n_steps'], model_dir, sku_id, keras_lock, store_growth_df)
            xgb_metrics, xgb_yhat, best_xgb_params = _evaluate_xgboost_model(train_df, test_df, xgb_features, sku_id)
            
            models_evaluated = {'Tuned_Prophet': prophet_metrics, 'XGBoost': xgb_metrics, 'LSTM': lstm_metrics, 'CNN_LSTM': cnn_lstm_metrics, 'Transformer': transformer_metrics}
            
            final_model_type, champion_metrics, weights = _select_and_forecast_champion_model(sku_id, models_evaluated, test_df, prophet_test_forecast, xgb_yhat, lstm_yhat, cnn_lstm_yhat, transformer_yhat)
            final_forecast_df, final_model_to_save = _generate_future_forecasts(model_df, holidays_df, store_growth_df, final_model_type, weights, best_prophet_params, prophet_regressors, best_xgb_params, dl_features, CONFIG.LSTM_CONFIG['n_steps'], model_dir, sku_id, keras_lock)
        
        sku_results.update(champion_metrics)
        sku_results['final_forecast_model_type'] = final_model_type
        
        for name, metrics in models_evaluated.items():
            if metrics:
                sku_results.update({f'{name.lower()}_{k}': v for k, v in metrics.items()})
        
        sku_results = _calculate_and_store_shap_values(final_model_type, final_model_to_save, model_df, xgb_features if 'xgb_features' in locals() else [], sku_results)
        
        try:
            inventory_metrics = calculate_inventory_metrics(final_forecast_df, sku_df['unit_price'].mean(), model_df['y'].std(), CONFIG)
            sku_results.update({k.replace(' ', '_').lower(): v for k, v in inventory_metrics.items()})
        except Exception as e:
            sku_results['inventory_calc_error'] = str(e)
            
        if not final_forecast_df.empty:
            sku_results['forecast_next_1_month'] = final_forecast_df.head(30)['yhat'].sum()
            sku_results['forecast_next_3_months'] = final_forecast_df.head(90)['yhat'].sum()
            sku_results['forecast_next_6_months'] = final_forecast_df.head(183)['yhat'].sum()
            ci_width = (final_forecast_df['yhat_upper'] - final_forecast_df['yhat_lower']).clip(lower=0)
            sku_results['forecast_ci_width_1_month'] = ci_width.head(30).mean()
            sku_results['forecast_ci_width_3_months'] = ci_width.head(90).mean()
        else:
            sku_results['forecast_next_1_month'] = np.nan
            sku_results['forecast_next_3_months'] = np.nan
            sku_results['forecast_next_6_months'] = np.nan
            sku_results['forecast_ci_width_1_month'] = np.nan
            sku_results['forecast_ci_width_3_months'] = np.nan
            
        if final_model_to_save:
            joblib.dump(final_model_to_save, os.path.join(model_dir, f'model_{sku_id}.pkl'))
            
        return sku_results, final_forecast_df
    except Exception as e:
        logging.error(f"SKU {sku_id} processing FAILED: {e}", exc_info=True)
        return {'sku_id': sku_id, 'error': str(e)}, pd.DataFrame()


def _load_and_prepare_data():
    """Loads and performs initial preparation of all data from MongoDB."""
    logging.info("\n--- 🚚 Loading data from MongoDB ---")
    try:
        sales_df = data_loader.load_dataframe_from_mongo(CONFIG.SALES_COLLECTION)
        inventory_df = data_loader.load_dataframe_from_mongo(CONFIG.INVENTORY_COLLECTION)
        holidays_df = data_loader.load_dataframe_from_mongo(CONFIG.HOLIDAYS_COLLECTION)
        store_growth_df = data_loader.load_dataframe_from_mongo(CONFIG.STORE_COUNT_COLLECTION)
        logging.info("✅ All data loaded successfully from MongoDB.")
    except Exception as e:
        sys.exit(f"❌ Critical Error: Failed to load data from MongoDB - {e}")
    
    
    for col in ['revenue', 'disc', 'qty']:
        if col in sales_df.columns:
            sales_df[col] = sales_df[col].apply(clean_numeric)
    sales_df.dropna(subset=['qty', 'revenue'], inplace=True)
    sales_df['created_at'] = pd.to_datetime(sales_df['created_at'], errors='coerce', dayfirst=True).dt.normalize()
    sales_df.dropna(subset=['created_at', 'sku'], inplace=True)
    inventory_df.rename(columns={'date': 'created_at'}, inplace=True)
    inventory_df['created_at'] = pd.to_datetime(inventory_df['created_at'], errors='coerce').dt.normalize()
    inventory_agg = inventory_df.groupby(['created_at', 'sku']).agg({'wh': 'sum'}).reset_index()
    sales_df = sales_df.merge(inventory_agg, on=['created_at', 'sku'], how='left')
    sales_df['was_stocked_out'] = (sales_df['wh'].fillna(1) == 0).astype(int)
    sales_df['wh'] = sales_df.groupby('sku')['wh'].ffill().bfill()
    sales_df['unit_price'] = (sales_df['revenue'] / sales_df['qty']).replace([np.inf, -np.inf], np.nan)
    sales_df['unit_price'].fillna(sales_df.groupby('sku')['unit_price'].transform('median'), inplace=True)
    sales_df['unit_price'].fillna(sales_df['unit_price'].median(), inplace=True)
    sales_df.sort_values('created_at', inplace=True)
    holidays_df = holidays_df.rename(columns={'Date': 'ds', 'Name': 'holiday'})
    holidays_df['ds'] = pd.to_datetime(holidays_df['ds'], errors='coerce', dayfirst=True).dt.normalize()
    holidays_df = holidays_df[holidays_df['Type'].str.contains('|'.join(CONFIG.HOLIDAY_TYPES), na=False)]
    holidays_df = holidays_df[['ds', 'holiday']].drop_duplicates().dropna()
    
    store_growth_df.columns = store_growth_df.columns.str.strip()
    store_growth_df['created_at'] = pd.to_datetime(store_growth_df['year'].astype(str) + '-' + store_growth_df['month'].astype(str) + '-01')
    store_growth_df.rename(columns={'stores': 'store_count'}, inplace=True)
    store_growth_df.drop(columns=['month', 'year'], inplace=True, errors='ignore')
    store_growth_df.dropna(subset=['created_at', 'store_count'], inplace=True)
    store_growth_df.set_index('created_at', inplace=True)
    store_growth_df.sort_index(inplace=True)
    full_date_range = pd.date_range(start=store_growth_df.index.min(), end=store_growth_df.index.max(), freq='D')
    store_growth_df = store_growth_df.reindex(full_date_range)
    store_growth_df['store_count'] = store_growth_df['store_count'].interpolate(method='linear').ffill().bfill()
    store_growth_df.reset_index(inplace=True)
    store_growth_df.rename(columns={'index': 'created_at'}, inplace=True)
    return sales_df, holidays_df, store_growth_df

def _perform_sku_clustering(sales_df):
    """Performs Time Series K-Means clustering on SKUs with enhanced logging."""
    all_skus = list(sales_df['sku'].dropna().unique())
    if CONFIG.MAX_SKUS_FOR_CLUSTERING is not None and CONFIG.MAX_SKUS_FOR_CLUSTERING > 0:
        random.shuffle(all_skus)
        all_skus = all_skus[:CONFIG.MAX_SKUS_FOR_CLUSTERING]
        logging.info(f"🎯 Limiting clustering to {len(all_skus)} SKUs as per 'max_skus_for_clustering' config.")
    total_skus_to_cluster = len(all_skus)
    logging.info(f"\n--- 📈 Clustering {total_skus_to_cluster} SKUs based on sales patterns (DTW) ---")
    sku_to_cluster_map = {}
    skus_processed_for_clustering_prep = 0
    skus_eligible_for_clustering = 0
    try:
        sku_timeseries = []
        processed_skus = []
        min_len = CONFIG.MIN_DATA_POINTS_DEEP_LEARNING
        logging.info(f"🔍 Preparing SKU time series for clustering (min_data_points_deep_learning = {min_len})...")
        for i, sku in enumerate(all_skus):
            s = sales_df[sales_df['sku'] == sku]['qty']
            if len(s) >= min_len:
                sku_timeseries.append(s.values)
                processed_skus.append(sku)
                skus_eligible_for_clustering += 1
            skus_processed_for_clustering_prep += 1
            if (skus_processed_for_clustering_prep % 500 == 0) or (skus_processed_for_clustering_prep == total_skus_to_cluster):
                logging.info(f"  Processed {skus_processed_for_clustering_prep}/{total_skus_to_cluster} SKUs for clustering prep. Eligible: {skus_eligible_for_clustering}")
        logging.info(f"✅ Finished preparing time series. Found {skus_eligible_for_clustering} SKUs eligible for DTW clustering.")
        if skus_eligible_for_clustering > CONFIG.DTW_N_CLUSTERS:
            logging.info(f"📊 Padding and scaling {skus_eligible_for_clustering} eligible time series...")
            padded_series = tf.keras.preprocessing.sequence.pad_sequences(sku_timeseries, padding='post', dtype='float32')
            scaled_series = TimeSeriesScalerMinMax().fit_transform(padded_series)
            logging.info("Done padding and scaling.")
            num_jobs_for_dtw = CONFIG.DTW_N_JOBS
            logging.info(f"Starting DTW K-Means clustering with {CONFIG.DTW_N_CLUSTERS} clusters and {num_jobs_for_dtw} jobs...")
            dtw_clusterer = TimeSeriesKMeans(n_clusters=CONFIG.DTW_N_CLUSTERS, metric="dtw", max_iter=5, random_state=42, n_jobs=num_jobs_for_dtw, verbose=1)
            start_time = time.time()
            logging.info("\nFitting the DTW K-Means model... (This may take a while, see progress below)")
            cluster_labels = dtw_clusterer.fit_predict(scaled_series)
            end_time = time.time()
            logging.info(f"\n🎉 DTW K-Means clustering complete in {end_time - start_time:.2f} seconds.")
            sku_to_cluster_map = dict(zip(processed_skus, cluster_labels))
            logging.info(f"Clustering results: Found {len(np.unique(cluster_labels))} distinct clusters among eligible SKUs.")
        else:
            logging.warning(f"❗ Not enough SKUs ({skus_eligible_for_clustering}) with sufficient data to perform clustering. Minimum required: > {CONFIG.DTW_N_CLUSTERS}.")
            sku_to_cluster_map = {sku: -1 for sku in all_skus}
    except Exception as e:
        logging.error(f"⚠️ DTW Clustering failed: {e}", exc_info=True)
        sku_to_cluster_map = {sku: -1 for sku in all_skus}
    return sku_to_cluster_map, all_skus

def _determine_skus_to_run(all_skus, existing_results_df):
    """Identifies which SKUs need to be processed or reprocessed."""
    skus_to_run = []
    logging.info("\n--- 🧠 Checking SKUs for retraining based on last_processed_date ---")
    last_run_dates = {}
    if not existing_results_df.empty and 'last_processed_date' in existing_results_df.columns:
        last_run_dates = pd.Series(existing_results_df['last_processed_date'].values, index=existing_results_df['sku_id']).to_dict()
    skus_to_process_candidate = all_skus
    if CONFIG.MAX_SKUS_TO_PROCESS is not None:
        skus_to_process_candidate = skus_to_process_candidate[:CONFIG.MAX_SKUS_TO_PROCESS]
    for sku_id in skus_to_process_candidate:
        current_max_date = sales_df[sales_df['sku'] == sku_id]['created_at'].max()
        last_model_date = last_run_dates.get(sku_id)
        if last_model_date is None:
            logging.info(f"SKU {sku_id}: 🆕 Not found in previous results. Adding to training queue.")
            skus_to_run.append(sku_id)
            continue
        last_model_date = pd.to_datetime(last_model_date, errors='coerce')
        if pd.isna(last_model_date) or current_max_date > last_model_date:
            logging.info(f"SKU {sku_id}: 📈 New data found (up to {current_max_date.date()}). Adding to retraining queue.")
            skus_to_run.append(sku_id)
        else:
            logging.info(f"SKU {sku_id}: ✔️ Model is up-to-date (processed on {last_model_date.date()}). Skipping.")
    logging.info(f"--- Check complete --- \nFinal number of SKUs to run/retrain: {len(skus_to_run)}")
    return skus_to_run

def train_category_models(sales_df, holidays_df, store_growth_df, model_dir, keras_lock, categories_to_train):
    print("\n\n--- 🏭 Starting Category-Level Model Training ---")
    category_models = {}
    sales_df['category'] = sales_df['sku'].apply(_get_sku_category)
    
    print(f"Target categories for training: {categories_to_train}")

    for category in categories_to_train:
        print(f"\n--- Processing Category: {category} ---")
        category_sales_df = sales_df[sales_df['category'] == category]
        sku_counts = category_sales_df.groupby('sku').size()
        rich_skus = sku_counts[sku_counts >= CONFIG.MIN_DATA_POINTS_DEEP_LEARNING].index.tolist()
        
        if not rich_skus:
            print(f"Category {category}: No SKUs with sufficient data found. Cannot train a category model.")
            continue
            
        print(f"Category {category}: Found {len(rich_skus)} data-rich SKUs for training the category model.")
        category_rich_df = category_sales_df[category_sales_df['sku'].isin(rich_skus)]
        category_model_df = _prepare_sku_data(category_rich_df, store_growth_df)
        
        try:
            category_model_df = _engineer_time_series_features(category_model_df)
            test_days = CONFIG.TEST_DAYS
            train_df, test_df = category_model_df.iloc[:-test_days], category_model_df.iloc[-test_days:].copy()
            dl_features = ['y', 'store_count', 'was_stocked_out', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'is_on_promotion', 'discount_percentage', 'is_month_start', 'is_month_end', 'sales_lag_1d', 'sales_lag_7d', 'sales_lag_14d', 'sales_lag_30d', 'rolling_avg_sales_7d', 'rolling_std_sales_7d']
            xgb_features = [col for col in dl_features if col != 'y']
            
            prophet_metrics, prophet_fcst, best_prophet_params, prophet_regressors = _evaluate_prophet_model(train_df, test_df, holidays_df, category)
            lstm_metrics, cnn_lstm_metrics, transformer_metrics, lstm_yhat, cnn_lstm_yhat, transformer_yhat = _evaluate_deep_learning_models(train_df, test_df, dl_features, CONFIG.LSTM_CONFIG['n_steps'], model_dir, f"cat_{category}", keras_lock, store_growth_df)
            xgb_metrics, xgb_yhat, best_xgb_params = _evaluate_xgboost_model(train_df, test_df, xgb_features, category)
            
            models_evaluated = {'Tuned_Prophet': prophet_metrics, 'XGBoost': xgb_metrics, 'LSTM': lstm_metrics, 'CNN_LSTM': cnn_lstm_metrics, 'Transformer': transformer_metrics}
            
            final_model_type, _, weights = _select_and_forecast_champion_model(
                category, models_evaluated, test_df, prophet_fcst, xgb_yhat, 
                lstm_yhat, cnn_lstm_yhat, transformer_yhat
            )
            
            _, final_model_to_save = _generate_future_forecasts(
                category_model_df, holidays_df, store_growth_df, final_model_type, weights,
                best_prophet_params, prophet_regressors, best_xgb_params,
                dl_features, CONFIG.LSTM_CONFIG['n_steps'], model_dir, f"cat_{category}", keras_lock
            )
            
            if final_model_to_save:
                champion_model_object = list(final_model_to_save.values())[0]
                category_models[category] = (final_model_type, champion_model_object)
                print(f"✅ Category {category}: Successfully trained and saved champion model ({final_model_type}).")
            else:
                print(f"Category {category}: Could not generate a final model to save.")
        except Exception as e:
            print(f"Category {category}: FAILED to train a champion model. {e}")
            
    print("\n--- ✅ Finished Training All Category-Level Models ---")
    return category_models

def _get_sku_category(sku_id):
    """Extracts the alphabetical prefix from an SKU ID to determine its category."""
    if not isinstance(sku_id, str):
        return None
    match = re.match(r'([A-Z]+)', sku_id)
    return match.group(1) if match else None

def main():
    """
    Main function to orchestrate the advanced SKU demand forecasting process.
    """
    verify_gpu()
    
    global sales_df
    sales_df, holidays_df, store_growth_df = _load_and_prepare_data()

    os.makedirs(CONFIG.MODEL_SAVE_PATH, exist_ok=True)

    try:
        existing_results_df = data_loader.load_dataframe_from_mongo(CONFIG.MASTER_RESULTS_COLLECTION)
        if 'last_processed_date' in existing_results_df.columns:
            existing_results_df['last_processed_date'] = pd.to_datetime(existing_results_df['last_processed_date'], errors='coerce')
        print(f"✅ Successfully loaded {len(existing_results_df)} records from existing output collection.")
    except Exception as e:
        print(f"⚠️ Could not load existing results: {e}. Starting fresh.")
        existing_results_df = pd.DataFrame()

    
    
    sku_to_cluster_map, all_skus = _perform_sku_clustering(sales_df)
    
    
    skus_to_run = _determine_skus_to_run(all_skus, existing_results_df)

    if not skus_to_run:
        print("\n✅ No new data or new SKUs to process. Script finished.")
        sys.exit()
        
    
    df_to_run = sales_df[sales_df['sku'].isin(skus_to_run)].copy()
    df_to_run['category'] = df_to_run['sku'].apply(_get_sku_category)
    categories_to_train = df_to_run['category'].dropna().unique().tolist()

    
    category_models = train_category_models(sales_df, holidays_df, store_growth_df, CONFIG.MODEL_SAVE_PATH, threading.Lock(), categories_to_train)

    
    keras_lock = threading.Lock()
    total_skus_to_run = len(skus_to_run)
    print(f"\nPreparing to process {total_skus_to_run} SKU(s).")
    
    tasks = [
        (sku, sales_df[sales_df['sku'] == sku].copy(), holidays_df, store_growth_df, CONFIG.MODEL_SAVE_PATH, keras_lock, idx + 1, total_skus_to_run, sku_to_cluster_map, category_models)
        for idx, sku in enumerate(skus_to_run)
    ]
    
    all_sku_results = []
    all_daily_forecasts = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG.MAX_WORKERS) as executor:
        future_to_sku = {executor.submit(process_sku, task): task[0] for task in tasks}
        
        for future in concurrent.futures.as_completed(future_to_sku):
            sku_id = future_to_sku[future]
            try:
                sku_results, daily_forecast = future.result()
                if sku_results:
                    all_sku_results.append(sku_results)
                if daily_forecast is not None and not daily_forecast.empty:
                    all_daily_forecasts.append(daily_forecast)
            except Exception as exc:
                print(f"SKU {sku_id} generated an exception during result collection: {exc}")

    if all_sku_results:
        master_results_df = pd.DataFrame(all_sku_results)
        data_loader.save_dataframe_to_mongo(master_results_df, CONFIG.MASTER_RESULTS_COLLECTION, mode='upsert', upsert_key='sku_id')
        print(f"✅ Saved/Updated {len(master_results_df)} master results to MongoDB.")

    if all_daily_forecasts:
        daily_forecasts_df = pd.concat(all_daily_forecasts, ignore_index=True)
        data_loader.save_dataframe_to_mongo(daily_forecasts_df, CONFIG.DAILY_FORECASTS_COLLECTION, mode='delete_and_insert')
        print(f"✅ Saved {len(daily_forecasts_df)} daily forecast records to MongoDB.")

    print("\n\n🎉 Advanced pipeline finished. All results have been saved to MongoDB.")

if __name__ == '__main__':
    main()
