import os

# --- Inventory & Business Logic ---
ORDERING_COST = 10
ANNUAL_HOLDING_COST_PER_UNIT = 20
LEAD_TIME_DAYS = 40
MINIMUM_ORDER_QUANTITY = 10

ABC_CONFIG = {
    'A_class_percentage': 0.2,
    'B_class_percentage': 0.3,
    'service_level_A': 0.98,
    'service_level_B': 0.95,
    'service_level_C': 0.90,
}

HORIZONS = {"1-Month": 30, "3-Month": 90, "6-Month": 183}
CHANNEL_TO_WH_RECOMMENDATION = {
    'Web': 'central_online_wh',
    'Offline': 'central_offline_wh',
    'App': 'central_online_wh',
    'Unknown': 'main_wh'
}

# --- System Paths ---
PROJECT_ROOT = "/app"
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "autogluon_models")
ARTIFACTS_PATH = os.path.join(PROJECT_ROOT, "artifacts")

# --- Data Schema ---
TIMESTAMP_COL = 'timestamp'
TARGET_COL = 'target'
ITEM_ID_COL = 'sku'

DATA_COLUMNS = {
    "ITEM_ID": "sku",
    "TIMESTAMP": "timestamp",
    "TARGET": "target",
}

MAX_SKUS = None  # Set to an integer for testing, None for all.
PREDICTION_LENGTH = 183
RANDOM_SEED = 42
# 
AUTOGLUON_PRESETS = ["fast_training"]

TIME_LIMIT = 3600  # 8 hours
NUM_VAL_WINDOWS = 5
EVAL_METRIC = "MASE"
FREQ = "D"
QUANTILE_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]
MIN_SEASONAL_STRENGTH = 0.2

# "medium_quality","high_quality","best_quality"
# ,"medium_quality","high_quality","best_quality"
TRAINING = {
    "PREDICTION_LENGTH": 183,
    "FREQ": "D",
    "EVAL_METRIC": "MASE", 
    "PRESETS": ["fast_training"], # Using a list for multi-preset training
    "TIME_LIMIT": 3600,  # 1 hour
    "NUM_VAL_WINDOWS": 5,
    "QUANTILE_LEVELS": [0.1, 0.25, 0.5, 0.75, 0.9],
    "RANDOM_SEED": 42,
    "MAX_SKUS_TO_TRAIN": None,  # Set to an integer for testing, None for all.
}

KNOWN_COVARIATES_NAMES = [
    'was_stocked_out', 'is_holiday', 'is_on_promotion', 'month', 'day_of_week',
    'is_weekend', 'is_wedding_season', 'is_diwali_period', 'is_valentine_month',
    'is_high_discount', 'sales_ma_7', 'sales_ma_14', 'stockout_days_last_7',
    'potential_lost_sales', 'gold_price_change', 'gold_price_ma_7',
    'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_std_7', 'sales_std_30',
    'feedback_score_30d_avg',
    'time_since_last_sale',
    'days_with_sales_last_7d'
]

IMPROVEMENT_THRESHOLD = 0.05
DRIFT_P_VALUE_THRESHOLD = 0.05

VALIDATION = {
    "IMPROVEMENT_THRESHOLD": 0.05,  # 5% better performance required.
    "DRIFT_P_VALUE_THRESHOLD": 0.05, # Statistical significance for drift.
}
