import os 

ORDERING_COST = 10
ANNUAL_HOLDING_COST_PER_UNIT = 20
LEAD_TIME_DAYS = 40
MINIMUM_ORDER_QUANTITY = 10

PROJECT_ROOT = "/app"
PROJECT_SRC = os.path.join(PROJECT_ROOT, "src")


MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "autogluon_models")
ARTIFACTS_PATH = os.path.join(PROJECT_ROOT, "artifacts")


ABC_CONFIG = {
    'A_class_percentage': 0.2,
    'B_class_percentage': 0.3,
    'service_level_A': 0.98,
    'service_level_B': 0.95,
    'service_level_C': 0.90,
}

MIN_SEASONAL_STRENGTH = 0.6 
SEASONAL_PERIOD_DAYS = 365 

DRIFT_MEAN_THRESHOLD = 0.1
DRIFT_P_VALUE_THRESHOLD = 0.05
IMPROVEMENT_THRESHOLD = 0.05 

MAX_SKUS = 100
PREDICTION_LENGTH = 183
RANDOM_SEED= 42

KNOWN_COVARIATES_NAMES = [
    'was_stocked_out', 'is_holiday', 'is_on_promotion', 'month', 'day_of_week',
    'is_weekend', 'is_wedding_season', 'is_diwali_period', 'is_valentine_month',
    'is_high_discount', 'sales_ma_7', 'sales_ma_14', 'stockout_days_last_7',
    'potential_lost_sales', 'gold_price_change', 'gold_price_ma_7',
     'sales_lag_1',
    'sales_lag_7',
    'sales_lag_14',
    'sales_std_7',
    'sales_std_30'
]

TIMESTAMP_COL = 'timestamp'
TARGET_COL = 'target'
ITEM_ID_COL = 'sku'

AUTOGLUON_PRESETS = "fast_training"  
TIME_LIMIT = 180
NUM_VAL_WINDOWS = 5
EVAL_METRIC = "MASE"
FREQ = "D"
QUANTILE_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]
TARGET_COLUMN = "target"

HORIZONS = {"1-Month": 30, "3-Month": 90, "6-Month": 183}
CHANNEL_TO_WH_RECOMMENDATION = {
    'Web': 'central_online_wh',
    'Offline': 'central_offline_wh',
    'App': 'central_online_wh',
    'Unknown': 'main_wh'
}

