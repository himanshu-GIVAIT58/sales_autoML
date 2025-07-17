import os

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SRC_DIR)
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "advanced_models")
ARTIFACTS_PATH = os.path.join(PROJECT_ROOT, "advanced_artifacts")
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "logs", "advanced_pipeline_log.txt")

SALES_COLLECTION = "sales_data"
INVENTORY_COLLECTION = "query_result"
HOLIDAYS_COLLECTION = "holidays_data"
STORE_COUNT_COLLECTION = "store_count"

MASTER_RESULTS_COLLECTION = "advanced_forecast_master"
DAILY_FORECASTS_COLLECTION = "advanced_forecast_daily"

MAX_SKUS_TO_PROCESS = 2      
MAX_SKUS_FOR_CLUSTERING = 2  
MAX_WORKERS = 1              
MIN_DATA_POINTS_DEEP_LEARNING = 30  
MIN_DATA_POINTS_PROPHET = 30
TEST_DAYS = 7                
SPORADIC_THRESHOLD = 0.60
SMA_WINDOW = 7               
ENSEMBLE_WAPE_THRESHOLD = 50.0
HOLIDAY_TYPES = ['National holiday', 'Optional holiday']

LSTM_CONFIG = {'n_steps': 2, 'epochs': 1, 'batch_size': 8, 'max_trials': 1}
CNN_LSTM_CONFIG = {'n_steps': 2, 'epochs': 1, 'batch_size': 8, 'max_trials': 1}
TRANSFORMER_CONFIG = {'n_steps': 2, 'epochs': 1, 'batch_size': 8, 'max_trials': 1}
PROPHET_TUNING_PARAMS = {
    'changepoint_prior_scale': [0.05],   
    'seasonality_prior_scale': [1.0]     
}

INVENTORY_CONFIG = {
    'avg_lead_time_days': 21,
    'moq': 20,
    'service_level': 0.95,
    'holding_cost_percentage': 0.10,
    'ordering_cost_percentage': 0.30
}

MC_DROPOUT_SAMPLES = 5      
QUANTILES = [0.1, 0.5, 0.9]
DTW_N_CLUSTERS = 2          
DTW_N_JOBS = 1              
