# config.py
"""
Contains all configuration variables for the inventory forecasting project.
Adjust the values in this file to suit your business needs.
"""

import pandas as pd

# --- 1. Inventory Management & Forecasting Configuration ---
# Cost to place a single order (e.g., administrative costs, shipping fees)
ORDERING_COST = 20.0  # Example: $20 per order

# Cost to hold one unit of inventory for one year (e.g., storage, insurance)
ANNUAL_HOLDING_COST_PER_UNIT = 3.0 # Example: $3 per unit per year

# Time in days it takes for an order to arrive after being placed
LEAD_TIME_DAYS = 40 # Example: 40 days

# The minimum number of units you can order from your supplier.
MINIMUM_ORDER_QUANTITY = 10

# --- 2. ABC Analysis Configuration ---
# This allows applying different service levels to different classes of products.
ABC_CONFIG = {
    'A_class_percentage': 0.20,  # Top 20% of items by sales volume are 'A'
    'B_class_percentage': 0.30,  # Next 30% of items are 'B'
    'service_level_A': 0.98,     # e.g., 98% service level for most important items
    'service_level_B': 0.95,     # e.g., 95% for moderately important items
    'service_level_C': 0.90,     # e.g., 90% for least important items
}

# --- 3. Forecast Horizon ---
# The number of days into the future to forecast.
# We need to forecast for the longest horizon required, which is 6 months (approx. 183 days)
PREDICTION_LENGTH = 183

# --- 4. Data Source File Names ---
SOURCE_FILENAME = 'sales_data_complete___daily_drill_down_2025-05-29T12_37_43.113222731+05_30.csv'
INVENTORY_FILENAME = 'query_result_2025-05-28T18_02_43.550629445+05_30.csv'
HOLIDAYS_FILENAME = 'indian_holidays.csv'
OUTPUT_FILENAME = 'inventory_recommendations.csv'

# --- 5. SKU Limiting for Test Runs ---
# Set to None to run on all SKUs, or a number (e.g., 10) to limit to the top N SKUs.
# This is useful for faster testing and debugging.
MAX_SKUS = 3000

# --- 6. Model Training Configuration ---
# These are the covariates that the model will know about in the future.
KNOWN_COVARIATES_NAMES = [
    'was_stocked_out', 'is_holiday', 'is_on_promotion', 'month', 'day_of_week',
    'is_weekend', 'is_wedding_season', 'is_diwali_period', 'is_valentine_month',
    'is_high_discount', 'sales_ma_7', 'sales_ma_14', 'stockout_days_last_7',
    'potential_lost_sales', 'gold_price_change', 'gold_price_ma_7'
]

# AutoGluon training settings
AUTOGLUON_PRESETS = "fast_training" # Use a lighter preset for quick testing
TIME_LIMIT = 300                    # Reduced time limit for training in seconds
NUM_VAL_WINDOWS = 5                 # Number of validation windows for speed

# --- 7. Inventory Calculation Horizons ---
HORIZONS = {"1-Month": 30, "3-Month": 90, "6-Month": 183}
CHANNEL_TO_WH_RECOMMENDATION = {'Web': 'central_online_wh', 'Offline': 'central_offline_wh', 'App': 'central_online_wh', 'Unknown': 'main_wh'}

