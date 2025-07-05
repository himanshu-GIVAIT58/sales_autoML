# src/promo_analyzer.py

import pandas as pd
from src import data_loader, model_handler, config
# Import the new function and the other necessary feature creators
from src.feature_engineering import (
    generate_static_features, 
    prepare_data_for_analysis 
)
from src.model_handler import prepare_prediction_data
from autogluon.timeseries import TimeSeriesDataFrame

def analyze_promotion_lift(predictor, lookback_days=184):
    """
    Analyzes the sales lift from promotions by comparing actual sales to a
    counterfactual forecast where no promotions were run.
    """
    print(f"\n🔬 Starting promotion lift analysis for the last {lookback_days} days...")

    # 1. --- KEY FIX: Load raw data and prepare it specifically for this analysis ---
    try:
        # Load the raw data sources
        sales_data = data_loader.load_dataframe_from_mongo("sales_data")
        holidays_data = data_loader.load_dataframe_from_mongo("holidays_data")
        
        # Use the new dedicated function to prepare the analysis data
        analysis_data = prepare_data_for_analysis(sales_data, holidays_data, max_skus=config.MAX_SKUS)
        
        static_feature_columns, _ = model_handler.load_prediction_artifacts()
    except Exception as e:
        print(f"❌ Error loading and preparing data: {e}")
        return pd.DataFrame()

    # 2. Filter data to the analysis period
    analysis_end_date = analysis_data['timestamp'].max()
    analysis_start_date = analysis_end_date - pd.Timedelta(days=lookback_days - 1)
    analysis_period_data = analysis_data[analysis_data['timestamp'] >= analysis_start_date].copy()
    
    # Use the 'disc' column to find promoted items
    promoted_items_mask = analysis_period_data['disc'] > 0
    analysis_data_promoted = analysis_period_data[promoted_items_mask]

    if analysis_data_promoted.empty:
        print(" -> No items with discounts > 0 found in the recent historical data to analyze.")
        return pd.DataFrame()

    # 3. Create and prepare the "counterfactual" dataset
    counterfactual_data = analysis_period_data.copy()
    counterfactual_data['disc'] = 0
    prepared_counterfactual_data = prepare_prediction_data(counterfactual_data, holidays_data)
    
    # ... (The rest of the function for generating predictions and summarizing is unchanged)
    # 4. Generate Static Features and TimeSeriesDataFrame
    static_features = generate_static_features(prepared_counterfactual_data, all_training_columns=static_feature_columns)
    static_features.reset_index(inplace=True)
    ts_counterfactual = TimeSeriesDataFrame.from_data_frame(
        prepared_counterfactual_data,
        id_column='item_id',
        timestamp_column='timestamp',
        static_features_df=static_features
    )

    # 5. Generate counterfactual predictions
    try:
        counterfactual_predictions = predictor.predict(
            ts_counterfactual,
            known_covariates=ts_counterfactual[config.KNOWN_COVARIATES_NAMES]
        )
    except Exception as e:
        counterfactual_predictions = predictor.predict(
            ts_counterfactual,
            known_covariates=ts_counterfactual[config.KNOWN_COVARIATES_NAMES],
            model='SeasonalNaive'
        )

    # 6. Compare and summarize results
    results = analysis_data_promoted[['item_id', 'timestamp', 'target']].copy()
    results.rename(columns={'target': 'actual_sales'}, inplace=True)
    results = pd.merge(results, counterfactual_predictions[['mean']].reset_index(), on=['item_id', 'timestamp'], how='inner')
    results.rename(columns={'mean': 'forecasted_sales_no_promo'}, inplace=True)
    
    results['sales_lift_units'] = results['actual_sales'] - results['forecasted_sales_no_promo']
    
    lift_summary = results.groupby('item_id').agg(
        total_actual_sales=("actual_sales", "sum"),
        total_forecasted_sales_no_promo=("forecasted_sales_no_promo", "sum"),
        total_lift_units=("sales_lift_units", "sum")
    ).reset_index()

    lift_summary['percentage_lift'] = (lift_summary['total_lift_units'] / lift_summary['total_forecasted_sales_no_promo'].replace(0, 1)) * 100
    lift_summary.replace([float('inf'), -float('inf')], 0, inplace=True)

    print("✅ Promotion lift analysis complete.")
    return lift_summary.sort_values(by='percentage_lift', ascending=False)
