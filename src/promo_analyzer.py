import pandas as pd
from src import data_loader, model_handler, config
from src.feature_engineering import (
    generate_static_features, 
    prepare_data_for_analysis 
)
from src.model_handler import prepare_prediction_data
from autogluon.timeseries import TimeSeriesDataFrame

def analyze_promotion_lift(predictor, lookback_days=184):
    print(f"\n🔬 Starting promotion lift analysis for the last {lookback_days} days...")

    try:
        # Load the raw data sources
        sales_data = data_loader.load_dataframe_from_mongo("sales_data")
        holidays_data = data_loader.load_dataframe_from_mongo("holidays_data")
        
        # Prepare the analysis data
        analysis_data = prepare_data_for_analysis(sales_data, holidays_data, max_skus=config.MAX_SKUS)
        static_feature_columns, _ = model_handler.load_prediction_artifacts()
    except Exception as e:
        print(f"❌ Error loading and preparing data: {e}")
        return pd.DataFrame()

    # Filter data to the analysis period
    analysis_end_date = analysis_data['timestamp'].max()
    analysis_start_date = analysis_end_date - pd.Timedelta(days=lookback_days - 1)
    analysis_period_data = analysis_data[analysis_data['timestamp'] >= analysis_start_date].copy()
    
    # Use the 'disc' column
    promoted_items_mask = analysis_period_data['disc'] > 0
    analysis_data_promoted = analysis_period_data[promoted_items_mask]

    if analysis_data_promoted.empty:
        print(" -> No items with discounts > 0 found in the recent historical data to analyze.")
        return pd.DataFrame()
    
    # Create and prepare the "counterfactual" dataset (set all 'disc' to 0)
    counterfactual_data = analysis_period_data.copy()
    counterfactual_data['disc'] = 0
    prepared_counterfactual_data = prepare_prediction_data(counterfactual_data, holidays_data)

    # Generate Static Features and TimeSeriesDataFrame
    static_features = generate_static_features(prepared_counterfactual_data, all_training_columns=static_feature_columns)
    static_features.reset_index(inplace=True)
    ts_counterfactual = TimeSeriesDataFrame.from_data_frame(
        prepared_counterfactual_data,
        id_column='item_id',
        timestamp_column='timestamp',
        static_features_df=static_features
    )

    try:
        counterfactual_predictions = predictor.predict(
            ts_counterfactual,
            known_covariates=ts_counterfactual[config.KNOWN_COVARIATES_NAMES]
        )
    except Exception:
        counterfactual_predictions = predictor.predict(
            ts_counterfactual,
            known_covariates=ts_counterfactual[config.KNOWN_COVARIATES_NAMES],
            model='SeasonalNaive'
        )

    results = analysis_data_promoted[['item_id', 'timestamp', 'target']].copy()
    results.rename(columns={'target': 'actual_sales'}, inplace=True)
    results = pd.merge(
        results,
        counterfactual_predictions[['mean']].reset_index(),
        on=['item_id', 'timestamp'],
        how='inner'
    )
    results.rename(columns={'mean': 'forecasted_sales_no_promo'}, inplace=True)
    results['sales_lift_units'] = results['actual_sales'] - results['forecasted_sales_no_promo']

    lift_summary = results.groupby('item_id').agg(
        total_actual_sales=("actual_sales", "sum"),
        total_forecasted_sales_no_promo=("forecasted_sales_no_promo", "sum"),
        total_lift_units=("sales_lift_units", "sum")
    ).reset_index()
    
    lift_summary['percentage_lift'] = (
        lift_summary['total_lift_units'] / lift_summary['total_forecasted_sales_no_promo'].replace(0, 1)
    ) * 100
    lift_summary.replace([float('inf'), -float('inf')], 0, inplace=True)

    print("✅ Promotion lift analysis complete.")
    return lift_summary.sort_values(by='percentage_lift', ascending=False)
