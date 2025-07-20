import pandas as pd
from src import data_loader, model_handler, config
from src.feature_engineering import (
    generate_static_features,
    prepare_data
)
from src.model_handler import prepare_prediction_data
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

def analyze_promotion_lift(predictor: TimeSeriesPredictor, lookback_days=184):
    print(f"\n🔬 Starting promotion lift analysis for the last {lookback_days} days...")

    try:
        sales_data = data_loader.load_dataframe_from_mongo("sales_data")
        print("Columns in sales_data:", sales_data.columns.tolist())
        print(sales_data.head())
        
        holidays_data = data_loader.load_dataframe_from_mongo("holidays_data")
        inventory_data = data_loader.load_dataframe_from_mongo("query_result")
        analysis_data_tuple = prepare_data(
            sales_data, 
            inventory_data=inventory_data, 
            holidays_data=holidays_data, 
            max_skus=config.MAX_SKUS
        )
        
        if isinstance(analysis_data_tuple, tuple):
            analysis_data = analysis_data_tuple[0]
        else:
            analysis_data = analysis_data_tuple
        static_feature_columns, _ = model_handler.load_prediction_artifacts()
    except Exception as e:
        print(f"❌ Error loading and preparing data: {e}")
        return pd.DataFrame()

    print("Columns in analysis_data after prepare_data:", analysis_data.columns.tolist())

    analysis_end_date = analysis_data['timestamp'].max()
    analysis_start_date = analysis_end_date - pd.Timedelta(days=lookback_days - 1)
    analysis_period_data = analysis_data.loc[analysis_data['timestamp'] >= analysis_start_date].copy()

    print("First 5 rows after date filtering:")
    print(analysis_period_data[['timestamp', 'sku', 'is_on_promotion', 'is_high_discount']].head())
    print("Unique is_on_promotion values:", analysis_period_data['is_on_promotion'].unique())
    print("Unique is_high_discount values:", analysis_period_data['is_high_discount'].unique())
    print("Rows with is_on_promotion == 1:", (analysis_period_data['is_on_promotion'] == 1).sum())
    print("Rows with is_high_discount == 1:", (analysis_period_data['is_high_discount'] == 1).sum())

    
    promoted_items_mask = analysis_period_data['is_on_promotion'] == 1
    print("Promoted items mask:", promoted_items_mask.sum(), "items found.")
    analysis_data_promoted = analysis_period_data[promoted_items_mask]
    print("Columns in analysis_data_promoted:", analysis_data_promoted.columns.tolist())

    if analysis_data_promoted.empty:
        print(" -> No items with is_on_promotion == 1 found in the recent historical data to analyze.")
        return pd.DataFrame()

    counterfactual_data = analysis_period_data.copy()
    print("Columns in counterfactual_data before modification:", counterfactual_data.columns.tolist())
    counterfactual_data['is_on_promotion'] = 0
    counterfactual_data['is_high_discount'] = 0

    print("Columns in counterfactual_data after modification:", counterfactual_data.columns.tolist())

    try:
        prepared_counterfactual_data = prepare_prediction_data(counterfactual_data, holidays_data)
        print("Columns in prepared_counterfactual_data:", prepared_counterfactual_data.columns.tolist())
    except Exception as e:
        print("❌ Error in prepare_prediction_data:", e)
        return pd.DataFrame()
    
    
    prepared_counterfactual_data = prepared_counterfactual_data.dropna(subset=['item_id', 'timestamp'])

    
    prepared_counterfactual_data['item_id'] = prepared_counterfactual_data['item_id'].astype(str)
    prepared_counterfactual_data['timestamp'] = pd.to_datetime(prepared_counterfactual_data['timestamp'], errors='coerce')

    
    prepared_counterfactual_data = prepared_counterfactual_data.drop_duplicates(subset=['item_id', 'timestamp'])

    
    prepared_counterfactual_data = prepared_counterfactual_data.sort_values(['item_id', 'timestamp'])

    print("Any missing item_id?", prepared_counterfactual_data['item_id'].isnull().any())
    print("Any missing timestamp?", prepared_counterfactual_data['timestamp'].isnull().any())
    print("First 5 rows:\n", prepared_counterfactual_data[['item_id', 'timestamp']].head())

    
    holiday_cols = [c for c in prepared_counterfactual_data.columns if c.startswith('is_holiday')]
    if len(holiday_cols) > 1:
        for col in holiday_cols:
            if col != 'is_holiday': 
                prepared_counterfactual_data = prepared_counterfactual_data.drop(columns=col)

    
    for col in prepared_counterfactual_data.columns:
        if prepared_counterfactual_data[col].dtype == 'O' and col not in ['item_id', 'sku', 'channel']:
            prepared_counterfactual_data = prepared_counterfactual_data.drop(columns=col)

    static_features = generate_static_features(prepared_counterfactual_data, all_training_columns=static_feature_columns)
    static_features.reset_index(inplace=True) 

    print('static_features',static_features.columns.tolist())

    ts_counterfactual = TimeSeriesDataFrame.from_data_frame(
        prepared_counterfactual_data,
        id_column='item_id',
        timestamp_column='timestamp',
        static_features_df=static_features
    )
    print('ts_counterfactual',ts_counterfactual.columns.tolist());
    
    
    if not hasattr(predictor, 'prediction_length'): 
        if not hasattr(config, 'PREDICTION_LENGTH'):
            print("Warning: config.PREDICTION_LENGTH not found. Assuming prediction_length = 7 for demonstration.")
            prediction_length = 7 
        else:
            prediction_length = config.PREDICTION_LENGTH
    else:
        prediction_length = predictor.prediction_length


    
    future_known_covariates_ts = predictor.make_future_data_frame(
        ts_counterfactual 
    )
    
    
    

    
    if 'is_on_promotion' in future_known_covariates_ts.columns:
        future_known_covariates_ts['is_on_promotion'] = 0
    else:
        print("Warning: 'is_on_promotion' not found in future_known_covariates_ts. Make sure it was a known covariate during training.")
    
    if 'is_high_discount' in future_known_covariates_ts.columns:
        future_known_covariates_ts['is_high_discount'] = 0
    else:
        print("Warning: 'is_high_discount' not found in future_known_covariates_ts. Make sure it was a known covariate during training.")

    
    if 'is_holiday' in future_known_covariates_ts.columns and holidays_data is not None:
        holidays_data_dates = pd.to_datetime(holidays_data['date'])
        future_known_covariates_ts['is_holiday'] = future_known_covariates_ts.index.get_level_values('timestamp').isin(holidays_data_dates).astype(int)
    else:
         print("Warning: 'is_holiday' not found in future_known_covariates_ts or holidays_data is None.")

    
    
    

    
    
    
    future_known_covariates_df = future_known_covariates_ts.reset_index()

    for col in config.KNOWN_COVARIATES_NAMES:
        
        if col not in ['item_id', 'timestamp', 'target']:
            if col in prepared_counterfactual_data.columns: 
                
                
                last_values_for_col = prepared_counterfactual_data.groupby('item_id')[col].last().reset_index()
                last_values_for_col.rename(columns={col: f'{col}_last'}, inplace=True) 

                
                
                future_known_covariates_df = pd.merge(
                    future_known_covariates_df,
                    last_values_for_col,
                    on='item_id',
                    how='left'
                )
                future_known_covariates_df[col] = future_known_covariates_df[f'{col}_last']
                
                
                future_known_covariates_df.drop(columns=[f'{col}_last'], inplace=True)

                
                if future_known_covariates_df[col].isnull().any():
                    mean_val = prepared_counterfactual_data[col].mean()
                    [col] = future_known_covariates_df[col].fillna(mean_val)
            else:
                print(f"Warning: Known covariate '{col}' not found in prepared_counterfactual_data. It will not be populated in future covariates.")
    future_known_covariates_ts = TimeSeriesDataFrame.from_data_frame(
        future_known_covariates_df,
        id_column='item_id',
        timestamp_column='timestamp',
        static_features_df=static_features 
    )


    print("Columns in future_known_covariates_ts:", future_known_covariates_ts.columns.tolist())
    print("Future known covariates head:\n", future_known_covariates_ts.columns.tolist())

    counterfactual_predictions = predictor.predict(
        ts_counterfactual, 
        known_covariates=future_known_covariates_ts 
    )
    
    print('counterfactual_predictions',counterfactual_predictions.columns.tolist())

    results = analysis_data_promoted[['item_id', 'timestamp', 'target']].copy()
    print('results_1',results)
    results.rename(columns={'target': 'actual_sales'}, inplace=True)
    print('result2',results)
    
    counterfactual_predictions_reset = counterfactual_predictions.reset_index()
    print('after_reset',counterfactual_predictions_reset)
    
    results = pd.merge(
        results,
        counterfactual_predictions_reset[['item_id', 'timestamp', 'mean']], 
        on=['item_id', 'timestamp'],
        how='inner'
    )

    results.rename(columns={'mean': 'forecasted_sales_no_promo'}, inplace=True)
    results['sales_lift_units'] = results['actual_sales'] - results['forecasted_sales_no_promo']

    print('results',results)

    lift_summary = results.groupby('item_id').agg(
        total_actual_sales=("actual_sales", "sum"),
        total_forecasted_sales_no_promo=("forecasted_sales_no_promo", "sum"),
        total_lift_units=("sales_lift_units", "sum")
    ).reset_index()

    print('lift_summary', lift_summary)

    lift_summary['percentage_lift'] = (
        lift_summary['total_lift_units'] / lift_summary['total_forecasted_sales_no_promo'].replace(0, 1)
    ) * 100

    lift_summary.replace([float('inf'), -float('inf')], 0, inplace=True)

    print("✅ Promotion lift analysis complete.")
    print('lift_summary',lift_summary)
    return lift_summary

def analyze_sku_growth(sales_df, skus, promo_start, promo_end, before_days=90):
    """
    Performs a comprehensive before-and-after analysis of SKU sales, including revenue and ROI.
    """
    promo_start_date = pd.to_datetime(promo_start)
    promo_end_date = pd.to_datetime(promo_end)
    before_start_date = promo_start_date - pd.Timedelta(days=before_days)
    before_end_date = promo_start_date - pd.Timedelta(days=1)

    sku_sales = sales_df[sales_df['sku'].isin(skus)].copy()
    sku_sales['created_at'] = pd.to_datetime(sku_sales['created_at'], dayfirst=True, errors='coerce')
    
    # Ensure 'price' and 'disc' columns exist and are numeric
    if 'price' not in sku_sales.columns or 'disc' not in sku_sales.columns:
        st.error("Error: 'price' and 'disc' columns are required for this analysis.")
        return pd.DataFrame()
    
    sku_sales['price'] = pd.to_numeric(sku_sales['price'], errors='coerce')
    sku_sales['disc'] = pd.to_numeric(sku_sales['disc'], errors='coerce')
    sku_sales.dropna(subset=['price', 'disc'], inplace=True)
    
    sku_sales['revenue'] = sku_sales['qty'] * sku_sales['price'] * (1 - sku_sales['disc'] / 100)

    before_period_sales = sku_sales[(sku_sales['created_at'] >= before_start_date) & (sku_sales['created_at'] <= before_end_date)]
    promo_period_sales = sku_sales[(sku_sales['created_at'] >= promo_start_date) & (sku_sales['created_at'] <= promo_end_date)]

    metrics = []
    for sku in skus:
        before_sku = before_period_sales[before_period_sales['sku'] == sku]
        promo_sku = promo_period_sales[promo_period_sales['sku'] == sku]

        total_sales_before = before_sku['qty'].sum()
        total_revenue_before = before_sku['revenue'].sum()
        avg_daily_sales_before = total_sales_before / before_days if before_days > 0 else 0

        promo_duration_days = (promo_end_date - promo_start_date).days + 1
        total_sales_promo = promo_sku['qty'].sum()
        total_revenue_promo = promo_sku['revenue'].sum()
        avg_daily_sales_promo = total_sales_promo / promo_duration_days if promo_duration_days > 0 else 0
        avg_discount_promo = promo_sku['disc'].mean()

        sales_lift = total_sales_promo - (avg_daily_sales_before * promo_duration_days)
        revenue_lift = total_revenue_promo - ((total_revenue_before / before_days if before_days > 0 else 0) * promo_duration_days)
        
        # Simple ROI: Revenue Lift / (Gross Revenue * Avg Discount)
        gross_revenue_promo = (promo_sku['qty'] * promo_sku['price']).sum()
        cost_of_discount = gross_revenue_promo * (avg_discount_promo / 100)
        roi = (revenue_lift / cost_of_discount) * 100 if cost_of_discount > 0 else 0

        metrics.append({
            "SKU": sku,
            "Avg Daily Sales (Before)": avg_daily_sales_before,
            "Avg Daily Sales (Promo)": avg_daily_sales_promo,
            "Sales Lift (Units)": sales_lift,
            "Revenue Lift ($)": revenue_lift,
            "Avg Discount (%)": avg_discount_promo,
            "Discount ROI (%)": roi
        })
        
    return pd.DataFrame(metrics)
