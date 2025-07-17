import pandas as pd
from typing import Tuple, List, Optional

def create_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds time-based seasonal features relevant to jewelry sales."""
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_wedding_season'] = df['month'].isin([10, 11, 12, 1, 2]).astype(int)
    df['is_diwali_period'] = df['month'].isin([10, 11]).astype(int)
    df['is_valentine_month'] = (df['month'] == 2).astype(int)
    return df

def add_feedback_features(main_df: pd.DataFrame, feedback_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features from user feedback and merges them into the main dataframe.

    Args:
        main_df: The main dataframe with sales and other features.
        feedback_df: The dataframe loaded from the 'feedback_data' collection.

    Returns:
        The main dataframe with added feedback features.
    """
    if feedback_df.empty:
        print("   -> No feedback data to process. Skipping feedback feature engineering.")
        main_df['feedback_score_30d_avg'] = 0
        return main_df

    print("   -> Engineering features from user feedback...")
    
    # Standardize column names
    feedback_df = feedback_df.rename(columns={'selected_sku': 'item_id'})
    feedback_df['timestamp'] = pd.to_datetime(feedback_df['timestamp'])

    # 1. Convert categorical feedback into a numerical score
    feedback_map = {
        'Forecast too high': -1,
        'Good forecast': 0,
        'Forecast too low': 1,
        'Stockout occurred': 1.5 # Higher weight for stockouts
    }
    feedback_df['feedback_score'] = feedback_df['feedback'].map(feedback_map).fillna(0)

    # 2. Create a rolling feature for each item
    # This captures recent feedback trends for each SKU over a 30-day window
    feedback_df = feedback_df.sort_values(by=['item_id', 'timestamp'])
    feedback_df_agg = feedback_df.set_index('timestamp').groupby('item_id')['feedback_score'].rolling('30D').mean().reset_index()
    feedback_df_agg = feedback_df_agg.rename(columns={'feedback_score': 'feedback_score_30d_avg'})
    
    # 3. Merge the feedback feature into the main dataframe
    main_df = pd.merge(main_df, feedback_df_agg, on=['item_id', 'timestamp'], how='left')
    
    # Forward-fill to propagate the last known feedback score, then fill remaining NaNs with 0
    main_df['feedback_score_30d_avg'] = main_df.groupby('item_id')['feedback_score_30d_avg'].ffill().fillna(0)
    
    print("   -> ✅ Feedback features successfully added.")
    return main_df
    
def create_price_elasticity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds features related to promotions and high-value discounts."""
    df['is_on_promotion'] = (df['disc'] > 0).astype(int)
    high_discount_thresholds = df[df['disc'] > 0].groupby('sku')['disc'].quantile(0.75)
    df['high_discount_threshold'] = df['sku'].map(high_discount_thresholds).fillna(0)
    df['is_high_discount'] = (df['disc'] > df['high_discount_threshold']).astype(int)
    df = df.drop(columns=['high_discount_threshold'])
    return df

def create_inventory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds features related to inventory levels and stockouts."""
    
    df['was_stocked_out'] = (df['warehouse_qty'] <= 0).astype(int)
    stockout_rolling_sum = df.groupby('sku')['was_stocked_out'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).sum())
    df['stockout_days_last_7'] = stockout_rolling_sum.fillna(0)
    return df

def create_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds features to capture sales momentum and trend."""
    df = df.sort_values(['sku', 'timestamp'])
    for window in [7, 14, 30]:
        rolling_mean = df.groupby('sku')['target'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        df[f'sales_ma_{window}'] = rolling_mean.fillna(0)
    df['potential_lost_sales'] = df['stockout_days_last_7'] * df['sales_ma_30']
    df['potential_lost_sales'] = df['potential_lost_sales'].fillna(0)
    return df

def generate_static_features(df: pd.DataFrame, all_training_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Generates a one-hot encoded static features DataFrame from a base DataFrame."""
    df_static = df.copy()
    if 'category' not in df_static.columns: df_static['category'] = 'unknown'
    if 'gender' not in df_static.columns: df_static['gender'] = 'unknown'
    if 'channel' not in df_static.columns: df_static['channel'] = 'Online'
    df_static['brand_category'] = 'GIVA_' + df_static['category'].astype(str)
    df_static['category_channel'] = df_static['category'].astype(str) + '_' + df_static['channel'].astype(str)
    static_cols_to_encode = ['category', 'gender', 'brand_category', 'category_channel']
    static_base = df_static[['item_id'] + static_cols_to_encode].drop_duplicates(subset=['item_id'])
    static_features_df = pd.get_dummies(static_base.set_index('item_id'), columns=static_cols_to_encode, dtype=float)
    if all_training_columns:
        static_features_df = static_features_df.reindex(columns=all_training_columns, fill_value=0)
    return static_features_df

def prepare_data(source_data: pd.DataFrame, inventory_data: pd.DataFrame, holidays_data: pd.DataFrame, max_skus: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Executes the full data preparation and feature engineering pipeline."""
    print("\nPreparing and regularizing data...")

    
    if max_skus is not None:
        print(f"Identifying top {max_skus} SKUs to reduce memory usage...")
        sku_sales_totals = source_data.groupby('sku')['qty'].sum().sort_values(ascending=False)
        top_n_skus = sku_sales_totals.nlargest(max_skus).index
        source_data = source_data[source_data["sku"].isin(top_n_skus)].copy()
        inventory_data = inventory_data[inventory_data["sku"].isin(top_n_skus)].copy()
        print(f"Data now contains only the top {len(top_n_skus)} SKUs.")

    
    sales_df = source_data[['created_at', 'sku', 'qty', 'category', 'gender', 'disc', 'Channel']].copy()
    sales_df['channel'] = sales_df['Channel'].str.strip().fillna('Unknown')
    sales_df.rename(columns={"created_at": "timestamp", "qty": "target"}, inplace=True)
    sales_df["timestamp"] = pd.to_datetime(sales_df["timestamp"], dayfirst=True, errors='coerce')
    sales_df.dropna(subset=['timestamp'], inplace=True)
    sales_df["target"] = pd.to_numeric(sales_df["target"], errors='coerce').fillna(0)
    sales_df["disc"] = pd.to_numeric(sales_df["disc"], errors='coerce').fillna(0)

    inventory_df = inventory_data[['date', 'sku', 'wh']].copy()
    inventory_df.rename(columns={"date": "timestamp", "wh": "warehouse_qty"}, inplace=True)
    inventory_df["timestamp"] = pd.to_datetime(inventory_df["timestamp"], dayfirst=True, errors='coerce')
    inventory_daily_df = inventory_df.groupby(["sku", "timestamp"]).agg(warehouse_qty=("warehouse_qty", "sum")).reset_index()

    holidays_df = holidays_data[['Date']].copy()
    holidays_df.rename(columns={'Date': 'timestamp'}, inplace=True)
    holidays_df['timestamp'] = pd.to_datetime(holidays_df['timestamp'].astype(str).str.split('T').str[0])
    holidays_df['is_holiday'] = 1
    holidays_df = holidays_df.drop_duplicates(subset=['timestamp'])

    
    df = pd.merge(sales_df, inventory_daily_df, on=["sku", "timestamp"], how="left")

    
    
    df['warehouse_qty'] = pd.to_numeric(df['warehouse_qty'], errors='coerce').fillna(1)
    
    
    df = create_seasonal_features(df)
    df = create_price_elasticity_features(df)
    df = create_inventory_features(df) 
    df = create_trend_features(df)
    
    df['gold_price_change'] = 0.0
    df['gold_price_ma_7'] = 0.0
    df['item_id'] = df['sku'].astype(str) + "_" + df['channel'].astype(str)

    
    print("Generating static features...")
    static_features_df = generate_static_features(df)
    print(f"✅ Generated static features. Column names: {list(static_features_df.columns)}")

    
    agg_cols = [c for c in df.columns if c not in ['sku', 'channel', 'target', 'timestamp', 'item_id', 'disc', 'Channel', 'category', 'gender', 'brand_category', 'category_channel']]
    agg_dict = {col: "first" for col in agg_cols}
    agg_dict['target'] = "sum"
    df_daily = df.groupby(["item_id", "sku", "channel", "timestamp"]).agg(agg_dict).reset_index()

    
    all_items = df_daily["item_id"].unique().tolist()
    date_range = pd.date_range(start=df_daily["timestamp"].min(), end=df_daily["timestamp"].max(), freq='D')
    multi_index = pd.MultiIndex.from_product([all_items, list(date_range)], names=["item_id", "timestamp"])
    regularized_data = pd.DataFrame(index=multi_index).reset_index()
    regularized_data = pd.merge(regularized_data, df_daily, on=["item_id", "timestamp"], how="left")

    
    print("   -> Propagating static attributes and filling data gaps...")
    id_to_static_map = df[['item_id', 'sku', 'channel']].drop_duplicates()
    regularized_data = regularized_data.drop(columns=['sku', 'channel'], errors='ignore')
    regularized_data = pd.merge(regularized_data, id_to_static_map, on='item_id', how='left')

    seasonal_cols = [col for col in regularized_data.columns if any(x in col for x in ['month', 'day_of_week', 'is_weekend', 'is_wedding_season', 'is_diwali_period', 'is_valentine_month'])]
    if seasonal_cols:
        regularized_data[seasonal_cols] = regularized_data.groupby('item_id')[seasonal_cols].transform(lambda x: x.ffill().bfill())

    regularized_data = pd.merge(regularized_data, holidays_df, on="timestamp", how="left")
    regularized_data = regularized_data.fillna(0)
    
    print("Data preparation and feature engineering complete.")
    return regularized_data, static_features_df

def prepare_data_for_analysis(source_data: pd.DataFrame, holidays_data: pd.DataFrame, max_skus: Optional[int] = None) -> pd.DataFrame:
    print("\nPreparing data for analysis...")
    if max_skus is not None:
        sku_sales_totals = source_data.groupby('sku')['qty'].sum().sort_values(ascending=False)
        top_n_skus = sku_sales_totals.nlargest(max_skus).index
        source_data = source_data[source_data["sku"].isin(top_n_skus)].copy()
    sales_df = source_data[['created_at', 'sku', 'qty', 'category', 'gender', 'disc', 'Channel']].copy()
    sales_df.rename(columns={"created_at": "timestamp", "qty": "target"}, inplace=True)
    sales_df["timestamp"] = pd.to_datetime(sales_df["timestamp"], dayfirst=True, errors='coerce').dropna()
    sales_df["target"] = pd.to_numeric(sales_df["target"], errors='coerce').fillna(0)
    sales_df["disc"] = pd.to_numeric(sales_df["disc"], errors='coerce').fillna(0)
    df = create_seasonal_features(sales_df)
    df = create_price_elasticity_features(df)
    if 'channel' not in df.columns:
        df['channel'] = 'Unknown'
    df['item_id'] = df['sku'].astype(str) + "_" + df['channel'].fillna('Unknown').astype(str)

    
    agg_dict = {
        'target': 'sum',
        'disc': 'mean', 
        'is_on_promotion': 'max', 
        'is_high_discount': 'max'
    }
    df_daily = df.groupby(["item_id", "sku", "timestamp"]).agg(agg_dict).reset_index()

    
    all_items = df_daily["item_id"].unique().tolist()
    date_range = pd.date_range(start=df_daily["timestamp"].min(), end=df_daily["timestamp"].max(), freq='D').tolist()
    multi_index = pd.MultiIndex.from_product([all_items, date_range], names=["item_id", "timestamp"])
    regularized_data = pd.DataFrame(index=multi_index).reset_index()
    regularized_data = pd.merge(regularized_data, df_daily, on=["item_id", "timestamp"], how="left")
    
    
    id_to_sku_map = df[['item_id', 'sku']].drop_duplicates()
    regularized_data = pd.merge(regularized_data, id_to_sku_map, on='item_id', how='left')
    
    holidays_df = holidays_data[['Date']].copy()
    holidays_df.rename(columns={'Date': 'timestamp'}, inplace=True)
    holidays_df['timestamp'] = pd.to_datetime(holidays_df['timestamp'].astype(str).str.split('T').str[0])
    holidays_df['is_holiday'] = 1
    
    regularized_data = pd.merge(regularized_data, holidays_df, on="timestamp", how="left")
    
    
    ffill_cols = [col for col in regularized_data.columns if col not in ['target', 'disc']]
    regularized_data[ffill_cols] = regularized_data.groupby('item_id')[ffill_cols].transform(lambda x: x.ffill().bfill())
    regularized_data.fillna(0, inplace=True)
    
    print("✅ Analysis data preparation complete.")
    return regularized_data
