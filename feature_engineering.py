# feature_engineering.py
"""
Contains all functions for data preparation and feature engineering.
"""
import pandas as pd

# --- Feature Creation Functions ---

def create_jewelry_features(df):
    """Adds jewelry-specific seasonal features."""
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_wedding_season'] = df['month'].isin([10, 11, 12, 1, 2]).astype(int)
    df['is_diwali_period'] = df['month'].isin([10, 11]).astype(int)
    df['is_valentine_month'] = (df['month'] == 2).astype(int)
    return df

def create_price_elasticity_features(df):
    """Adds features related to promotions and discounts."""
    df['is_on_promotion'] = (df['disc'] > 0).astype(int)
    high_discount_thresholds = df[df['disc'] > 0].groupby('sku')['disc'].quantile(0.75)
    df['high_discount_threshold'] = df['sku'].map(high_discount_thresholds)
    df['high_discount_threshold'].fillna(0, inplace=True)
    df['is_high_discount'] = (df['disc'] > df['high_discount_threshold']).astype(int)
    df.drop(columns=['high_discount_threshold'], inplace=True)
    return df

def create_inventory_features(df):
    """Adds features related to inventory levels and stockouts."""
    df['warehouse_qty'] = df.groupby('sku')['warehouse_qty'].transform(lambda x: x.ffill().bfill())
    df['warehouse_qty'].fillna(1, inplace=True) # Assume in stock if no data
    df['was_stocked_out'] = (df['warehouse_qty'] <= 0).astype(int)
    df['stockout_days_last_7'] = df.groupby('sku')['was_stocked_out'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).sum())
    return df

def create_trend_features(df):
    """Adds features to capture sales momentum and trend, preventing data leakage."""
    df_sorted = df.sort_values(['sku', 'timestamp'])
    for window in [7, 14, 30]:
        df_sorted[f'sales_ma_{window}'] = df_sorted.groupby('sku')['target'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
    df_sorted.fillna(0, inplace=True)
    # Estimate potential lost sales based on recent stockouts and sales trend
    df_sorted['potential_lost_sales'] = df_sorted['stockout_days_last_7'] * df_sorted['sales_ma_30']
    return df_sorted

def create_hierarchy_features(df):
    """Creates hierarchical features for the model to learn from."""
    df['brand_category'] = 'GIVA_' + df['category']
    df['category_channel'] = df['category'] + '_' + df['channel']
    return df

# --- Main Data Preparation Pipeline ---

def prepare_data(source_data, inventory_data, holidays_data, max_skus=None):
    """
    Executes the full data preparation and feature engineering pipeline.
    """
    print("\nPreparing and regularizing data...")

    # MEMORY EFFICIENCY: Filter to top SKUs before heavy processing
    if max_skus is not None:
        print(f"Identifying top {max_skus} SKUs to reduce memory usage...")
        sku_sales_totals = source_data.groupby('sku')['qty'].sum().sort_values(ascending=False)
        top_n_skus = sku_sales_totals.nlargest(max_skus).index
        source_data = source_data[source_data["sku"].isin(top_n_skus)]
        inventory_data = inventory_data[inventory_data["sku"].isin(top_n_skus)]
        print(f"Data now contains only the top {len(top_n_skus)} SKUs.")

    # a. Prepare sales data
    df_sales = source_data[['created_at', 'sku', 'qty', 'category', 'gender', 'disc', 'Channel']].copy()
    df_sales['channel'] = df_sales['Channel'].str.strip().fillna('Unknown')
    df_sales.rename(columns={"created_at": "timestamp", "qty": "target"}, inplace=True)
    df_sales["timestamp"] = pd.to_datetime(df_sales["timestamp"])
    df_sales["target"] = pd.to_numeric(df_sales["target"], errors='coerce').fillna(0)
    df_sales["disc"] = pd.to_numeric(df_sales["disc"], errors='coerce').fillna(0)

    # b. Prepare inventory data
    df_inventory = inventory_data[['date', 'sku', 'wh']].copy()
    df_inventory.rename(columns={"date": "timestamp", "wh": "warehouse_qty"}, inplace=True)
    df_inventory["timestamp"] = pd.to_datetime(df_inventory["timestamp"], dayfirst=True)
    df_inventory["warehouse_qty"] = pd.to_numeric(df_inventory["warehouse_qty"], errors='coerce').fillna(0)
    df_inventory_daily = df_inventory.groupby(["sku", "timestamp"]).agg(warehouse_qty=("warehouse_qty", "sum")).reset_index()

    # c. Prepare holiday data
    df_holidays = holidays_data[['Date']].copy()
    df_holidays.rename(columns={'Date': 'timestamp'}, inplace=True)
    df_holidays['timestamp'] = pd.to_datetime(df_holidays['timestamp'].astype(str).str.split('T').str[0])
    df_holidays['is_holiday'] = 1
    df_holidays = df_holidays.drop_duplicates(subset=['timestamp'])

    # d. Merge and engineer features
    df = pd.merge(df_sales, df_inventory_daily, on=["sku", "timestamp"], how="left")
    df = create_jewelry_features(df)
    df = create_price_elasticity_features(df)
    df = create_inventory_features(df)
    df = create_trend_features(df)
    df = create_hierarchy_features(df)

    # Add placeholder for gold prices (can be enhanced later)
    df['gold_price_change'] = 0
    df['gold_price_ma_7'] = 0

    # e. Final aggregation and regularization to create a daily time series
    df['item_id'] = df['sku'] + "_" + df['channel']
    static_features_base = df[['sku', 'category', 'gender', 'brand_category', 'category_channel']].drop_duplicates(subset=['sku'])
    static_features_base = pd.get_dummies(static_features_base, columns=['category', 'gender', 'brand_category', 'category_channel'])

    agg_cols = [c for c in df.columns if c not in ['sku', 'channel', 'target', 'timestamp', 'item_id', 'disc', 'Channel', 'category', 'gender', 'warehouse_qty']]
    agg_dict = {col: "first" for col in agg_cols}
    agg_dict['target'] = "sum"
    df_daily = df.groupby(["item_id", "sku", "channel", "timestamp"]).agg(agg_dict).reset_index()

    # Regularize data to ensure one row per day per item
    all_items_channel = df_daily["item_id"].unique()
    min_date = df_daily["timestamp"].min()
    max_date = df_daily["timestamp"].max()
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    full_template = pd.MultiIndex.from_product([all_items_channel, date_range], names=["item_id", "timestamp"]).to_frame(index=False)
    regularized_data = pd.merge(full_template, df_daily, on=["item_id", "timestamp"], how="left")

    # Re-create SKU and channel from item_id and forward-fill missing features
    temp_sku_channel = regularized_data['item_id'].str.split('_', n=1, expand=True)
    regularized_data['sku'] = temp_sku_channel[0]
    regularized_data['channel'] = temp_sku_channel[1]
    regularized_data = pd.merge(regularized_data, df_holidays, on="timestamp", how="left")

    for col in agg_cols + ['sku', 'channel']:
        regularized_data[col] = regularized_data.groupby('item_id')[col].transform(lambda x: x.ffill().bfill())
    
    regularized_data.fillna(0, inplace=True)
    print("Data preparation and feature engineering complete.")

    return regularized_data, static_features_base
