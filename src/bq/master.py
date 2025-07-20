import os
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

# ======================= CONFIGURATION =======================
# Path to your service account JSON key file
SERVICE_ACCOUNT_FILE = "/app/service_account.json" 

# Your Google Cloud project ID
PROJECT_ID = "avnimetabase"

# Define where to save the final dataset locally
OUTPUT_FOLDER = "data"
OUTPUT_FILENAME = "master_sku_sales_dataset.parquet"
# =============================================================

# Ensure the output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)


# The final SQL query to generate the master dataset
sql_query = """
-- =================================== CONFIGURATION ===================================
DECLARE forecast_start_date DATE DEFAULT '2024-01-01';
DECLARE forecast_end_date DATE DEFAULT CURRENT_DATE();
-- =====================================================================================

WITH
  -- Step 1: Create a complete calendar of dates for the training period.
  date_range AS (
    SELECT forecast_date
    FROM UNNEST(GENERATE_DATE_ARRAY(forecast_start_date, forecast_end_date, INTERVAL 1 DAY)) AS forecast_date
  ),

  -- Step 2: Get all unique SKUs and their static attributes.
  all_skus AS (
    SELECT DISTINCT
      p.sku AS sku_id,
      p.category,
      p.product_type,
      p.vendor,
      p.tags,
      oi.price AS mrp
    FROM `giva_metabase.products` p
    LEFT JOIN `partitioned.order_items` oi ON p.sku = oi.sku
    WHERE p.sku IS NOT NULL
  ),

  -- Step 3: Create a dedicated CTE for important holidays and events.
  holidays_and_events AS (
    SELECT event_date, event_name
    FROM UNNEST([
      STRUCT(DATE(2024, 1, 26) AS event_date, 'Republic Day' AS event_name),
      STRUCT(DATE(2024, 2, 14), 'Valentines Day'),
      STRUCT(DATE(2024, 3, 25), 'Holi'),
      STRUCT(DATE(2025, 8, 15), 'Independence Day'),
      STRUCT(DATE(2025, 10, 21), 'Diwali'),
      STRUCT(DATE(2025, 12, 25), 'Christmas Day')
    ])
  ),

  -- Step 4: Create the base scaffold.
  scaffold AS (
    SELECT d.forecast_date, s.*
    FROM date_range d
    CROSS JOIN all_skus s
  ),

  -- Step 5: Aggregate daily sales.
  daily_sku_sales AS (
    SELECT
      DATE(created_at) AS sale_date,
      sku AS sku_id,
      SUM(quantity) AS quantity_sold,
      AVG(price) AS average_selling_price
    FROM `partitioned.order_items`
    GROUP BY 1, 2
  ),

  -- Step 6: Aggregate daily inventory levels.
  daily_inventory AS (
    SELECT
      DATE(snapshot_date) AS inv_date,
      sku AS sku_id,
      LAST_VALUE(SUM(stock_on_hand) IGNORE NULLS) OVER (PARTITION BY sku ORDER BY DATE(snapshot_date)) as daily_stock_level
    FROM `transformed_toc.warehouse_inventory`
    GROUP BY 1, 2
  ),

  -- Step 7: Aggregate daily page views.
  daily_page_views AS (
    SELECT
      event_date,
      sku AS sku_id,
      COUNT(*) AS page_views
    FROM `marketing_ga4_v1_prod.page_views`
    WHERE sku IS NOT NULL
    GROUP BY 1, 2
  ),

  -- Step 8: Join all data sources.
  combined_data AS (
    SELECT
      s.forecast_date,
      s.sku_id,
      s.category,
      s.product_type,
      s.vendor,
      s.tags,
      s.mrp,
      COALESCE(ds.quantity_sold, 0) AS total_quantity_sold,
      COALESCE(ds.average_selling_price, s.mrp) as selling_price,
      (s.mrp - COALESCE(ds.average_selling_price, s.mrp)) / NULLIF(s.mrp, 0) as discount_percentage,
      COALESCE(di.daily_stock_level, 0) as stock_level,
      COALESCE(dpv.page_views, 0) as product_page_views,
      EXTRACT(DAYOFWEEK FROM s.forecast_date) AS day_of_week,
      EXTRACT(DAY FROM s.forecast_date) AS day_of_month,
      EXTRACT(WEEK FROM s.forecast_date) AS week_of_year,
      EXTRACT(MONTH FROM s.forecast_date) AS month_of_year,
      h.event_name IS NOT NULL AS is_holiday_or_event,
      h.event_name
    FROM scaffold s
    LEFT JOIN daily_sku_sales ds ON s.forecast_date = ds.sale_date AND s.sku_id = ds.sku_id
    LEFT JOIN daily_inventory di ON s.forecast_date = di.inv_date AND s.sku_id = di.sku_id
    LEFT JOIN daily_page_views dpv ON s.forecast_date = dpv.event_date AND s.sku_id = dpv.sku_id
    LEFT JOIN holidays_and_events h ON s.forecast_date = h.event_date
  )

-- Step 9: Final selection with lag and rolling window features.
SELECT
  *,
  LAG(total_quantity_sold, 7) OVER (PARTITION BY sku_id ORDER BY forecast_date) AS sales_lag_7_days,
  LAG(total_quantity_sold, 30) OVER (PARTITION BY sku_id ORDER BY forecast_date) AS sales_lag_30_days,
  AVG(total_quantity_sold) OVER (PARTITION BY sku_id ORDER BY forecast_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS sales_rolling_avg_30_days,
  SUM(total_quantity_sold) OVER (PARTITION BY sku_id ORDER BY forecast_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS sales_rolling_sum_30_days,
  STDDEV(total_quantity_sold) OVER (PARTITION BY sku_id ORDER BY forecast_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS sales_rolling_stddev_30_days
FROM combined_data
ORDER BY sku_id, forecast_date;
"""

def main():
    """Fetches the master dataset from BigQuery and saves it locally."""
    print("Authenticating with Google Cloud...")
    try:
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
        client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
        print("✅ Authentication successful.")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return

    try:
        print("\nExecuting BigQuery query to generate master dataset...")
        print("(This may take several minutes depending on the data size)")
        df_master = client.query(sql_query).to_dataframe()
        print(f"✅ Query successful. Fetched {len(df_master)} rows.")

        print(f"\nSaving dataset to '{output_path}'...")
        df_master.to_parquet(output_path, index=False)
        # --- Alternative: To save as CSV, uncomment the line below ---
        # df_master.to_csv(output_path.replace('.parquet', '.csv'), index=False)
        
        print("\n🎉 Master dataset successfully created and saved locally!")

    except Exception as e:
        print(f"❌ An error occurred during the process: {e}")

if __name__ == "__main__":
    main()
