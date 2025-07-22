import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

# ======================= CONFIGURATION =======================
SERVICE_ACCOUNT_FILE = "/app/service_account.json"
PROJECT_ID = "avnimetabase"
START_DATE = "2025-06-28"
END_DATE = "2025-06-29"
# ==============================================================

def run_sales_query():
    sql_query = f"""
    WITH
      base_orders AS (
        SELECT
          source_name,
          name,
          created_at,
          total_price,
          location_id,
          billing_address.province AS state,
          line_items,
          tags
        FROM
          givametabse.orders
        WHERE
          DATE(created_at) BETWEEN DATE('{START_DATE}') AND DATE('{END_DATE}')
          AND lower(email) NOT LIKE '%test%'
          AND lower(name) NOT LIKE '%test%'
          AND lower(tags) NOT LIKE '%replacement%'
          AND lower(tags) NOT LIKE '%warranty%'
          AND lower(tags) NOT LIKE '%rr%'
      ),
      
      pos AS (
        SELECT
          DATE(o.created_at) AS date,
          item.value.sku AS sku,
          osn.store_name AS channel,
          SUM(item.value.quantity) AS quantity,
          SUM(item.value.quantity * item.value.price) AS revenue
        FROM
          base_orders o,
          UNNEST(line_items) AS item
        LEFT JOIN UNNEST(item.value.discount_allocations) AS disc ON TRUE
        LEFT JOIN givametabse.offline_stores_new osn ON o.location_id = osn.store_id
        WHERE
          o.source_name = 'pos'
          AND item.value.price <> 0
        GROUP BY date, sku, channel
      ),

      web AS (
        SELECT
          DATE(o.created_at) AS date,
          item.value.sku AS sku,
          'Web' AS channel,
          SUM(item.value.quantity) AS quantity,
          SUM((item.value.quantity * item.value.price) - COALESCE(disc.value.amount, 0)) AS revenue
        FROM
          base_orders o,
          UNNEST(line_items) AS item
        LEFT JOIN UNNEST(item.value.discount_allocations) AS disc ON TRUE
        WHERE
          o.source_name NOT IN ('pos', '5204949', '232217378817')
          AND item.value.price <> 0
        GROUP BY date, sku
      ),

      app_cte AS (
        SELECT
          o.created_at,
          o.total_price AS initial_price,
          CASE
            WHEN tags LIKE '%app_wallet_used%' AND tags LIKE '%COD + Wallet%' THEN (
              SAFE_CAST(REPLACE(
                (SELECT t FROM UNNEST(SPLIT(tags, ',')) t WHERE t LIKE '%app_wallet_used_amount%'),
                'app_wallet_used_amount_',
                ''
              ) AS NUMERIC) + total_price
            )
            ELSE total_price
          END AS final_price,
          o.line_items
        FROM
          base_orders o
        WHERE
          source_name IN ('5204949', '232217378817')
      ),

      app AS (
        SELECT
          DATE(cte.created_at) AS date,
          item.value.sku AS sku,
          'App' AS channel,
          SUM(item.value.quantity) AS quantity,
          SUM(
            CASE
              WHEN (
                SELECT t.value.amount FROM UNNEST(item.value.discount_allocations) t LIMIT 1
              ) IS NOT NULL THEN (
                (item.value.price / initial_price * final_price) * item.value.quantity - (
                  SELECT t.value.amount FROM UNNEST(item.value.discount_allocations) t LIMIT 1
                )
              )
              ELSE (item.value.price / initial_price * final_price) * item.value.quantity
            END
          ) AS revenue
        FROM
          app_cte cte,
          UNNEST(cte.line_items) AS item
        WHERE item.value.price <> 0
        GROUP BY date, sku
      ),

      all_sales AS (
        SELECT * FROM pos
        UNION ALL
        SELECT * FROM app
        UNION ALL
        SELECT * FROM web
      )

    SELECT
      date,
      sku,
      channel,
      SUM(quantity) AS quantity,
      SUM(revenue) AS revenue
    FROM
      all_sales
    WHERE
      sku NOT LIKE 'GW%'
      AND sku NOT LIKE 'GD%'
      AND sku NOT LIKE 'FREE%'
      AND sku NOT LIKE 'PX%'
      AND sku NOT LIKE '%+%'
      AND sku NOT LIKE 'PM%'
      AND sku NOT LIKE 'JO%'
      AND sku NOT LIKE 'CA%'
      AND sku NOT LIKE 'PR%'
      AND sku NOT LIKE 'ID%'
      AND sku NOT LIKE 'UT%'
    GROUP BY date, sku, channel
    ORDER BY date DESC, sku;
    """

    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
    client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
    df = client.query(sql_query).to_dataframe()
    return df

def main():
    print(f"Fetching daily SKU sales data from {START_DATE} to {END_DATE}...")
    df = run_sales_query()
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
