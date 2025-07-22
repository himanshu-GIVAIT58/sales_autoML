import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

# ======================= CONFIGURATION =======================
SERVICE_ACCOUNT_FILE = "/app/service_account.json"
PROJECT_ID = "avnimetabase"
# =============================================================

def run_fixed_range_query():
    sql_query = """
    WITH
      web AS (
        SELECT
          item.value.sku,
          SUM(item.value.quantity) AS total_quantity,
          SUM((item.value.quantity * item.value.price) - COALESCE(disc.value.amount, 0)) AS rev
        FROM
          givametabse.orders,
          UNNEST (line_items) AS item
          LEFT JOIN UNNEST (item.value.discount_allocations) AS disc ON TRUE
        WHERE
          source_name = 'web'
          AND created_at BETWEEN '2025-05-29' AND '2025-06-29'
          AND item.value.price <> 0
        GROUP BY 1
      ),
      cte AS (
        SELECT
          total_price AS initial_price,
          CASE
            WHEN tags LIKE '%app_wallet_used%' AND tags LIKE '%COD + Wallet%' THEN (
              SAFE_CAST(REPLACE((SELECT t FROM UNNEST(SPLIT(tags, ',')) t WHERE t LIKE '%app_wallet_used_amount%'), 'app_wallet_used_amount_', '') AS NUMERIC) + total_price
            )
            ELSE total_price
          END AS final_price,
          line_items
        FROM
          givametabse.orders
        WHERE
          source_name IN ('android', 'ios')
          AND created_at BETWEEN '2025-05-29' AND '2025-06-29'
      ),
      app AS (
        SELECT
          item.value.sku,
          SUM(item.value.quantity) AS total_quantity,
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
          ) AS rev
        FROM
          cte,
          UNNEST (line_items) AS item
        WHERE item.value.price <> 0
        GROUP BY 1
      )
    SELECT
      sku,
      source,
      SUM(total_quantity) AS qty,
      SUM(rev) AS revenue
    FROM (
      SELECT sku, total_quantity, rev, 'web' AS source FROM web
      UNION ALL
      SELECT sku, total_quantity, rev, 'app' AS source FROM app
    )
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
    GROUP BY 1, 2
    ORDER BY 3 DESC
    """  # <== LIMIT REMOVED

    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
    client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
    return client.query(sql_query).to_dataframe()

def main():
    print("\nFetching full SKU sales data from 29 May 2025 to 29 June 2025 (no LIMIT)...")
    df = run_fixed_range_query()
    print(f"Total rows: {len(df)}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
