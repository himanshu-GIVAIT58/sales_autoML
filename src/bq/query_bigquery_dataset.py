import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from google.api_core.exceptions import NotFound

# ======================= CONFIGURATION =======================
# Path to your service account JSON key file
SERVICE_ACCOUNT_FILE = "/app/service_account.json" 

# Your Google Cloud project ID
PROJECT_ID = "avnimetabase"

# Define the specific datasets you want to inspect
# These are the most relevant datasets for your sales forecasting model.
DATASETS_OF_INTEREST = [
    "partitioned",
    "giva_metabase",
    "transformed_toc",
    "guts",
    "marketing_ga4_v1_prod",
    "marketing_ads_base_prod",
    # "ebo_footfall",
    # "purchase_propensity"
]
# =============================================================

def main():
    """Connects to BigQuery and lists tables and columns for specific datasets."""
    print("Authenticating with Google Cloud...")
    try:
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
        client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
        print("✅ Authentication successful.")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return

    # Iterate through only the specified datasets of interest
    for dataset_id in DATASETS_OF_INTEREST:
        print("\n" + "="*50)
        print(f"📁 Dataset: {dataset_id}")
        print("="*50)

        try:
            tables = list(client.list_tables(dataset_id))
            if not tables:
                print("  (No tables found in this dataset)")
                continue

            # Loop through each table in the dataset
            for table in tables:
                full_table_id = f"{PROJECT_ID}.{dataset_id}.{table.table_id}"
                print(f"\n  └─ Table: {table.table_id}")

                # Get and print the schema (columns and data types)
                try:
                    table_schema = client.get_table(full_table_id).schema
                    print("    Columns:")
                    for field in table_schema:
                        print(f"      - {field.name:<30} | {field.field_type}")
                except Exception as e:
                    print(f"      (Could not read schema: {e})")

        except NotFound:
            print(f"  ❌ Dataset '{dataset_id}' not found.")
        except Exception as e:
            print(f"  ❌ An error occurred while processing dataset '{dataset_id}': {e}")


if __name__ == "__main__":
    main()
