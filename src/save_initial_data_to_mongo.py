import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/")  # Use the Docker service name 'mongodb'
MONGO_DB = os.getenv("MONGO_DB", "sales_automl")

# Map your local CSV files to MongoDB collection names
DATA_FILES = {
    "Expected Delivery Date - Base.csv": "expected_delivery",
    "indian_holidays.csv": "holidays_data",
    "query_result_2025-05-28T18_02_43.550629445+05_30 (1).csv": "query_result",
    "store_count.csv": "store_count",
    "sales_data_complete___daily_drill_down_2025-05-29T12_37_43.113222731+05_30 (1).csv": "sales_data",
}

# Define the directory where the CSV files are located
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))

def save_csv_to_mongo(csv_path, collection_name):
    """
    Reads a CSV file and inserts its data into a MongoDB collection.
    """
    print(f"Processing {csv_path} -> {collection_name}")
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_path)
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collection = db[collection_name]

        # Clear old data in the collection
        collection.delete_many({})
        print(f"Cleared existing data in '{collection_name}' collection.")

        # Insert new data if the DataFrame is not empty
        if not df.empty:
            collection.insert_many(df.to_dict("records"))
            print(f"Inserted {len(df)} records into '{collection_name}' collection.")
        else:
            print(f"No data found in {csv_path}. Skipped.")
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
    finally:
        client.close()

def main():
    """
    Iterates over the defined CSV files and uploads their data to MongoDB.
    """
    for filename, collection_name in DATA_FILES.items():
        csv_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(csv_path):
            save_csv_to_mongo(csv_path, collection_name)
        else:
            print(f"File not found: {csv_path}. Skipping.")

if __name__ == "__main__":
    main()
