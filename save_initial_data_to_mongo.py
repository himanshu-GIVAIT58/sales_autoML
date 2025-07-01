import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "sales_automl")

# Map your local CSV files to MongoDB collection names
DATA_FILES = {
    "inventory_recommendations.csv": "inventory_recommendations",
}

DATA_DIR = os.path.join(os.path.dirname(__file__),'./')

def save_csv_to_mongo(csv_path, collection_name):
    print(f"Processing {csv_path} -> {collection_name}")
    df = pd.read_csv(csv_path)
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    collection = db[collection_name]
    collection.delete_many({})  # Clear old data
    if not df.empty:
        collection.insert_many(df.to_dict("records"))
        print(f"Inserted {len(df)} records into '{collection_name}' collection.")
    else:
        print(f"No data found in {csv_path}. Skipped.")
    client.close()

def main():
    for filename, collection_name in DATA_FILES.items():
        csv_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(csv_path):
            save_csv_to_mongo(csv_path, collection_name)
        else:
            print(f"File not found: {csv_path}")

if __name__ == "__main__":
    main()
