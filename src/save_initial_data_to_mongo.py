import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import json


load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/")  
MONGO_DB = os.getenv("MONGO_DB", "sales_automl")


DATA_FILES = {
    "inventory_recommendations.csv": "inventory_recommendations_20250710_113711",
    # "indian_holidays.csv": "holidays_data",
    # "query_result_2025-05-28T18_02_43.550629445+05_30 (1).csv": "query_result",
    # "store_count.csv": "store_count",
    # "sales_data_complete___daily_drill_down_2025-05-29T12_37_43.113222731+05_30 (1).csv": "sales_data",
    # "mongo_data.json": "all_collections",  
}


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
print(DATA_DIR)
def save_file_to_mongo(file_path, collection_name):
    print(f"Processing {file_path} -> {collection_name}")
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collection = db[collection_name]

        
        collection.delete_many({})
        print(f"Cleared existing data in '{collection_name}' collection.")

        
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            if not df.empty:
                collection.insert_many(df.to_dict("records"))
                print(f"Inserted {len(df)} records into '{collection_name}' collection.")
            else:
                print(f"No data found in {file_path}. Skipped.")

        
        elif file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)
                if isinstance(data, list) and data:  
                    collection.insert_many(data)
                    print(f"Inserted {len(data)} records into '{collection_name}' collection.")
                elif isinstance(data, dict):  
                    for sub_collection, documents in data.items():
                        sub_collection_ref = db[sub_collection]
                        sub_collection_ref.delete_many({})
                        if isinstance(documents, list) and documents:
                            sub_collection_ref.insert_many(documents)
                            print(f"Inserted {len(documents)} records into '{sub_collection}' collection.")
                        else:
                            print(f"No data found in '{sub_collection}' collection. Skipped.")
                else:
                    print(f"No valid data found in {file_path}. Skipped.")

        else:
            print(f"Unsupported file format: {file_path}. Skipped.")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    finally:
        client.close()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/")
MONGO_DB = os.getenv("MONGO_DB", "sales_automl")

def delete_inventory_recommendations_collections():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    collections = db.list_collection_names()
    deleted = []
    for name in collections:
        if name.startswith("inventory_recommendations_") and name != "inventory_recommendations":
            db.drop_collection(name)
            deleted.append(name)
    client.close()
    print(f"Deleted collections: {deleted}")


def main():
    delete_inventory_recommendations_collections()
    for filename, collection_name in DATA_FILES.items():
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(file_path):
            save_file_to_mongo(file_path, collection_name)
        else:
            print(f"File not found: {file_path}. Skipping.")

if __name__ == "__main__":
    main()
