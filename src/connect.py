from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get MONGO_URI from .env
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

try:
    # List all databases
    print("Databases:", client.list_database_names())

    # Access the 'sales_automl' database
    db = client["sales_automl"]

    # List all collections in the 'sales_automl' database
    collections = db.list_collection_names()
    print("Collections in 'sales_automl':", collections)

    # Show column names (keys) for each collection
    for collection_name in collections:
        collection = db[collection_name]
        sample_document = collection.find_one()  # Get a sample document
        if sample_document:
            print(f"Columns in '{collection_name}':", list(sample_document.keys()))
        else:
            print(f"'{collection_name}' is empty.")

    print("Connection successful!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
finally:
    client.close()
