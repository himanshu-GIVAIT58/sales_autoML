from pymongo import MongoClient
from dotenv import load_dotenv
import os
from src import dbConnect

load_dotenv()


MONGO_URI = dbConnect.connection_uri
client = dbConnect.client

try:
    print("Databases:", client.list_database_names())
    db = client["sales_automl"]
    collections = db.list_collection_names()
    print("Collections in 'sales_automl':", collections)
    for collection_name in collections:
        collection = db[collection_name]
        sample_document = collection.find_one()  
        if sample_document:
            print(f"Columns in '{collection_name}':", list(sample_document.keys()))
        else:
            print(f"'{collection_name}' is empty.")
    print("Connection successful!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
finally:
    client.close()
