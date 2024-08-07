from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import FastAPI, Depends, Request
from urllib.parse import urlparse, unquote
import asyncio
from fastapi import FastAPI, Query
from pymongo import MongoClient
from typing import Dict, Any
import csv

app = FastAPI()


async def process_csv(file_contents: bytes) -> pd.DataFrame:
    # Read the CSV file using pandas
    df = pd.read_csv(file_contents)
    # Process the CSV data (for example, we'll just return the dataframe here)
    return df

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    # Check if the uploaded file is a CSV
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Uploaded file is not a CSV.")

    # Read the CSV file
    contents = await file.read()
    # Process the file
    try:
        df = await process_csv(contents)
        return {"data": df.to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing CSV file")
    

@app.post("/mongo/{db_name}/{collection_name}/create/")
async def create_db_collection_data(db_name: str, collection_name: str, db_url: str, csv_file: UploadFile = File(...)):
    """
    Endpoint to create a new MongoDB database and a collection within it, and populate the collection with data from a CSV file.
    """
    client = MongoClient(db_url)
    try:
        # Check if the database already exists
        if db_name in client.list_database_names():
            raise HTTPException(status_code=400, detail=f"Database '{db_name}' already exists.")

        # Create the new database and collection
        db = client[db_name]
        collection = db[collection_name]

        # Read the CSV file and insert data into the collection
        csv_content = await csv_file.read()
        csv_data = csv.reader(csv_content.decode('utf-8').splitlines())
        headers = next(csv_data)
        for row in csv_data:
            data = dict(zip(headers, row))
            collection.insert_one(data)

        return {"message": f"Database '{db_name}' and collection '{collection_name}' created successfully and populated with data from the CSV file."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating database: {str(e)}")


    ####

def connect_to_mongodb(connection_string: str):
    """
    Connect to MongoDB and return the client object.
    """
    client = MongoClient(connection_string)
    db = client.admin
    return client, db


@app.get("/mongo/databases", response_model=Dict[str, Dict[str, Any]])
async def list_databases(connection_string: str = Query("mongodb://localhost:27017/", description="MongoDB connection string")):
    """
    Endpoint to list all databases and their hierarchy in MongoDB.
    """
    client, db = connect_to_mongodb(connection_string)
    database_hierarchy = {}
    
    databases = client.list_database_names()
    for database_name in databases:
        database_hierarchy[database_name] = {}
        database = client[database_name]
        collections = database.list_collection_names()
        for collection_name in collections:
            documents_count = database[collection_name].count_documents({})
            database_hierarchy[database_name][collection_name] = documents_count

    return database_hierarchy





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
   