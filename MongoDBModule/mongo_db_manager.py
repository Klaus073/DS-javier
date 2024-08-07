from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, InvalidName, PyMongoError
import pandas as pd
import hashlib
import pickle
from datetime import datetime
import re
import json
import pymongo

class MongoDBManager:
    """
    A class to manage MongoDB operations including saving and loading data, and saving models and scalers.

    Parameters:
        meip (str): MongoDB connection string.

    Attributes:
        client: A MongoClient object for connecting to MongoDB.
        selected_db: The selected MongoDB database.
        dev_db: Development database.
        Modelsdb: Database for storing models.
        Models_evaluation: Database for storing model evaluations.
        Anomalydb: Database for storing anomaly data.
        vector_db: Database for storing document embeddings.
        db_client: Client for MongoDB operations.
    """
    def __init__(self, meip):
        self.client = MongoClient(meip)
        self.selected_db = self.client["sensorsdb"]
        self.dev_db = self.client["dev_db"]
        self.Modelsdb = self.client["Modelsdb"]
        self.Models_evaluation = self.client["Models_evaluation"]
        self.Anomalydb = self.client["Anomaly_db"]
        self.vector_db = self.client["document_embeddings"]
        self.db_client = self.client

        
    
    def get_collections(self):
        """
        Returns the database collections.

        Parameters:
            None

        Returns:
            tuple: A tuple containing the database collections in the following order:
                (selected_db, dev_db, Modelsdb, Models_evaluation, Anomalydb, vector_db, db_client)
        """
        return self.selected_db, self.dev_db, self.Modelsdb, self.Models_evaluation, self.Anomalydb, self.vector_db, self.db_client

    def save_to_mongo(self, dataframe, dataset_name, collection_suffix, db_connection):
        """
        Save DataFrame to MongoDB.

        Parameters:
            dataframe (DataFrame): The DataFrame to save.
            dataset_name (str): Name of the dataset.
            collection_suffix (str): Suffix for the collection name.
            db_connection: MongoDB connection.
        """
        collection_name = f"{dataset_name}{collection_suffix}"
        collection = db_connection[collection_name]
        collection.delete_many({})  # Clear the collection before saving
        collection.insert_many(dataframe.to_dict('records'))

    def load_from_mongo(self, dataset_name, collection_suffix, db_connection):
        """
        Load data from MongoDB.

        Parameters:
            dataset_name (str): Name of the dataset.
            collection_suffix (str): Suffix for the collection name.
            db_connection: MongoDB connection.

        Returns:
            DataFrame: Loaded data.
        """
        collection_name = f"{dataset_name}{collection_suffix}"
        collection = db_connection[collection_name]

        try:
            if collection.count_documents({}) == 0:
                print('Data not found. Please first run the appropriate section.')
                return pd.DataFrame()

            data_cursor = collection.find()
            dataframe = pd.DataFrame(list(data_cursor))

            dataframe.drop(columns=['_id'], inplace=True, errors='ignore')
            return dataframe
        except Exception as e:
            print(f'Error loading data from MongoDB: {e}')
            return pd.DataFrame()

    

    def save_sklearn_model_to_mongo(self, model, db, classificationReport):
        """
        Save or update a scikit-learn model to MongoDB.

        Parameters:
            model: The scikit-learn model to save.
            db: MongoDB connection.
            classificationReport: The classification report for the model.
        """
        try:
            model_class_name = model.__class__.__name__
            model_hash_value = self.model_hash(model)
            serialized_model = pickle.dumps(model)

            model_entry = {
                'class': model_class_name,
                'hash': model_hash_value,
                'model_data': serialized_model,
                'classification_report': classificationReport,
                'saved_on': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

            existing_model = db.models.find_one({'class': model_class_name, 'hash': model_hash_value})

            if existing_model is not None:
                db.models.replace_one({'class': model_class_name, 'hash': model_hash_value}, model_entry)
                print(f"Updated model as {model_class_name} with hash {model_hash_value}.")
            else:
                db.models.insert_one(model_entry)
                print(f"Saved new model as {model_class_name} with hash {model_hash_value}.")

        except Exception as e:
            print(f"Error saving model: {e}")

    def save_sklearn_scaler_to_mongo(self, artifact, artifact_type, db):
        """
        Save or update a scikit-learn scaler to MongoDB.

        Parameters:
            artifact: The artifact to save.
            artifact_type (str): Type of the artifact.
            db: MongoDB connection.
        """
        try:
            artifact_class_name = artifact.__class__.__name__
            artifact_hash_value = self.model_hash(artifact)
            serialized_artifact = pickle.dumps(artifact)

            artifact_entry = {
                'class': artifact_class_name,
                'hash': artifact_hash_value,
                'type': artifact_type,
                'artifact_data': serialized_artifact,
                'saved_on': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

            existing_artifact = db.models.find_one({'class': artifact_class_name, 'hash': artifact_hash_value})

            if existing_artifact is not None:
                db.models.replace_one({'class': artifact_class_name, 'hash': artifact_hash_value}, artifact_entry)
                print(f"Updated {artifact_type} as {artifact_class_name} with hash {artifact_hash_value}.")
            else:
                db.models.insert_one(artifact_entry)
                print(f"Saved new {artifact_type} as {artifact_class_name} with hash {artifact_hash_value}.")

        except Exception as e:
            print(f"Error saving {artifact_type}: {e}")

    def model_hash(self, model):
        """
        Generate a hash value for a scikit-learn model.

        Parameters:
            model: The scikit-learn model.

        Returns:
            str: Hash value of the model.
        """
        return hashlib.sha256(pickle.dumps(model)).hexdigest()

    def model_hash(self, model):
        """
        Generate a hash value for a scikit-learn model.

        Parameters:
            model: Scikit-learn model object.

        Returns:
            str: Hash value of the model.
        """
        model_info = str(model.__class__.__name__) + str(model.get_params())
        return hashlib.md5(model_info.encode()).hexdigest()

    def clean_column_names(self, columns):
        """
        Clean column names of a DataFrame.

        Parameters:
            columns: List of column names.

        Returns:
            List: Cleaned column names.
        """
        new_columns = []
        pattern = re.compile(r'filtered/application/\d+/device/[^/]+/event/up\.(.+)')
        for col in columns:
            match = pattern.search(col)
            col = match.group(1) if match else col.split("/")[-1]
            col = col.split('.', 1)[-1] if '.' in col else col
            new_columns.append(col)
        return new_columns

    def mongodbdatasets(self, iotdatabase, nameOfCollection: str):
        """
        Load data from MongoDB collection and preprocess.

        Parameters:
            iotdatabase: MongoDB database object.
            nameOfCollection (str): Name of the collection.

        Returns:
            DataFrame: Preprocessed DataFrame.
        """
        try:
            data = list(iotdatabase[nameOfCollection].find())
            df = pd.json_normalize(data)
            if 'payload.objectJSON' in df.columns:
                df['payload.objectJSON'] = df['payload.objectJSON'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                object_df = pd.json_normalize(df['payload.objectJSON'].dropna())
                df = df.drop(columns='payload.objectJSON').join(object_df, lsuffix='_left')
            rxInfo_pattern = re.compile(r'filtered/application/\d+/device/.+/event/up\.rxInfo')
            rxInfo_columns = [col for col in df.columns if rxInfo_pattern.match(col)]
            for rxInfo_column in rxInfo_columns:
                df[rxInfo_column] = df[rxInfo_column].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                exploded_df = df.explode(rxInfo_column).reset_index(drop=True)
                normalized_df = pd.json_normalize(exploded_df[rxInfo_column].dropna())
                df = exploded_df.drop(columns=rxInfo_column).join(normalized_df, rsuffix='_rxInfo')
            for col in df.columns:
                if not df.empty and not isinstance(df[col].iloc[0], pymongo.objectid.ObjectId):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                    except Exception as e:
                        print(f"Failed to convert column {col} to numeric. Error: {e}")
            cleaned_columns = self.clean_column_names(df.columns)
            df.columns = cleaned_columns
            df.columns = df.columns.str.lower()
            return df
        except Exception as e:
            print(f"An error occurred: {e}")
            return None