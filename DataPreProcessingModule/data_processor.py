import numpy as np
import pandas as pd
from datetime import datetime
import re
from sklearn.preprocessing import RobustScaler, StandardScaler
import calendar
from pymongo import MongoClient
from bson.objectid import ObjectId
import json

class DataProcessor:
    """
    A class to process and manipulate data.

    Attributes:
        None
    """

    def __init__(self):
        pass

    def identify_datetime_cols(self, df, datetime_keywords):
        """
        Identify columns containing datetime information.

        Parameters:
            df (DataFrame): The DataFrame to search for datetime columns.
            datetime_keywords (list): List of keywords to identify potential datetime columns.

        Returns:
            list: List of identified datetime columns.
        """
        datetime_cols = []
        for col in df.columns:
            try:
                if df[col].dtype == 'datetime64[ns]':
                    datetime_cols.append(col)
                elif df[col].dtype == 'object' or col in datetime_keywords:
                    datetime_cols.append(col)
                elif col in datetime_keywords:
                    datetime_cols.append(col)
            except Exception as e:
                print(f'{e}')
        return datetime_cols

    def identify_and_convert_datetime_columns(self, df, datetime_keywords):
        """
        Identify and convert datetime columns to a standard format.

        Parameters:
            df (DataFrame): The DataFrame to process.
            datetime_keywords (list): List of keywords to identify potential datetime columns.

        Returns:
            DataFrame: Processed DataFrame.
            list: List of identified datetime columns.
            str: Name of the main datetime column.
        """
        datetime_cols = self.identify_datetime_cols(df, datetime_keywords)
        for col in datetime_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if df[col].dtype == 'datetime64[ns]':
                    df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                print(f"Error converting column '{col}' to datetime format: {e}")
        main_datetime_col = datetime_cols[0] if datetime_cols else None
        if main_datetime_col:
            df[main_datetime_col] = pd.to_datetime(df[main_datetime_col])
        return df, datetime_cols, main_datetime_col

    def drop_unwanted_columns(self, df):
        """
        Drop unwanted columns from the DataFrame.

        Parameters:
            df (DataFrame): The DataFrame to process.

        Returns:
            DataFrame: Processed DataFrame with unwanted columns dropped.
        """
        columns_to_drop = ['_id', 'index', 'Unnamed: 0', 'level_0','unnamed: 0']
        df_cleaned = df.drop(columns=columns_to_drop, axis=1, errors='ignore')
        return df_cleaned

    def create_features(self, df, datecol=None, label=None):
        """
        Create additional features from datetime data.

        Parameters:
            df (DataFrame): The DataFrame to process.
            datecol (str): Name of the datetime column.
            label (str): Name of the label column.

        Returns:
            DataFrame: DataFrame with added features.
        """
        df[datecol] = pd.to_datetime(df[datecol])

        season_dict = {
        '01': 'Winter',
        '02': 'Winter',
        '03': 'Spring',
        '04': 'Spring',
        '05': 'Spring',
        '06': 'Summer',
        '07': 'Summer',
        '08': 'Summer',
        '09': 'Fall',
        '10': 'Fall',
        '11': 'Fall',
        '12': 'Winter'}

        df['timeofday'] = pd.to_datetime(df[datecol]).dt.hour.apply(
        lambda x: 'night' if 0 <= x < 5 else
                'morning' if 5 <= x < 12 else
                'afternoon' if 12 <= x < 18 else
                'evening')

        df['month'] = df[datecol].dt.month.apply(lambda x: f"{x:02d}")
        Season = df['month'].apply(lambda x: season_dict[x])

        df['timeofday'] = pd.to_datetime(df[datecol]).dt.hour.apply(lambda x: 'morning' if 5 <= x <= 12 else 'afternoon' if 13 <= x <= 17 else 'evening')

        df['season'] = Season

        df['dayofweek'] = df[datecol].dt.dayofweek.apply(lambda x: calendar.day_name[x].lower())

        df['month'] = df[datecol].dt.month.apply(lambda x: calendar.month_name[x].lower())

        df['hour'] = df[datecol].dt.hour

        return df

    def best_fill_na_method(self, column):
        """
        Determine the best method to fill missing values in a column.

        Parameters:
            column: The column to process.

        Returns:
            str: The best method to fill missing values.
        """
        methods = ['mean', 'median', 'zero', 'bfill', 'ffill']
        original_total = column.sum()
        best_method = None
        best_difference = float('inf')

        for method in methods:
            col_copy = column.copy()
            fill_value = 0
            if method in ['mean', 'median']:
                fill_value = getattr(col_copy, method)()  # Use getattr to call mean or median
            col_copy.fillna(fill_value if method != 'bfill' and method != 'ffill' else method, inplace=True)

            total = col_copy.sum()
            difference = abs(total - original_total)

            if difference < best_difference:
                best_difference = difference
                best_method = method

        return best_method

    def fill_missing_values(self, df):
        """
        Fill missing values in the DataFrame.

        Parameters:
            df (DataFrame): The DataFrame to process.

        Returns:
            DataFrame: DataFrame with missing values filled.
        """
        for col_name in df.select_dtypes(include=[np.number]).columns:
            best_method = self.best_fill_na_method(df[col_name])
            if best_method in ['mean', 'median', 'zero']:
                fill_value = getattr(df[col_name], best_method)() if best_method != 'zero' else 0
                df[col_name].fillna(fill_value, inplace=True)
            else:  # For 'bfill' and 'ffill'
                df[col_name].fillna(method=best_method, inplace=True)
        return df

    def remove_low_variance_features(self, dataset):
        """
        Remove features with low variance from the dataset.

        Parameters:
            dataset (DataFrame): The DataFrame to process.

        Returns:
            DataFrame: DataFrame with low variance features removed.
        """
        exclusion_list = ['deveui', 'devicename', 'devicetype', 'latitude', 'longitude', 'gatewayid','fire alarm','occupancy','hour',
                          'alarm', 'label','target',"Fire_Alarm"]

        for col in dataset.columns:
            try:
                if dataset[col].nunique() <= 2 and col not in exclusion_list:
                    dataset.drop(col, axis=1, inplace=True)
            except Exception as e:
                print(f"Failed to convert column {col} to float. Skipping conversion for this column.")
                continue

        return dataset

    def split_and_store_data(self, df, datetime_col, dataset_name, db_connection , mongoDB_manager):
        """
        Split the data based on datetime and store in MongoDB.

        Parameters:
            df (DataFrame): The DataFrame to process.
            datetime_col (str): Name of the datetime column.
            dataset_name (str): Name of the dataset.
            db_connection: MongoDB connection for storing processed data.
        """
        scaled_data, scaler = self.scale_ml_data(df, datetime_col)
        scaled_data.reset_index(inplace=True)

        scaled_train, scaled_test = self.train_test_split_datetime(scaled_data, datetime_col)
        train, test = self.train_test_split_datetime(df, datetime_col)

        scaler_name = f'{dataset_name}_scaler'
        mongoDB_manager.save_sklearn_scaler_to_mongo(scaler, scaler_name, db_connection)

        mongoDB_manager.save_to_mongo(train, dataset_name, "_train", db_connection)
        mongoDB_manager.save_to_mongo(test, dataset_name, "_test", db_connection)

        mongoDB_manager.save_to_mongo(scaled_train, dataset_name, "Scaled_train", db_connection)
        mongoDB_manager.save_to_mongo(scaled_test, dataset_name, "Scaled_test", db_connection)

        print(f"Data processing complete. Train and test sets saved to MongoDB.")

    def process_datafile(self, df, datetime_keywords):
        """
        Process the datafile.

        Parameters:
            df (DataFrame): The DataFrame to process.
            datetime_keywords (list): List of keywords to identify datetime columns.

        Returns:
            DataFrame: Processed DataFrame.
            DataFrame: Fully processed DataFrame.
            str: Name of the main datetime column.
        """
        df_processed, datetime_cols, main_datetime_col = self.identify_and_convert_datetime_columns(df, datetime_keywords)
        df_processed = self.drop_unwanted_columns(df_processed)
        df_processed = self.remove_low_variance_features(df_processed)
        df_processed = self.fill_missing_values(df_processed)
        df_processed.columns = df_processed.columns.str.lower()
        fully_processed = self.create_features(df_processed, main_datetime_col)
        return df_processed, fully_processed, main_datetime_col

    def load_and_process_data(self, dataselection, datetime_keywords, db_connection):
        """
        Load, process, and store data.

        Parameters:
            dataselection (str): Name of the collection to load data from.
            datetime_keywords (list): List of keywords to identify datetime columns.
            db_connection: MongoDB connection for storing processed data.

        Returns:
            DataFrame: Fully processed DataFrame.
            str: Name of the main datetime column.
        """
        df = self.mongodbdatasets(db_connection, dataselection)
        processedNoFeats, fullyprocessed, main_datetime_col = self.process_datafile(df, datetime_keywords)
        self.split_and_store_data(fullyprocessed, main_datetime_col, dataselection, db_connection)
        print("Data processing and storage complete.")
        return fullyprocessed, main_datetime_col

    def mongodbdatasets(self, iotdatabase, nameOfCollection: str):
        """
        Retrieve data from MongoDB.

        Parameters:
            iotdatabase: MongoDB database.
            nameOfCollection (str): Name of the collection.

        Returns:
            DataFrame: Retrieved data.
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
                if not df.empty and not isinstance(df[col].iloc[0], ObjectId):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                    except Exception as e:
                        print(f"Failed to convert column {col} to numeric. Error: {e}")

            df.columns = df.columns.str.lower()

            return df

        except Exception as e:
            print(f"An error occurred: {e}")
            return None


    def train_test_split_datetime(self, df, date_column, test_size=0.2):
        """
        Split the dataset based on datetime, ensuring chronological order.

        Parameters:
            df (DataFrame): The DataFrame to split.
            date_column (str): Name of the datetime column.
            test_size (float): Size of the test dataset.

        Returns:
            DataFrame: Train dataset.
            DataFrame: Test dataset.
        """
        if date_column not in df.columns:
            raise ValueError(f"{date_column} column not found in the DataFrame.")

        df[date_column] = pd.to_datetime(df[date_column])
        df_sorted = df.sort_values(by=date_column)
        split_index = int((1 - test_size) * len(df_sorted))
        return df_sorted[:split_index], df_sorted[split_index:]

    def scale_ml_data(self, data, main_datetime_col):
        """
        Scale numerical columns using RobustScaler and preserve other types.

        Parameters:
            data (DataFrame): The DataFrame to scale.
            main_datetime_col (str): Name of the main datetime column.

        Returns:
            DataFrame: Scaled DataFrame.
            dict: Scaler information.
        """
        exclusion_list = ['deveui', 'devicename', 'devicetype', 'latitude', 'longitude', 'gatewayid','fire alarm','occupancy','hour',
                          'alarm', 'label','target',"Fire_Alarm",'frequency']

        data.set_index(main_datetime_col, inplace=True)
        numeric_cols = [col for col in data.select_dtypes(include=['number']).columns if col not in exclusion_list]

        scaler = RobustScaler()
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

        scaler_info = {'scaler': scaler, 'columns': numeric_cols}

        return data, scaler_info

   
    def keywordConditions(self):
        """
        Define conditions for identifying columns based on keywords.

        Returns:
            tuple: Tuple of keyword conditions.
        """
        keywords = ['Anomaly', 'error', 'anomaly', 'severity', 'optimization', 'prediction','distance']
        anomaly_condition = lambda col: 'Anomaly' in col or 'error' in col
        keyword_condition = lambda col: any(keyword in col for keyword in keywords)
        scores_condition = lambda col: ('Anomaly' in col and 'Score' in col) or 'error' in col
        all_conditions = [anomaly_condition, keyword_condition, scores_condition]

        return anomaly_condition, keyword_condition, scores_condition, all_conditions

    def extract_columns_and_data(self, df, anomaly_condition, keyword_condition, scores_condition):
        """
        Extract relevant columns from the DataFrame.

        Parameters:
            df (DataFrame): The DataFrame to process.
            anomaly_condition (function): Condition for anomaly detection columns.
            keyword_condition (function): Condition for keyword-based columns.
            scores_condition (function): Condition for score columns.

        Returns:
            tuple: Tuple containing lists of extracted columns.
        """
        anomaly_columns = []
        keyword_columns = []
        scores_columns = []

        for col in df.columns:
            if anomaly_condition(col):
                anomaly_columns.append(col)
            if keyword_condition(col):
                keyword_columns.append(col)
            if scores_condition(col):
                scores_columns.append(col)

        collected_columns = list(set(anomaly_columns + keyword_columns + scores_columns))
        return anomaly_columns, keyword_columns, scores_columns, collected_columns