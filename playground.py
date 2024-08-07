import pandas as pd
import numpy as np
from MongoDBModule.mongo_db_manager import MongoDBManager
from DataPreProcessingModule.data_processor import DataProcessor
from AnomalyClassifierModule.anomaly_classifier import AnomalyClassifier
from AnomalyDetectionModule.anomaly_detection import AnomalyDetection
from AnomalyDetectionModelsModules.anomaly_detection_models import AnomalyDetectionModels
from StatisticalyModelingModule.statistical_modeling import StatisticalModeling


connection_string =  "mongodb://localhost:27017/" 

mongo_db = MongoDBManager(connection_string)
Data = DataProcessor()
ADM = AnomalyDetectionModels()
AD = AnomalyDetection()
SM = StatisticalModeling()
AC = AnomalyClassifier()

datetime_keywords = set(['rx.ts', 'info.datecreated', 'date', 'time',
                        'iothubenqueuedtime' 'timestamp', 'utc', 'published',
                        'publishedat','payload.publishedat', 'datecreated',
                        'payload.timestamp', 'noted_date', 'created_at',
                        'updated_at', 'modified', 'expires', 'expiry_date',
                        'accessed_at', 'deleted_at', 'published_on', 'event_time',
                        'transaction_time', 'log_time', 'start_date', 'start_time', 'end_date',
                        'end_time', 'recorded_at', 'received_at', 'sent_at'])

dataselection = 'dev_db'
databaseoptions = ["Industry 4.0", "Cisco","MDS", "Vertiv", "Smart City", "Smart Farm"]



selected_db, dev_db, Modelsdb, Models_evaluation, Anomalydb, vector_db, db_client = mongo_db.get_collections()
anomaly_condition, keyword_condition, scores_condition, all_conditions = Data.keywordConditions()
sensor_data_with_fire_alarms = pd.read_csv('Data/AirQualitySensor.csv')
dashFacingdf, fullyprocessed, main_datetime_col = Data.process_datafile(sensor_data_with_fire_alarms, datetime_keywords)
Data.split_and_store_data(fullyprocessed, main_datetime_col, 'AirQualitySensor', dev_db , mongo_db)


Vertical_selection = databaseoptions[-1]
Scaled_trainData = mongo_db.load_from_mongo("AirQualitySensor", 'Scaled_train', dev_db).iloc[:1200].dropna(axis=1)
train_data = mongo_db.load_from_mongo("AirQualitySensor", '_train', dev_db).iloc[:1200].dropna(axis=1)
cleantraindata = train_data.set_index([main_datetime_col]).select_dtypes(np.number)
scaled_cleantraindata = Scaled_trainData.set_index([main_datetime_col]).select_dtypes(np.number)

RETScaledMLInputdata, RETrawinputdata = ADM.run_anomaly_detection_models(cleantraindata,scaled_cleantraindata)
RetFromSkModels = AD.anomaly_insights(RETrawinputdata)

RetFromSkModels.reset_index(inplace=True)
anomaly_columns, keyword_columns, scores_columns, collected_columns = Data.extract_columns_and_data(RetFromSkModels, anomaly_condition, keyword_condition, scores_condition)

AnomalyLabelsCols = [col for col in anomaly_columns if col not in scores_columns]
scaled_cleantraindata['Detected_Anomaly'] = RetFromSkModels['Dynamic_Anomaly'].values
# traindf['Anomaly_Level'] = Anomaliestopass['Enhanced_Anomaly_Score'].values
scaled_cleantraindata['Anomaly_Level_Cat'] =  RetFromSkModels['Anomaly_Level']
scaled_cleantraindata['Anomaly_Level'] =  RetFromSkModels['Anomaly_Level'].factorize()[0]
scaled_cleantraindata['Voted_Anomaly_Score'] = RetFromSkModels['Voted_Anomaly_Score'].values


anomalycolumns = ['Anomaly_Level','Enhanced_Anomaly_Score','Voted_Anomaly','Anomaly_Score_STD']
aggregations = {
    'Enhanced_Anomaly_Score': ['mean', 'std', 'max'],
    'Voted_Anomaly': ['sum', 'mean'],
    'Anomaly_Score_STD':['mean', 'var', 'max']
    }

groupedhours, grouped_aggregations = SM.genAnmAgg(RETrawinputdata, anomalycolumns, aggregations)

mongo_db.save_to_mongo(grouped_aggregations.reset_index(),dataselection,"Mean_Anomaly_Scores",dev_db)
scaled_cleantraindata.dropna(axis=1, inplace=True)
anomaly_lin_reg = SM.stats_reg_model(scaled_cleantraindata, 'Detected_Anomaly')
anomaly_lin_reg_summary = anomaly_lin_reg.summary()

summary_text = str(anomaly_lin_reg_summary)
# Assuming anomaly_lin_reg_summary is your summary object
# Split the summary text into lines
lines = summary_text.split('\n')

# Parse the lines to create a dictionary
data = {}
for line in lines:
    parts = line.split()
    if len(parts) >= 2:
        key = parts[0]
        value = ' '.join(parts[1:])
        data[key] = [value]

# Convert the dictionary to a DataFrame
summary_df = pd.DataFrame(data)
mongo_db.save_to_mongo(summary_df.reset_index(),dataselection,"Anomaly_Summary",dev_db)
print("Summary Type : ",type(anomaly_lin_reg_summary))

fullprediction, grid_search_best_estimator_, x_test, y_test, test_cm, test_classreport, accuracy = AC.anomaly_classifier(scaled_cleantraindata, main_datetime_col, col='Detected_Anomaly')


selected_features, feature_scores = AC.compute_feature_values(x_test, y_test)
Aggregatefeat = feature_scores[['feature_importance']].reset_index()


fromtrain = cleantraindata.reset_index()
fullprediction = fullprediction.reset_index()
fromtrain['Detected_Anomaly'] = fullprediction['Anomaly_Predictions']
fromtrain['Detected_Anomaly_Probability'] = fullprediction['Anomaly_PredictedProb']



model_name = f'{dataselection}model'
mongo_db.save_to_mongo(RETrawinputdata.reset_index(), dataselection, "_Anomaly_Ensemble", dev_db)
mongo_db.save_sklearn_model_to_mongo(grid_search_best_estimator_, Modelsdb, test_classreport)
mongo_db.save_to_mongo(fullprediction.reset_index(), dataselection, "_Test_Prediction", dev_db)
mongo_db.save_to_mongo(Aggregatefeat, dataselection, "_feature_importance_scores", Models_evaluation)
mongo_db.save_to_mongo(fromtrain.reset_index(), dataselection, "_Test_warning", dev_db)


anomaly_columns, keyword_columns, scores_columns, collected_columns = Data.extract_columns_and_data(RETrawinputdata, anomaly_condition, keyword_condition, scores_condition)
ISO_anomaly_columns = [col for col in anomaly_columns if col not in scores_columns]
AnomaliesScores = RETrawinputdata[scores_columns]
dataclean = cleantraindata.iloc[:,:9]
feature_importanceCols = [k for k in dataclean.columns if k not in ['latitude', 'longitude', 'frequency']]


pie_chart_info = AD.display_anomaly_level_summary(RETrawinputdata)
mongo_db.save_to_mongo(pie_chart_info.reset_index(), dataselection, "PieChartInfo",dev_db)
distributionofanomalies = AD.display_anomaly_complementary_summary(RETrawinputdata)
Feat_toModel_feature_importance = AD.plot_feature_importance(dataclean[feature_importanceCols], AnomaliesScores)
featToLabelImportance = AD.anomaly_detection(dataclean[feature_importanceCols], AnomaliesScores)
multimodelFeatimpToAnomaly = AD.plot_feature_importances_interactive(Aggregatefeat)
# print(pie_chart_info , distributionofanomalies , Feat_toModel_feature_importance , featToLabelImportance , multimodelFeatimpToAnomaly)

if 'latitude' in RETrawinputdata.columns and 'longitude' in RETrawinputdata.columns:
    AD.make_anomaly_map(RETrawinputdata.reset_index(), 'Anomaly_Level', main_datetime_col,lat_col='latitude', lon_col='longitude', width=600, height=500)


# print(groupedhours, grouped_aggregations, anomaly_lin_reg_summary)
# print(distributionofanomalies,
#     grid_search_best_estimator_.best_iteration_, test_classreport,
#     pie_chart_info)
# print(multimodelFeatimpToAnomaly,
#     featToLabelImportance,
#     Feat_toModel_feature_importance)
