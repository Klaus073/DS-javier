import numpy as np
from statsmodels.regression.linear_model import OLS, GLSAR
import statsmodels.api as sm

class StatisticalModeling:
    def __init__(self):
        pass

    def gen_anomaly_aggregations(self, raw_values_full, anomaly_columns, aggregations):
        numeric_cols = raw_values_full.select_dtypes(np.number).columns

        grouped_anomaly_level = raw_values_full.groupby('Anomaly_Level')

        grouped_aggregations = grouped_anomaly_level[numeric_cols].agg(aggregations)
        grouped_aggregations.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped_aggregations.columns.values]

        grouped_hours = raw_values_full.groupby(['Anomaly_Level', 'hour']).agg(aggregations)
        grouped_hours.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped_hours.columns.values]

        return grouped_hours, grouped_aggregations
    
    def genAnmAgg(self, rawvaluesFull, anomalycolumns, aggregations):
        numeric_cols = rawvaluesFull.select_dtypes(np.number).columns

        groupedAnomalyLevel = rawvaluesFull.groupby('Anomaly_Level')

        # AnomalySystemDesc = rawvaluesFull.groupby(anomalycolumns).describe()
        grouped_aggregations = groupedAnomalyLevel[numeric_cols].agg(aggregations)

        grouped_aggregations.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped_aggregations.columns.values]

        groupedhours = rawvaluesFull.groupby(['Anomaly_Level', 'hour']).agg(aggregations)
        groupedhours.columns = ['_'.join(col).strip() if col[1] else col[0] for col in groupedhours.columns.values]

        return groupedhours, grouped_aggregations

    def stats_reg_model(self, data, target: str):
        keywords = ['anomaly', 'error', 'anomaly', 'severity', 'distance', 'cluster','anomaly_level','voted_anomaly_score']
        anomaly_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in keywords)]

        y = data[target]
        X = data.drop(columns=anomaly_cols)

        # Prepare the features and target variable
        X = sm.add_constant(X)

        # Fit the linear regression model
        lin_reg = OLS(y, X).fit()

        return lin_reg