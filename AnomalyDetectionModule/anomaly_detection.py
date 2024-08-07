import numpy as np
import pandas as pd
import calendar

import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import networkx as nx

class AnomalyDetection:
    def __init__(self):
        pass

    def anomaly_insights(self, RetFromSkModels):
        anoms = ['IsolationForest_Anomaly', 'EllipticEnvelope_Anomaly',
                'OneClassSVM_Anomaly', 'Autoencoder_Anomaly']

        anomscore = ['IsolationForest_AnomalyScore', 'EllipticEnvelope_AnomalyScore',
                    'OneClassSVM_AnomalyScore', 'Autoencoder_reconstruction_error']

        Scores = RetFromSkModels[anomscore]
        Anomalies = RetFromSkModels[anoms]

        Anomaly_Score_STD = Scores.std(axis=1)
        anomalyscore_mean = Scores.mean(axis=1)
        anomaly_sum = Anomalies.sum(axis=1)
        anomaly_counts = Anomalies.sum(axis=0)
        positive_classification_rates = Anomalies.mean(axis=0)

        RetFromSkModels['Voted_Anomaly'] = np.where(anomaly_sum >= 3, 1, 0)
        RetFromSkModels['Voted_Anomaly_Score_Mean'] = anomalyscore_mean
        RetFromSkModels['Voted_Anomaly_Score'] = anomaly_sum * 0.25

        # Calculate a dynamic threshold based on the statistical distribution of the anomaly_sum
        dynamic_threshold = anomaly_sum.median()  # Using median as an example

        # Calculate standard deviation for anomaly scores to assess variability
        RetFromSkModels['Anomaly_Score_STD'] = Anomaly_Score_STD

        normalized_rates = positive_classification_rates / positive_classification_rates.sum()

        # Invert the rates to penalize models with higher rates of positive classifications
        # This creates a scenario where a model with more frequent positive classifications gets a lower weight
        inverted_rates = 1 - normalized_rates

        # Normalize inverted rates to ensure they sum up to 1 and can be used as weights
        weights = inverted_rates / inverted_rates.sum()

        # Calculate a weighted anomaly score mean based on dynamically adjusted weights
        RetFromSkModels['Weighted_Anomaly_Score'] = Anomalies.dot(weights) / weights.sum()

        # Updating the 'Voted_Anomaly' column with the dynamic threshold
        RetFromSkModels['Dynamic_Anomaly'] = np.where(Anomalies.sum(axis=1) >= dynamic_threshold, 1, 0)

        # Enhanced scoring mechanism considering both mean score and variability
        # For instance, you may penalize scores with high variability
        penalty_factor = 0.1  # Example penalty for high variability
        RetFromSkModels['Enhanced_Anomaly_Score'] = RetFromSkModels['Weighted_Anomaly_Score'] - (RetFromSkModels['Anomaly_Score_STD'] * penalty_factor)

        # Incorporate a check for unanimous agreement among models for a bonus
        unanimous_bonus = 0.05  # Bonus to add for unanimous agreement among models
        RetFromSkModels['Enhanced_Anomaly_Score'] += np.where(Anomalies.sum(axis=1) == len(anoms), unanimous_bonus, 0)

        # You may want to normalize or scale the 'Enhanced_Anomaly_Score' as needed

        Scorequantile = RetFromSkModels['Enhanced_Anomaly_Score'].quantile(q=[0.25, 0.50, 0.75, 0.95])
        std_quantiles = RetFromSkModels['Anomaly_Score_STD'].quantile(q=[0.25, 0.50, 0.75])

        RetFromSkModels['Anomaly_Level'] = RetFromSkModels.apply(lambda row: self.classify_anomaly_enhanced(row['Enhanced_Anomaly_Score'], row['Anomaly_Score_STD'], Scorequantile, std_quantiles), axis=1)

        return RetFromSkModels

    def classify_anomaly_enhanced(self, score, std_dev, quantiles, std_quantiles):
        if std_dev >= std_quantiles[0.75]:
            sensitivity_adjustment = 1  
        elif std_dev <= std_quantiles[0.25]:
            sensitivity_adjustment = -1  
        else:
            sensitivity_adjustment = 0  

        critical_threshold = quantiles[0.95] + (0.05 * sensitivity_adjustment)
        high_threshold = quantiles[0.75] + (0.05 * sensitivity_adjustment)

        if score >= critical_threshold:
            return 'Critical'
        elif score >= high_threshold:
            return 'Concern'
        elif score >= quantiles[0.50]:
            return 'Notice'
        elif score >= quantiles[0.25]:
            return 'Low'
        else:
            return 'Normal'

    def display_anomaly_level_summary(self, Anomalies):
        """
        The function takes a DataFrame of anomaly levels, calculates the counts for each level,
        and displays a pie chart showing the distribution of different anomaly levels.

        :param Anomalies: pandas DataFrame with anomaly level results.
        """

        anomaly_level_counts = Anomalies['Anomaly_Level'].value_counts()

        labels = anomaly_level_counts.index
        values = anomaly_level_counts.values

        fig = px.pie(
            names=labels,
            values=values,
            title='Distribution of Anomaly Levels',
            hole=0.4,
            template='plotly_white'
        )

        fig.update_traces(textinfo='percent+label')
        fig.update_layout(width=550,  height=500, margin=dict(t=20, b=40))


        pie_chart_info = "The pie chart above shows the distribution of different anomaly levels.\n"
        for level, count in anomaly_level_counts.items():
            pie_chart_info += f"{level}: {count}\n"


        print(pie_chart_info)
        fig.show()

        data = {}
        for line in pie_chart_info.strip().split('\n'):
            parts = line.split(': ')
            if len(parts) == 2:
                key, value = parts
                data[key] = [int(value)]
            else:
                print(f"Ignoring invalid line: {line}")
                
        dataframe_pie_chart_info = pd.DataFrame(data)
        return dataframe_pie_chart_info #pie_chart_info
    
    def display_anomaly_complementary_summary(self, Anomalies):
        """
        The `display_anomaly_complementary_summary` function takes in a DataFrame of anomaly results from different
        algorithms, calculates the counts of anomalies and non-anomalies for each algorithm and the final
        decision, and displays a pie chart showing the distribution of anomalies and non-anomalies across all algorithms.

        :param Anomalies: The parameter "Anomalies" is expected to be a pandas DataFrame that contains the
        anomaly detection results for each data point. The DataFrame should have the following columns:
        """
        anomaly_countsdf = Anomalies[
            [
                'IsolationForest_Anomaly', 'EllipticEnvelope_Anomaly','OneClassSVM_Anomaly',
                'Autoencoder_Anomaly','Voted_Anomaly']
            ]

        anomaly_counts = anomaly_countsdf.apply(lambda col: (col == 1).sum())
        non_anomaly_counts = anomaly_countsdf.apply(lambda col: (col == 0).sum())

        total_anomalies = anomaly_counts['Voted_Anomaly'].sum()
        total_non_anomalies = non_anomaly_counts['Voted_Anomaly'].sum()

        algorithms = anomaly_counts.index.tolist()

        pooled_average_anomalies = anomaly_counts.mean()
        pooled_average_non_anomalies = non_anomaly_counts.mean()

        labels = ['Anomalies', 'Non-Anomalies']
        values = [total_anomalies, total_non_anomalies]

        fig = px.pie(
            names=labels,
            values=values,
            title='Distribution of Anomalies and Non-Anomalies for Final Classification',
            hole=0.4,
            template='plotly_white'
        )

        fig.update_traces(textinfo='percent+label')
        fig.update_layout(width=550,  height=500, margin=dict(t=50, b=20))


        pie_chart_info = f"The pie chart above shows the distribution of anomalies\nand non-anomalies Final Classification.\n"
        pie_chart_info += f"Total Anomalies: {total_anomalies}\n"
        pie_chart_info += f"Total Non-Anomalies: {total_non_anomalies}\n"

        print(pie_chart_info)
        fig.show()

        pie_chart_info += f"Pooled average anomalies in the final decision:\n{round(pooled_average_anomalies, 4)}\n"
        pie_chart_info += f"Pooled average non-anomalies in the final decision:\n{round(pooled_average_non_anomalies, 4)}"


        return pie_chart_info
    
    def plot_feature_importance(self, data, anomaly_scores):
        """
        The function `plot_feature_importance` generates a bar chart to visualize the feature importance for
        different anomaly detection algorithms.

        :param data: The `data` parameter is a pandas DataFrame that contains the features used for anomaly
        detection. Each column represents a different feature
        :param anomaly_scores: The `anomaly_scores` parameter is a dictionary that contains the anomaly
        scores for each algorithm. The keys of the dictionary are the names of the algorithms, and the
        values are the corresponding anomaly scores
        """

        anomaly_algorithms = ['IsolationForest_AnomalyScore', 'EllipticEnvelope_AnomalyScore',
                        'OneClassSVM_AnomalyScore', 'Autoencoder_reconstruction_error']

        feature_importance = pd.DataFrame(index=data.columns)
        for name in anomaly_algorithms:
            correlations = data.corrwith(anomaly_scores[name])
            feature_importance[name] = correlations.abs()


        feature_importance['Average'] = feature_importance.mean(axis=1)
        feature_importance = feature_importance.sort_values(by='Average', ascending=False)

        colors = px.colors.qualitative.Plotly
        fig = go.Figure()
        for i, column in enumerate(feature_importance.columns):
            fig.add_trace(go.Bar(x=feature_importance.index,
                                y=feature_importance[column],
                                name=column,
                                marker_color=colors[i % len(colors)],
                                hoverinfo='y',
                                ))

        fig.update_layout(
            title='Feature Importance in Anomaly Detection Analysis',
            xaxis_title='Features',
            yaxis_title='Importance',
            barmode='group',
            yaxis_type='log',
            font=dict(size=10),
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            width=450,
            height=500)

        fig.show()

        return feature_importance
    
    def anomaly_detection(self, datapoints, anomaly_results_rela):
        """
        The `anomaly_detection` function calculates the feature importance of different anomaly detection
        algorithms using correlation and creates a heatmap visualization of the results.

        :param datapoints: The `datapoints` parameter is a DataFrame that contains the data points for which
        you want to calculate the feature importance
        :param anomaly_results_rela: The `anomaly_results_rela` parameter is a DataFrame that contains the
        results of different anomaly detection algorithms. Each column in the DataFrame represents a
        different algorithm, and each row represents a datapoint. The values in the DataFrame indicate the
        anomaly score or label assigned by each algorithm to each datapoint
        :return: a DataFrame called "feature_importance" which contains the feature importance values
        calculated for each anomaly algorithm.
        """
        anomaly_algorithms = anomaly_results_rela.columns

        feature_importance = pd.DataFrame()

        for name in anomaly_algorithms:
            correlations = datapoints.corrwith(anomaly_results_rela[name]).abs()
            feature_importance[name] = correlations

        feature_importance['average_importance'] = feature_importance.mean(axis=1)
        text_labels = feature_importance.T.applymap(lambda x: '{:.2f}'.format(x)).values


        heatmap = go.Heatmap(
            z=feature_importance.T.values,
            x=feature_importance.T.columns,
            y=feature_importance.T.index,
            zmin=0,
            zmax=3,
            colorscale='RdBu_r',
            zmid=2.5,
            xgap=10,
            ygap=10,
            text=text_labels,
            hoverinfo='text'
        )

        layout = go.Layout(
            title='Heatmap of Feature Importance',
            xaxis=dict(tickangle=-45, nticks=30),
            yaxis=dict(nticks=30),
            font=dict(size=12),
            width=450,
            height=500
        )


        fig = go.Figure(data=[heatmap], layout=layout)

        fig.show()


        return feature_importance
    
    def generate_anomaly_color_map(self, AnomalyLevels):
        color_gradient = {
            'Low': '#ffcccc',  # lighter red
            'Notice': '#ff9999',  # light red
            'Concern': '#ff6666',  # medium red
            'Critical': '#cc0000'  # dark red
        }
        default_color = '#cccccc'

        color_map = {level: color_gradient.get(level, default_color) for level in AnomalyLevels}

        return color_map


    def make_anomaly_map(self, df, col, time_col,lat_col='latitude', lon_col='longitude', width=600, height=700): #time_col:str,

        color_map = self.generate_anomaly_color_map(df[col].unique())
        df['color'] = df[col].map(color_map)

        df[time_col] = pd.to_datetime(df[time_col]).dt.strftime('%Y-%m-%d')

        hover_data = {c: True for c in df.columns if c not in ['color','Detected_Anomaly', 'Anomaly_Score','Detected_Anomaly_Probability','Anomaly_lvl','devicetype','gatewayid']}

        fig = px.scatter_mapbox(df, lat=lat_col, lon=lon_col, color=col,
                                color_discrete_map=color_map,
                                # hover_name=col,
                                hover_data=hover_data,
                                animation_frame=time_col,
                                size_max=18, zoom=13,
                                title='Anomalies Over Time',
                                mapbox_style="open-street-map")

        fig.update_traces(marker=dict(size=15))


        fig.update_layout(mapbox=dict(center=dict(lat=df[lat_col].mean(), lon=df[lon_col].mean())),
                        margin={"r":0,"t":0,"l":0,"b":0},
                        width=width,
                        height=height)

        fig.show()

    def plot_feature_importances_interactive(self, feature_scores):
        long_df = feature_scores.melt(id_vars='index', var_name='feature_importance', value_name='Score')

        marker_size = long_df['Score'].abs() * 10

        fig = px.bar(long_df, x="index", y="Score", color="index",template="simple_white",
                        hover_data=["Score"], title="Feature Importance in Anomaly Classification Analysis")

        fig.add_hline(y=0, line_dash="dash", line_color="black")

        fig.update_layout(
            width=900,
            height=600,
            hoverlabel=dict(
                bgcolor="black",
                font_size=15,
                font_family="Rockwell"
            ),
            xaxis_title="Feature",
            yaxis_title="Importance Score"
        )

        fig.show()
        return long_df