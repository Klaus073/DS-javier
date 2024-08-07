
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.exceptions import NotFittedError
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, roc_auc_score, make_scorer, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif

from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

class AnomalyDetectionModels:
    def __init__(self, random_state=42, cv=4, anomaly=1, not_anomaly=0, unfitted=-1):
        self.random_state = random_state
        self.cv = cv
        self.anomaly = anomaly
        self.not_anomaly = not_anomaly
        self.unfitted = unfitted

    def sklearn_models(self, raw_input_data, scaled_ml_input_data):
        """
        The function `sklearnModels` takes in a scaled dataset and a dataframe of anomalies, tunes the
        hyperparameters of three anomaly detection algorithms (Isolation Forest, Local Outlier Factor, and
        One-Class SVM), applies the algorithms to the dataset, and returns the updated dataframe of
        anomalies.

        :param PWRdf_scaled: PWRdf_scaled is a scaled version of the PWRdf dataset. It is likely a pandas
        DataFrame or numpy array that contains the input data for the anomaly detection models. The data
        should be preprocessed and scaled before passing it to the models
        :param Anomalies: Anomalies is a DataFrame that contains the data on which the anomaly detection
        algorithms will be applied. It should have the necessary columns for the algorithms to work properly
        :return: the "Anomalies" dataframe, which contains the results of the anomaly detection algorithms.
        """

        anomaly_algorithms = {
            'EllipticEnvelope': EllipticEnvelope(support_fraction=.35),
            'IsolationForest': IsolationForest(),
            'OneClassSVM': OneClassSVM(kernel='rbf'),
        }
        for name, algorithm in anomaly_algorithms.items():
            if name == 'EllipticEnvelope':
                raw_input_data["EllipticEnvelope_Anomaly"] = np.where(algorithm.fit_predict(scaled_ml_input_data) == self.unfitted, self.anomaly, self.not_anomaly)
                raw_input_data["EllipticEnvelope_AnomalyScore"] = algorithm.mahalanobis(scaled_ml_input_data) / 100
            elif name == 'IsolationForest':
                raw_input_data["IsolationForest_Anomaly"] = np.where(algorithm.fit_predict(scaled_ml_input_data) == self.unfitted, self.anomaly, self.not_anomaly)
                raw_input_data["IsolationForest_AnomalyScore"] = algorithm.decision_function(scaled_ml_input_data)
            else:
                raw_input_data['OneClassSVM_Anomaly'] = np.where(algorithm.fit_predict(scaled_ml_input_data) == self.unfitted, self.anomaly, self.not_anomaly)
                raw_input_data['OneClassSVM_AnomalyScore'] = algorithm.decision_function(scaled_ml_input_data) / 100
        return scaled_ml_input_data, raw_input_data

    def train_auto_encoder(self, data, epochs=50, batch_size=32, test_size=0.2):
        """
        Train an autoencoder model on the given data.

        :param data: Input data for training the autoencoder.
        :param epochs: Number of training epochs (default=50).
        :param batch_size: Size of training batch (default=32).
        :param test_size: Fraction of data to use for testing (default=0.1).
        :return: Normalized reconstruction error and anomaly threshold.
        """


        train_data, test_data = train_test_split(data, test_size=test_size, random_state=self.random_state)
        input_dim = train_data.shape[1]
        encoding_dim = int(input_dim / 2)

        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='relu')(encoded)

        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        autoencoder.fit(train_data, train_data, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(test_data, test_data), verbose=0)

        scaled_data_predictions = autoencoder.predict(data)
        reconstruction_error = np.mean(np.power(data - scaled_data_predictions, 2), axis=1)
        normalized_error = (reconstruction_error - np.min(reconstruction_error)) / (np.max(reconstruction_error) - np.min(reconstruction_error))
        threshold = np.percentile(normalized_error, 95)

        return normalized_error, threshold

    def run_deep_auto_encoder(self, scaled_ml_input_data, raw_input_data):
        """
        Use a trained autoencoder model to detect anomalies in the data.

        :param ScaledMLInputdata: Scaled input data for anomaly detection.
        :param rawinputdata: DataFrame containing previously detected anomalies.
        :return: Returns updated versions of ScaledMLInputdata and rawinputdata, with additional columns for autoencoder-based anomaly detection.
        """
        
        normalized_error, threshold = self.train_auto_encoder(scaled_ml_input_data)

        raw_input_data['Autoencoder_Anomaly'] = np.where(normalized_error > threshold, 1, 0)
        raw_input_data['Autoencoder_reconstruction_error'] = normalized_error.tolist()

        return scaled_ml_input_data, raw_input_data

    def classify_anomaly_enhanced(self, score, std_dev, quantiles, std_quantiles):
        """
        Classify anomalies using score mean, standard deviation, and quantiles.

        :param score: Mean anomaly score for the data point.
        :param std_dev: Standard deviation of anomaly scores for the data point.
        :param quantiles: Quantile thresholds for scoring.
        :param std_quantiles: Quantile thresholds for standard deviation of scores.
        :return: Anomaly classification level.
        """

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
    
    def trainAutoEncoder(self, data, epochs=50, batch_size=32, test_size=0.2):
        """
        Train an autoencoder model on the given data.

        :param data: Input data for training the autoencoder.
        :param epochs: Number of training epochs (default=50).
        :param batch_size: Size of training batch (default=32).
        :param test_size: Fraction of data to use for testing (default=0.1).
        :return: Normalized reconstruction error and anomaly threshold.
        """
        # try:
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        input_dim = train_data.shape[1]
        encoding_dim = int(input_dim / 2)

        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='relu')(encoded)

        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        autoencoder.fit(train_data, train_data, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(test_data, test_data), verbose=0)

        scaled_data_predictions = autoencoder.predict(data)
        reconstruction_error = np.mean(np.power(data - scaled_data_predictions, 2), axis=1)
        normalized_error = (reconstruction_error - np.min(reconstruction_error)) / (np.max(reconstruction_error) - np.min(reconstruction_error))
        threshold = np.percentile(normalized_error, 95)

        return normalized_error, threshold
    
    def runDeepAutoEncoder(self, ScaledMLInputdata, rawinputdata):
        """
        Use a trained autoencoder model to detect anomalies in the data.

        :param ScaledMLInputdata: Scaled input data for anomaly detection.
        :param rawinputdata: DataFrame containing previously detected anomalies.
        :return: Returns updated versions of ScaledMLInputdata and rawinputdata, with additional columns for autoencoder-based anomaly detection.
        """
        normalized_error, threshold = self.trainAutoEncoder(ScaledMLInputdata)


        print("Autoencoder training successful.")
        rawinputdata['Autoencoder_Anomaly'] = np.where(normalized_error > threshold, 1, 0)  # 1 is for anomaly and 0 for not_anomaly
        rawinputdata['Autoencoder_reconstruction_error'] = normalized_error.tolist()

        return ScaledMLInputdata, rawinputdata

    def run_anomaly_detection_models(self, raw_data, scaled_data):
        """
        Run anomaly detection models on both scaled and raw data and return the results.

        :param raw_data: Original input data.
        :param scaled_data: Preprocessed and scaled input data.
        :return: DataFrame containing anomaly detection results.
        """
        ScaledMLInputdata, rawinputdata  = self.sklearn_models(raw_data, scaled_data)
        RETScaledMLInputdata, RETrawinputdata = self.runDeepAutoEncoder(ScaledMLInputdata, rawinputdata)

        return RETScaledMLInputdata, RETrawinputdata