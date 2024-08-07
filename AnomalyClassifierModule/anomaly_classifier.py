import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, make_scorer, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
class AnomalyClassifier:
    def __init__(self):
        pass

    def perform_grid_search_with_lgbm(self, X_train, y_train, scoring_metric='roc_auc', cv_folds=3):
        param_grid = {
            'num_leaves': [31, 50, 100],
            'is_unbalance': [True],
            'learning_rate': [0.01, 0.1, 0.05],
            'n_estimators': [100, 200, 500]
        }

        classifier = LGBMClassifier(objective='binary')

        scoring = make_scorer(roc_auc_score) if scoring_metric == 'roc_auc' else scoring_metric

        grid_search = GridSearchCV(classifier, param_grid, cv=cv_folds, scoring=scoring, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_

    def optimize_xgboost(self, X_train, y_train):
        lgbmc_classifier = LGBMClassifier(is_unbalance=True, learning_rate=0.1, objective='binary').fit(X_train, y_train)
        # best_estimator_, best_score_, best_params_ = perform_grid_search_with_lgbm(X_train, y_train)

        return lgbmc_classifier

    def anomaly_classifier(self, raw_data, main_datetime_col, col):
        keywords = ['Anomaly', 'error', 'anomaly', 'severity', 'distance', 'cluster']

        anomaly_cols = [col for col in raw_data.columns if any(keyword in col.lower() for keyword in keywords)]
        y = raw_data[col]
        X = raw_data.drop(columns=anomaly_cols)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        best_model = self.optimize_xgboost(x_train, y_train)
        test_prediction = x_test.copy()

        y_pred = best_model.predict(x_test)
        y_pred_prob = best_model.predict_proba(x_test)[:, 1]

        test_prediction['Anomaly_Labels'] = y_test
        test_prediction['Anomaly_Predictions'] = y_pred
        test_prediction['Anomaly_PredictedProb'] = y_pred_prob

        test_cm = confusion_matrix(test_prediction['Anomaly_Labels'], test_prediction['Anomaly_Predictions'])
        test_classreport = classification_report(test_prediction['Anomaly_Labels'], test_prediction['Anomaly_Predictions'], output_dict=True)
        accuracy = accuracy_score(test_prediction['Anomaly_Labels'], test_prediction['Anomaly_Predictions'])

        return test_prediction, best_model, x_test, y_test, test_cm, test_classreport, accuracy

    def feature_selection(self, X, y, n_features=None):
        anova_selector = SelectKBest(score_func=f_classif, k='all').fit(X, y)

        rf_model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
        rf_importance = rf_model.feature_importances_

        mi_scores = mutual_info_classif(X, y)

        feature_scores = pd.DataFrame({
            'ANOVA': anova_selector.scores_,
            'RandomForest': rf_importance,
            'MutualInformation': mi_scores}, index=X.columns)

        scaler = StandardScaler()
        feature_scores_scaled = pd.DataFrame(scaler.fit_transform(feature_scores), columns=feature_scores.columns, index=feature_scores.index)
        feature_scores['feature_importance'] = feature_scores_scaled.sum(axis=1)

        selected_features = feature_scores.nlargest(n_features if n_features else X.shape[1], 'feature_importance').index.tolist()

        return selected_features, feature_scores

    def compute_feature_values(self, X_train, y_train, n_features=None):
        selected_features, feature_scores = self.feature_selection(X_train, y_train, n_features=n_features)
        return selected_features, feature_scores