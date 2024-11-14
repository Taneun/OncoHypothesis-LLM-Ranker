"""
XGBoost model for pan-cancer classification
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score

def load_and_split_data(filepath):
    """
    Load the data from the given file and split it into training, validation and testing sets.
    The labels are the "Cancer Type" column. The data was already preprocessed.
    """
    features_to_drop = ['Cancer Type', 'Cancer Type Detailed', 'Tumor Stage', 'Sample Type', 'PATIENT_ID']
    data = pd.read_csv(filepath)
    X = data.drop(features_to_drop, axis=1)
    y = data['Cancer Type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_and_evaluate_model(X_train, X_val, y_train, y_val):
    """
    Fit the AdaBoost model to the training data and evaluate it on the validation data.
    """
    # dtrain_clf = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    # dval_clf = xgb.DMatrix(X_val, y_val, enable_categorical=True)
    xtra_cheese = xgb.XGBClassifier(n_estimators=250, objective='multi:softmax',
                                   tree_method='hist', enable_categorical=True)
    xtra_cheese.fit(X_train, y_train)
    y_pred = xtra_cheese.predict(X_val)
    return accuracy_score(y_val, y_pred)


if __name__ == "__main__":
    filepath = "data/pan_cancer.csv"
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(filepath)
    accuracy = fit_and_evaluate_model(X_train, X_val, y_train, y_val)
    print(f"Validation Accuracy: {accuracy}")