"""
AdaBoost Model for Pan-Cancer Analysis, using the TCGA dataset.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go


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
    ada_yonat = AdaBoostClassifier(n_estimators=250, random_state=42)
    ada_yonat.fit(X_train, y_train)
    y_pred = ada_yonat.predict(X_val)
    # todo: add RFECV?
    return accuracy_score(y_val, y_pred)


if __name__ == "__main__":
    filepath = "pan_cancer_data_for_model.csv"
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(filepath)
    accuracy = fit_and_evaluate_model(X_train, X_val, y_train, y_val)
    print(f"Validation Accuracy: {accuracy}")

