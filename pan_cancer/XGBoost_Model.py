"""
XGBoost model for pan-cancer classification
"""

import pandas as pd
from matplotlib import pyplot as plt
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
    for cat_col in ["Hugo_Symbol", "SNP_event", "Codons", "Exon_Number"]:
        X[cat_col] = X[cat_col].astype('category')
    y, uniques = pd.factorize(data['Cancer Type'])
    # Create a dictionary mapping from cancer type to label
    # label_dict = {cancer: idx for idx, cancer in enumerate(uniques)}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_and_evaluate_model(X_train, X_val, y_train, y_val):
    """
    Fit the XGBoost model to the training data and evaluate it on the validation data.
    """
    # dtrain_clf = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    # dval_clf = xgb.DMatrix(X_val, y_val, enable_categorical=True)
    xtra_cheese = xgb.XGBClassifier(n_estimators=250, objective='multi:softmax',
                                   tree_method='hist', enable_categorical=True)
    xtra_cheese.fit(X_train, y_train)
    y_pred = xtra_cheese.predict(X_val)
    # Get feature importances
    importance = xtra_cheese.feature_importances_
    # Create a dictionary mapping features to their importance
    feature_importance_dict = dict(zip(X_train.columns, importance))
    # Sort by importance
    sorted_feature_importance = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))
    print("Feature importances (sorted):", sorted_feature_importance)
    plt.figure(figsize=(20, 16))
    xgb.plot_importance(xtra_cheese, max_num_features=10)  # Adjust max_num_features as needed
    plt.show()
    return accuracy_score(y_val, y_pred)


if __name__ == "__main__":
    filepath = "/Users/talneumann/Downloads/pan_cancer_data_for_model.csv"
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(filepath)
    accuracy = fit_and_evaluate_model(X_train, X_val, y_train, y_val)
    print(f"Validation Accuracy: {accuracy}")