import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold
import xgboost as xgb
from sklearn.metrics import accuracy_score, auc, roc_curve, confusion_matrix, precision_score, recall_score, f1_score
import plotly.graph_objects as go
from sklearn.preprocessing import label_binarize


def load_data(filepath):
    features_to_drop = ['Cancer Type', 'Cancer Type Detailed', 'Tumor Stage', 'Sample Type']
    data = pd.read_csv(filepath)
    cancer_types = data["Cancer Type"].unique()
    mapping = {}
    # Convert object columns to categorical
    object_columns = data.select_dtypes(include=['object', 'bool']).columns
    for col in object_columns:
        mapping[col] = dict(enumerate(data[col].astype('category').cat.categories))
    data[object_columns] = data[object_columns].astype('category')

    # Encode categorical columns using cat.codes
    for col in data.select_dtypes(include='category').columns:
        data[col] = data[col].cat.codes

    # Separate features and labels
    X = data.drop(features_to_drop, axis=1)
    y, uniques = pd.factorize(data['Cancer Type'])
    label_dict = {cancer: idx for idx, cancer in enumerate(cancer_types)}
    return X, y, label_dict, mapping


def apply_category_mappings(data, mappings):
    """
    Apply saved mappings to translate codes back to original values.

    Parameters:
        data (DataFrame): DataFrame with encoded categorical columns.
        mappings (dict): Mappings dictionary from save_category_mappings.

    Returns:
        DataFrame: DataFrame with original values restored.
    """
    for col in mappings.keys():
        if col not in data.columns:
            continue
        data[col] = data[col].map(mappings[col])
    return data


def split_data(X, y):
    """
    Split data into training, validation, and testing sets.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def stratified_split_by_patient(X, y, train_ratio=0.7, test_ratio=0.3):
    """
    Split data into training and testing sets with stratification by PATIENT_ID.
    """
    # Ensure the ratios sum to 1
    assert train_ratio + test_ratio == 1, "Ratios must sum to 1."

    # Get unique patient IDs
    unique_ids = X['PATIENT_ID'].unique()

    # Map PATIENT_ID to a corresponding target value (first occurrence)
    patient_labels = dict(zip(X['PATIENT_ID'], y))
    unique_patient_labels = [patient_labels[pid] for pid in unique_ids]

    # Initial split: train+val and test
    train_ids, test_ids = train_test_split(
        unique_ids,
        test_size=test_ratio,
        stratify=unique_patient_labels,
        random_state=42
    )

    # Split data into subsets
    X_train = X[X['PATIENT_ID'].isin(train_ids)].drop(columns=['PATIENT_ID'])
    # X_val = X[X['PATIENT_ID'].isin(val_ids)].drop(columns=['PATIENT_ID'])
    X_test = X[X['PATIENT_ID'].isin(test_ids)].drop(columns=['PATIENT_ID'])
    X_test_with_id = X[X['PATIENT_ID'].isin(test_ids)]  # Keep validation set with PATIENT_ID for patient-level analysis

    y_train = y[X['PATIENT_ID'].isin(train_ids)]
    # y_val = y[X['PATIENT_ID'].isin(val_ids)]
    y_test = y[X['PATIENT_ID'].isin(test_ids)]

    return X_train, X_test, y_train, y_test, X_test_with_id


def fit_and_evaluate_model(X_train, X_test, y_train, y_test, label_dict, show_plots=False):
    """
    Fit the XGBoost model to the training data and evaluate it on the validation data.
    """
    xtra_cheese = xgb.XGBClassifier(n_estimators=250, objective='multi:softmax',
                                    tree_method='hist', enable_categorical=True)
    xtra_cheese.fit(X_train, y_train)
    y_pred = xtra_cheese.predict(X_test)

    if show_plots:
        # Feature importance
        feature_important = xtra_cheese.get_booster().get_score(importance_type='weight')
        keys = list(feature_important.keys())
        values = list(feature_important.values())

        data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
        data.nlargest(15, columns="score").plot(kind='barh', figsize=(20, 10)) # top 15 features

        # Feature importance plot
        # plt.figure(figsize=(20, 16))
        # xgb.plot_importance(xtra_cheese, max_num_features=10)
        # plt.show()

        # Binarize the output labels for multiclass ROC computation
        classes = np.unique(y_train)
        y_test_bin = label_binarize(y_test, classes=classes)
        y_proba = xtra_cheese.predict_proba(X_test)

        # Plot ROC curve using Plotly
        fig_roc = go.Figure()
        for i, class_label in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)

            # Reverse the label_dict to get cancer type names from numeric labels
            reversed_label_dict = {v: k for k, v in label_dict.items()}

            # Add ROC curve to the plot for each class
            fig_roc.add_trace(
                go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{reversed_label_dict[class_label]} (AUC = {roc_auc:.2f})'))

        # Add chance line
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Chance'))

        fig_roc.update_layout(
            title="Multiclass ROC Curve - Test",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            showlegend=True
        )
        fig_roc.show()

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, normalize='true')

        # Map numeric indices to cancer type names
        reversed_label_dict = {v: k for k, v in label_dict.items()}
        display_labels = [reversed_label_dict[i] for i in range(len(label_dict))]

        # Plot Confusion Matrix using Plotly
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=display_labels,
            y=display_labels,
            colorscale='Blues',
            colorbar=dict(title='Normalized Count'),
        ))

        fig_cm.update_layout(
            title="Confusion Matrix - Test",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            xaxis=dict(tickmode='array', tickvals=np.arange(len(display_labels))),
            yaxis=dict(tickmode='array', tickvals=np.arange(len(display_labels))),
        )

        fig_cm.update_xaxes(tickangle=45)
        fig_cm.show()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    return xtra_cheese, y_pred


def classify_patients(X, y_pred, y_true, label_dict):
    """
    Classify patients using the trained model and generate a confusion matrix.
    """

    # Map numeric indices to cancer type names
    reversed_label_dict = {v: k for k, v in label_dict.items()}

    # Create a DataFrame to associate patient IDs with predictions and true labels
    patients_by_predictions = pd.DataFrame({
        'PATIENT_ID': X['PATIENT_ID'],
        'Predicted Cancer Type': [reversed_label_dict[y] for y in y_pred],
        'True Cancer Type': [reversed_label_dict[y] for y in y_true]
    })

    # Majority vote for each patient
    patient_predictions = patients_by_predictions.groupby('PATIENT_ID').agg(
        {
            'Predicted Cancer Type': lambda x: x.mode().iloc[0],  # Majority vote for predictions
            'True Cancer Type': 'first'  # Assume true label is the same for all entries
        }
    ).reset_index()

    # Map cancer types to integers for confusion matrix
    unique_cancer_types = list(reversed_label_dict.values())
    type_to_int = {cancer_type: i for i, cancer_type in enumerate(unique_cancer_types)}

    # Convert labels to integers
    true_labels = patient_predictions['True Cancer Type'].map(type_to_int).to_numpy()
    pred_labels = patient_predictions['Predicted Cancer Type'].map(type_to_int).to_numpy()

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels, normalize='true')

    # Plot confusion matrix using Plotly
    fig_cmat = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=unique_cancer_types,
        y=unique_cancer_types,
        colorscale='Greens',
        colorbar=dict(title='Normalized Count'),
    ))

    fig_cmat.update_layout(
        title="Confusion Matrix - By Patient (Test)",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        xaxis=dict(tickmode='array', tickvals=np.arange(len(unique_cancer_types)), ticktext=unique_cancer_types),
        yaxis=dict(tickmode='array', tickvals=np.arange(len(unique_cancer_types)), ticktext=unique_cancer_types),
    )

    fig_cmat.update_xaxes(tickangle=45)
    fig_cmat.show()

    return patient_predictions, conf_matrix
