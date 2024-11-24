import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, auc, roc_curve, confusion_matrix
import plotly.graph_objects as go
from sklearn.preprocessing import label_binarize


def load_and_split_data(filepath):
    """
    Load the data from the given file and split it into training, validation, and testing sets.
    """
    features_to_drop = ['Cancer Type', 'Cancer Type Detailed', 'Tumor Stage', 'Sample Type', 'PATIENT_ID']
    data = pd.read_csv(filepath)
    cancer_types = data["Cancer Type"].unique()

    # Convert object columns to categorical
    object_columns = data.select_dtypes(include='object').columns
    data[object_columns] = data[object_columns].astype('category')

    # Encode categorical columns using cat.codes
    for col in data.select_dtypes(include='category').columns:
        data[col] = data[col].cat.codes

    # Separate features and labels
    X = data.drop(features_to_drop, axis=1)
    y, uniques = pd.factorize(data['Cancer Type'])
    label_dict = {cancer: idx for idx, cancer in enumerate(cancer_types)}
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, label_dict


def fit_and_evaluate_model(X_train, X_val, y_train, y_val, label_dict):
    """
    Fit the XGBoost model to the training data and evaluate it on the validation data.
    """
    xtra_cheese = xgb.XGBClassifier(n_estimators=250, objective='multi:softmax',
                                    tree_method='hist', enable_categorical=True)
    xtra_cheese.fit(X_train, y_train)
    y_pred = xtra_cheese.predict(X_val)

    # Feature importance
    feature_important = xtra_cheese.get_booster().get_score(importance_type='weight')
    keys = list(feature_important.keys())
    values = list(feature_important.values())

    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
    data.nlargest(40, columns="score").plot(kind='barh', figsize=(20, 10))  ## plot top 40 features

    # Feature importance plot
    plt.figure(figsize=(20, 16))
    xgb.plot_importance(xtra_cheese, max_num_features=10)
    plt.show()

    # Binarize the output labels for multiclass ROC computation
    classes = np.unique(y_train)
    y_val_bin = label_binarize(y_val, classes=classes)
    y_proba = xtra_cheese.predict_proba(X_val)

    # Plot ROC curve using Plotly
    fig_roc = go.Figure()
    for i, class_label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)

        # Reverse the label_dict to get cancer type names from numeric labels
        reversed_label_dict = {v: k for k, v in label_dict.items()}

        # Add ROC curve to the plot for each class
        fig_roc.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{reversed_label_dict[class_label]} (AUC = {roc_auc:.2f})'))

    # Add chance line
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Chance'))

    fig_roc.update_layout(
        title="Multiclass ROC Curve - Validation",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        showlegend=True
    )
    fig_roc.show()

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred, normalize='true')

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
        title="Confusion Matrix - Validation",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        xaxis=dict(tickmode='array', tickvals=np.arange(len(display_labels))),
        yaxis=dict(tickmode='array', tickvals=np.arange(len(display_labels))),
    )

    fig_cm.update_xaxes(tickangle=45)
    fig_cm.show()

    return accuracy_score(y_val, y_pred), xtra_cheese, y_pred


# if __name__ == "__main__":
#     filepath = "pan_cancer_data_for_model.csv"
#     X_train, X_val, X_test, y_train, y_val, y_test, label_dict = load_and_split_data(filepath)
#     accuracy = fit_and_evaluate_model(X_train, X_val, y_train, y_val, label_dict)
#     print(f"Validation Accuracy: {accuracy}")
