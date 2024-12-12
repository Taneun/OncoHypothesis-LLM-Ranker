from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def fit_and_evaluate_tree(X_train, X_test, y_train, y_test, label_dict, show_plots=False):
    """
    Fit the DecisionTree model to the training data and evaluate it on the validation data.
    """
    # Initialize the model
    model = DecisionTreeClassifier(random_state=39)

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    if show_plots:
        # Binarize the output labels for multiclass ROC computation
        classes = np.unique(y_train)
        y_test_bin = label_binarize(y_test, classes=classes)
        y_proba = model.predict_proba(X_test)

        # Plot ROC curve using Plotly
        fig_roc = go.Figure()
        for i, class_label in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)

            # Reverse the label_dict to get cancer type names from numeric labels
            reversed_label_dict = {v: k for k, v in label_dict.items()}

            # Add ROC curve to the plot for each class
            fig_roc.add_trace(
                go.Scatter(x=fpr, y=tpr, mode='lines',
                           name=f'{reversed_label_dict[class_label]} (AUC = {roc_auc:.2f})'))

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

    return model, y_pred