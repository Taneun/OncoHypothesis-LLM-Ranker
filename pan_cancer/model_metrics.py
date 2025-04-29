from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    precision_recall_curve
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import make_scorer
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt


def fit_and_evaluate(model, X_train, X_test, y_train, y_test, print_eval=True):
    """
    Fit the DecisionTree model to the training data and evaluate it on the validation data.
    """
    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model (optional printing)
    if print_eval:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        print("\n****** Model Evaluation ******\n")
        print(f"    Test Accuracy: {accuracy:.4f}")
        print(f"    Test Precision: {precision:.4f}")
        print(f"    Test Recall: {recall:.4f}")
        print(f"    Test F1 Score: {f1:.4f}")

    return model, y_pred

def plot_roc_curve(classes, label_dict, model_type, y_test_bin, y_proba, show_auc):
    """
    Plot the ROC curve for each class.
    """
    if show_auc:
        fig_roc = go.Figure()
        for i, class_label in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            reversed_label_dict = {v: k for k, v in label_dict.items()}
            fig_roc.add_trace(
                go.Scatter(x=fpr, y=tpr, mode='lines',
                           name=f'{reversed_label_dict[class_label]} (AUC = {roc_auc:.2f})'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Chance'))
        fig_roc.update_layout(
            title=f"{model_type} Multiclass ROC Curve - Test",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            showlegend=True,
            font=dict(size=18)
        )
        fig_roc.show()
        fig_roc.write_image(f"figures/{model_type}_roc.png", width=1600, height=900, scale=3)

def plot_confusion_matrix(classes, label_dict, model_type, y_test, y_pred, show_cm):
    """
    Plot the confusion matrix.
    """
    if show_cm:
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        reversed_label_dict = {v: k for k, v in label_dict.items()}
        display_labels = [reversed_label_dict[i] for i in range(len(label_dict))]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=display_labels,
            y=display_labels,
            colorscale='Blues',
            colorbar=dict(title='Normalized Count'),
        ))
        fig_cm.update_layout(
            title=f"{model_type} Confusion Matrix - Test",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            xaxis=dict(tickmode='array', tickvals=np.arange(len(display_labels))),
            yaxis=dict(tickmode='array', tickvals=np.arange(len(display_labels))),
            font=dict(size=18)
        )
        fig_cm.update_xaxes(tickangle=45)
        fig_cm.show()
        fig_cm.write_image(f"figures/{model_type}_cm.png", width=1600, height=900, scale=3)

def plot_precision_recall_curve(classes, label_dict, model_type, y_test_bin, y_proba, show_precision_recall):
    """
    Plot the Precision-Recall curve for each class.
    """
    if show_precision_recall:
        fig_pr_rc = go.Figure()
        for i, class_label in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
            pr_rc_auc = auc(recall, precision)
            reversed_label_dict = {v: k for k, v in label_dict.items()}
            fig_pr_rc.add_trace(
                go.Scatter(x=recall, y=precision, mode='lines',
                           name=f'{reversed_label_dict[class_label]} (AUC = {pr_rc_auc:.2f})'))
        fig_pr_rc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Chance'))
        fig_pr_rc.update_layout(
            title=f"{model_type} Multiclass Precision-Recall AUC - Test",
            xaxis_title="Recall Rate",
            yaxis_title="Precision Rate",
            showlegend=True,
            font=dict(size=18)
        )
        fig_pr_rc.show()
        fig_pr_rc.write_image(f"figures/{model_type}_pr_rc.png", width=1600, height=900, scale=3)


def generate_performance_plots(model, X_test, y_test, label_dict, model_type="Model", show_auc=False, show_cm=False,
                               show_precision_recall=False):
    """
    Generate performance plots for the given model, including ROC curve, confusion matrix, and precision-recall curve.
    """
    # Predict the probabilities for the test data
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # Binarize the labels for multiclass metrics
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)

    # Plot ROC Curve
    if show_auc:
        plot_roc_curve(classes, label_dict, model_type, y_test_bin, y_proba, show_auc)

    # Plot Confusion Matrix
    if show_cm:
        plot_confusion_matrix(classes, label_dict, model_type, y_test, y_pred, show_cm)

    # Plot Precision-Recall Curve
    if show_precision_recall:
        plot_precision_recall_curve(classes, label_dict, model_type, y_test_bin, y_proba, show_precision_recall)