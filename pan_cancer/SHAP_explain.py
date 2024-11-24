import shap
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def shap_analysis(model, X_val, y_val, y_pred, label_dict):
    """
    Perform SHAP analysis for a multiclass classification model.
    """
    reversed_label_dict = {v: k for k, v in label_dict.items()}
    # Initialize SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values for multiclass (shape: [n_samples, n_features, n_classes])
    shap_values = explainer.shap_values(X_val)  # Already in (n_samples, n_features, n_classes)

    if len(shap_values.shape) != 3:
        raise ValueError("Expected SHAP values to have 3 dimensions (samples, features, classes).")

    # Correct predictions
    correct_indices = (y_val == y_pred)  # Boolean array of shape (n_samples,)

    # Select only correct predictions
    shap_values_correct = shap_values[correct_indices, :, :]  # Filter by correct samples

    # Aggregate SHAP values across correct predictions
    mean_shap_correct = np.mean(np.abs(shap_values_correct), axis=0)  # Mean across samples
    mean_shap_features = np.mean(mean_shap_correct, axis=1)  # Mean across classes

    # Create a feature importance DataFrame
    feature_importance_correct = pd.DataFrame({
        'Feature': X_val.columns,
        'Mean SHAP Value': mean_shap_features
    }).sort_values(by='Mean SHAP Value', ascending=False)

    print("Top features for correct predictions:")
    print(feature_importance_correct.head(10))

    # Generate a SHAP summary plot for each class
    for i in range(shap_values.shape[2]):  # Iterate over classes
        shap.summary_plot(shap_values[:, :, i], X_val, show=False)
        plt.title(f"SHAP Summary for Class {reversed_label_dict[i]}")
        plt.show()
        # plt.savefig(f"figures/shap_summary_class_{reversed_label_dict[i]}.png")

    return feature_importance_correct



