import shap
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter


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

    feature_importance_correct = extract_top_features(shap_values_correct, X_val[correct_indices], print_table=False)

    # Generate a SHAP summary plot for each class
    # for i in range(shap_values.shape[2]):  # Iterate over classes
    #     shap.summary_plot(shap_values[:, :, i], X_val, show=False)
    #     plt.title(f"SHAP Summary for Class {reversed_label_dict[i]}")
    #     plt.show()
    #     plt.savefig(f"figures/shap_summary_class_{reversed_label_dict[i]}.png")

    return feature_importance_correct


def extract_top_features(shap_values, correct_X, print_table=True, num_to_print=10):
    """
    Compute the top features for correct predictions based on SHAP values.
    """
    mean_shap_correct = np.mean(np.abs(shap_values), axis=0)  # Mean across samples
    mean_shap_features = np.mean(mean_shap_correct, axis=1)  # Mean across classes

    # Create a feature importance DataFrame
    feature_importance_correct = pd.DataFrame({
        'Feature': correct_X.columns,
        'Mean SHAP Value': mean_shap_features
    }).sort_values(by='Mean SHAP Value', ascending=False)

    if print_table:
        print("Top features for correct predictions:")
        print(feature_importance_correct.head(num_to_print))

    return feature_importance_correct


def generate_hypotheses_db(model, X, y_true, label_dict, top_k_features=4, min_support=2):
    """
    Generate a hypotheses database from an XGBoost model's correct predictions.

    Parameters:
    - model: Trained XGBoost model.
    - X: DataFrame of features for validation samples.
    - y_true: Array of true cancer type labels.
    - label_dict: Dictionary mapping class indices to cancer types.
    - top_k_features: Number of top features to include in hypotheses.
    - min_support: Minimum number of occurrences for a hypothesis to be included.

    Returns:
    - DataFrame of hypotheses with feature-value combinations and cancer types.
    """
    # Reverse label dictionary
    reversed_label_dict = {v: k for k, v in label_dict.items()}

    # Get predictions
    y_pred = model.predict(X)

    # Filter correct predictions
    correct_indices = y_pred == y_true
    correct_X = X[correct_indices]
    print("correctX " + str(len(list(correct_X.columns))))
    correct_y = y_true[correct_indices]

    # Compute SHAP values for correct predictions
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(correct_X)

    extract_top_features(shap_values, correct_X)

    # Store hypotheses
    hypotheses = []
    top_feat = set()

    # Iterate over correct predictions
    for i, sample_idx in enumerate(correct_X.index):
        cancer_type = reversed_label_dict[correct_y[i]]
        sample_shap = shap_values[i, :, :]  # Extract SHAP values for the i-th sample across all features and classes

        # Get top K features contributing to the prediction
        top_features_indices = sample_shap[correct_y[i]].argsort()[-top_k_features:][::-1]
        top_features = [(X.columns[idx], correct_X.iloc[i, idx]) for idx in top_features_indices]
        for f in top_features:
            top_feat.add(f[0])

        # Format hypothesis
        hypothesis = {f"{feature}": value for feature, value in top_features}
        hypothesis["cancer_type"] = cancer_type
        hypotheses.append(frozenset(hypothesis.items()))
    # todo: remove all the debugging prints
    print(len(top_feat))
    print(top_feat)
    columns_missing = [col for col in list(correct_X.columns) if col not in list(top_feat)]
    print(f"columns missing: {columns_missing}")
    print("hypo " + str(len(list(hypotheses[0]))))
    # Count occurrences of each hypothesis
    hypothesis_counts = Counter(hypotheses)

    # Filter hypotheses by support
    filtered_hypotheses = [
        dict(hypo) for hypo, count in hypothesis_counts.items() if count >= min_support
    ]

    # Create hypotheses database
    hypotheses_db = pd.DataFrame(filtered_hypotheses).fillna("")
    hypotheses_db["support"] = [
        hypothesis_counts[frozenset(hypo.items())] for hypo in filtered_hypotheses
    ]

    return hypotheses_db.sort_values(by="support", ascending=False)
