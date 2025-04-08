import shap
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections import Counter
from XGBoost_Model import apply_category_mappings
import torch


def shap_analysis(explainer, X_val, y_val, y_pred, label_dict):
    """
    Perform SHAP analysis for a multiclass classification model.
    """
    # Compute SHAP values for multiclass (shape: [n_samples, n_features, n_classes])
    shap_values = explainer.shap_values(X_val)  # Already in (n_samples, n_features, n_classes)

    if len(shap_values.shape) != 3:
        raise ValueError("Expected SHAP values to have 3 dimensions (samples, features, classes).")

    # Correct predictions
    correct_indices = (y_val == y_pred)  # Boolean array of shape (n_samples,)

    # Select only correct predictions
    shap_values_correct = shap_values[correct_indices, :, :]  # Filter by correct samples

    feature_importance_correct = extract_top_features(shap_values_correct, X_val[correct_indices], print_table=True)

    # Generate a SHAP summary plot for each class
    # for i in range(shap_values.shape[2]):  # Iterate over classes
    #     shap.summary_plot(shap_values[:, :, i], X_val, show=False)
    #     plt.title(f"SHAP Summary for Class {reversed_label_dict[i]}")
    #     plt.show()
    #     plt.savefig(f"figures/shap_summary_class_{reversed_label_dict[i]}.png")

    return feature_importance_correct


def extract_top_features(shap_values, correct_X, print_table=False, num_to_print=10):
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


def get_shap_interactions(explainer, X, y, label_dict, batch_size=100):
    """
    Get SHAP interaction values for a multiclass classification model using batching.
    Shows progress with tqdm.
    """
    from tqdm import tqdm

    reversed_label_dict = {v: k for k, v in label_dict.items()}
    n_samples = len(X)
    n_classes = len(label_dict)
    n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division

    # Process data in batches
    all_interactions = []

    # Create tqdm progress bar
    for start_idx in tqdm(range(0, n_samples, batch_size),
                          total=n_batches,
                          desc="Computing SHAP interactions"):
        # Clear GPU memory before each batch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        end_idx = min(start_idx + batch_size, n_samples)
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]

        try:
            # Compute SHAP interaction values for the batch
            batch_interactions = explainer.shap_interaction_values(X_batch, y_batch)
            all_interactions.append(batch_interactions)
        except Exception as e:
            print(f"Error processing batch {start_idx // batch_size}: {str(e)}")
            continue

    # Combine all batches
    try:
        shap_interaction_values = np.concatenate(all_interactions, axis=0)
    except Exception as e:
        raise ValueError(f"Failed to combine interaction values: {str(e)}")

    if len(shap_interaction_values.shape) != 4:
        raise ValueError(
            "Expected SHAP interaction values to have 4 dimensions (samples, features, features, classes).")

    # Generate plots with memory cleanup between classes
    for i in range(n_classes):
        # Clear memory before each plot
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_interaction_values[:, :, :, i], X, show=False)
        plt.title(f"SHAP Interaction Summary for Class {reversed_label_dict[i]}")
        plt.savefig(f"figures/shap_interaction_summary_class_{reversed_label_dict[i]}.png")
        plt.close()  # Explicitly close the figure to free memory

    return shap_interaction_values



def generate_hypotheses_db(explainer, model, X, y_true, label_dict, mapping,
                           min_features=3, relative_threshold_percent=10, min_support=3):
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
    hypotheses_db = generate_raw_df(X, explainer, label_dict, min_features, min_support, model,
                                    relative_threshold_percent, y_true)
    hypotheses_db = apply_category_mappings(hypotheses_db, mapping) # here
    hypotheses_db = hypotheses_db.sort_values(["cancer_type", 'support'], ascending=[True, False])
    hypotheses_db = generate_sentences(hypotheses_db)
    return hypotheses_db


def calculate_shap_in_batches(explainer, X, batch_size=256):
    """
    Calculate SHAP values in batches with a progress bar.

    Parameters:
    explainer: SHAP explainer object
    X: Input data
    batch_size: Size of each batch

    Returns:
    numpy array: Concatenated SHAP values
    """
    shap_list = []

    # Calculate number of batches
    n_samples = X.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))

    # Create batches
    for i in tqdm(range(n_batches), desc="Calculating SHAP values", unit="batch", colour='blue'):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)

        # Calculate SHAP values for batch
        batch_shap = explainer.shap_values(X[start_idx:end_idx])
        shap_list.append(batch_shap)

    # Concatenate all batches
    return np.concatenate(shap_list, axis=0)


def generate_raw_df(X, explainer, label_dict, min_features, min_support, model, relative_threshold_percent, y_true):
    # Reverse label dictionary
    reversed_label_dict = {v: k for k, v in label_dict.items()}
    # Get predictions
    y_pred = model.predict(X)
    # Filter correct predictions
    correct_indices = y_pred == y_true
    correct_X = X[correct_indices]
    correct_y = y_true[correct_indices]
    # Compute SHAP values for correct predictions
    shap_values = calculate_shap_in_batches(explainer, correct_X)
    extract_top_features(shap_values, correct_X)
    # Store hypotheses
    hypotheses = []
    top_feat = set()
    # Iterate over correct predictions
    for i, sample_idx in tqdm(enumerate(correct_X.index),
                              desc="Generating hypotheses from correct predictions"
                              , unit="sample", colour='green'):
        cancer_type = reversed_label_dict[correct_y[i]]
        sample_shap = shap_values[i, :, :]  # Extract SHAP values for the i-th sample across all features and classes
        sample_shap = sample_shap.transpose()[correct_y[i]]

        total_contribution = np.sum(sample_shap)
        contribution_cutoff = (relative_threshold_percent / 100) * total_contribution
        significant_indices = np.where(sample_shap >= contribution_cutoff)[0]

        if len(significant_indices) < min_features:
            top_k_features = min_features
        else:
            top_k_features = len(significant_indices)
        # Get top K features contributing to the prediction, adding 1 to handle smoke status
        top_features_indices = sample_shap.argsort()[-(top_k_features + 1):][::-1]
        top_features = [(X.columns[idx], correct_X.iloc[i, idx]) for idx in top_features_indices]
        # Remove the feature Smoke Status ifs its unknown and add another in its place if it exists
        if ("Smoke Status", 2) in top_features:
            top_features.remove(("Smoke Status", 2))
        else:
            # if Smoke Status is not in the top features, remove the extra feature
            top_features = top_features[:-1]
        for f in top_features:
            top_feat.add(f[0])

        # Format hypothesis
        hypothesis = {f"{feature}": value for feature, value in top_features}
        hypothesis["cancer_type"] = cancer_type
        hypotheses.append(frozenset(hypothesis.items()))
    # Count occurrences of each hypothesis
    hypothesis_counts = Counter(hypotheses)
    # Filter hypotheses by support
    filtered_hypotheses = [
        dict(hypo) for hypo, count in hypothesis_counts.items() if count >= min_support
    ]
    hypotheses_db = pd.DataFrame(filtered_hypotheses).fillna("")
    hypotheses_db["support"] = [
        hypothesis_counts[frozenset(hypo.items())] for hypo in filtered_hypotheses
    ]
    return hypotheses_db


def generate_sentences(df):
    """
    Generate sentences from a DataFrame of hypotheses.
    """
    sentences = []

    for _, row in df.iterrows():
        sentence_parts = []
        for col in df.columns:
            if col in ["cancer_type", "support"]:
                continue

            value = row[col]
            if pd.isna(value) or value == "" or value is None:
                # Skip NaN values
                continue
            elif value in [0, 1]:
                sentence_parts.append(f"{'is' if value == 1 else 'is NOT'} {col}")
            else:
                sentence_parts.append(f"{col} value is {value}")

        sentence = " AND ".join(sentence_parts).replace("_", " ")
        sentences.append(sentence)

    df['hypothesis'] = sentences
    return df
