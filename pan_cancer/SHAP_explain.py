# === Imports ===
from XGBoost_Model import apply_category_mappings
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import torch
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


# === Memory Management ===
def safe_gpu_clear():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# === SHAP Value Calculation ===
def calculate_shap_in_batches(explainer, X, batch_size=256):
    shap_list = []
    n_samples = X.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))
    for i in tqdm(range(n_batches), desc="Calculating SHAP values", unit="batch", colour='blue'):
        start_idx, end_idx = i * batch_size, min((i + 1) * batch_size, n_samples)
        try:
            batch_shap = explainer.shap_values(X[start_idx:end_idx])
            shap_list.append(batch_shap)
        except Exception as e:
            print(f"Batch {i} failed: {e}")
        safe_gpu_clear()
    return np.concatenate(shap_list, axis=0)


def calculate_shap_interactions_in_batches(explainer, X, y, batch_size=256, save_path=None):
    all_interactions = []
    for i in tqdm(range(0, len(X), batch_size), desc="SHAP Interaction Batches"):
        batch_X = X.iloc[i:i + batch_size]
        try:
            batch_interactions = explainer.shap_interaction_values(batch_X)
            all_interactions.append(batch_interactions)
        except Exception as e:
            print(f"Batch {i // batch_size} failed: {e}")
        safe_gpu_clear()

    interactions = np.concatenate(all_interactions, axis=0)

    # Save interactions to a file for future use
    if save_path:
        np.save(save_path, interactions)
        print(f"SHAP interactions saved to {save_path}")

    return interactions


# === SHAP Feature Importance ===
def extract_top_features(shap_values, correct_X, print_table=False, num_to_print=10):
    mean_shap_correct = np.mean(np.abs(shap_values), axis=0)
    mean_shap_features = np.mean(mean_shap_correct, axis=1)
    feature_importance = pd.DataFrame({
        'Feature': correct_X.columns,
        'Mean SHAP Value': mean_shap_features
    }).sort_values(by='Mean SHAP Value', ascending=False)
    if print_table:
        print("Top features for correct predictions:")
        print(feature_importance.head(num_to_print))
    return feature_importance


def generate_raw_df(shap_values, X, y_true, label_dict, top_n=10):
    reversed_label_dict = {v: k for k, v in label_dict.items()}
    df_summary = pd.DataFrame()
    for class_idx in range(shap_values.shape[2]):
        top_features = np.argsort(np.abs(shap_values[:, :, class_idx]).mean(axis=0))[::-1][:top_n]
        for f_idx in top_features:
            mean_val = np.abs(shap_values[:, f_idx, class_idx]).mean()
            df_summary = pd.concat([
                df_summary,
                pd.DataFrame({
                    "class": [reversed_label_dict[class_idx]],
                    "feature": [X.columns[f_idx]],
                    "mean_abs_shap": [mean_val]
                })
            ])
    return df_summary.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)


# === SHAP Plotting ===
def efficient_shap_summary_plots(shap_values, X, label_dict, max_display=15):
    reversed_label_dict = {v: k for k, v in label_dict.items()}
    for class_idx in range(shap_values.shape[2]):
        # Create bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values[:, :, class_idx], X, plot_type='bar', show=False, max_display=max_display)
        plt.title(f"SHAP Feature Importance - {reversed_label_dict[class_idx]}", pad=20)
        plt.tight_layout()
        plt.savefig(f"figures/shap_summary_bar_{reversed_label_dict[class_idx]}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Create violin plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values[:, :, class_idx], X, plot_type='violin', show=False, max_display=max_display)
        plt.title(f"SHAP Value Distribution - {reversed_label_dict[class_idx]}", pad=20)
        plt.tight_layout()
        plt.savefig(f"figures/shap_summary_violin_{reversed_label_dict[class_idx]}.png", dpi=300, bbox_inches='tight')
        plt.close()

        safe_gpu_clear()


def generate_shap_interaction_heatmap(interactions, X, class_idx, label_dict):
    reversed_label_dict = {v: k for k, v in label_dict.items()}
    mean_interactions = np.abs(interactions[:, :, :, class_idx]).mean(axis=0)
    plt.figure(figsize=(12, 10))
    sns.heatmap(mean_interactions, xticklabels=X.columns, yticklabels=X.columns, cmap='coolwarm', square=True)
    plt.title(f"SHAP Interaction Heatmap - {reversed_label_dict[class_idx]}")
    plt.tight_layout()
    plt.savefig(f"figures/shap_interaction_heatmap_class_{reversed_label_dict[class_idx]}.png")
    plt.close()
    safe_gpu_clear()


def shap_dependence_top_interactions(interactions, X, class_idx, label_dict, top_k=3):
    reversed_label_dict = {v: k for k, v in label_dict.items()}
    interaction_matrix = np.abs(interactions[:, :, :, class_idx]).mean(axis=0)
    np.fill_diagonal(interaction_matrix, 0)
    top_pairs = np.dstack(np.unravel_index(np.argsort(interaction_matrix.ravel())[::-1], interaction_matrix.shape))[0]
    seen, count = set(), 0
    for i, j in top_pairs:
        if (i, j) not in seen and (j, i) not in seen and count < top_k:
            seen.add((i, j))
            try:
                x_feature = X.columns[i]
                y_feature = X.columns[j]

                # Get the interaction values for this pair
                interaction_values = interactions[:, i, j, class_idx]

                # Create a DataFrame with the feature values and interaction values
                plot_data = pd.DataFrame({
                    x_feature: X[x_feature].values,
                    'interaction': interaction_values
                })

                # Plot using seaborn for better control
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=plot_data, x=x_feature, y='interaction', alpha=0.5)
                plt.title(f"Dependence Plot ({x_feature} & {y_feature}) - {reversed_label_dict[class_idx]}")
                plt.xlabel(x_feature)
                plt.ylabel(f"SHAP Interaction with {y_feature}")
                plt.tight_layout()
                plt.savefig(f"figures/shap_dependence_{x_feature}_{y_feature}_{reversed_label_dict[class_idx]}.png")
                plt.close()
                safe_gpu_clear()
                count += 1
            except Exception as e:
                print(f"Could not plot dependence for {X.columns[i]} & {X.columns[j]}: {e}")


def plot_feature_importance_heatmap(shap_values, X, label_dict, mapping, top_n=10):
    """Create a heatmap showing top-N feature importance across all classes"""
    reversed_label_dict = {v: k for k, v in label_dict.items()}
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create DataFrame for heatmap: features as rows, classes as columns
    importance_df = pd.DataFrame(
        mean_abs_shap,
        index=X.columns,
        columns=[reversed_label_dict[i] for i in range(len(label_dict))]
    )

    # Sort by total importance and keep top-N features
    importance_df['total'] = importance_df.sum(axis=1)
    importance_df = importance_df.sort_values('total', ascending=False).drop('total', axis=1)
    top_features_df = importance_df.head(top_n)

    # Plot heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(top_features_df, cmap='YlOrRd', annot=True, fmt='.3f', cbar_kws={'label': 'Mean |SHAP value|'})
    plt.title(f'Top {top_n} Feature Importance Across Cancer Types', pad=20)
    plt.tight_layout()
    plt.savefig('figures/feature_importance_heatmap_top10.png', dpi=300, bbox_inches='tight')
    plt.close()
    safe_gpu_clear()


def plot_top_feature_distributions(shap_values, X, label_dict, mapping, top_n=5):
    """Plot distributions of top features for each class"""
    reversed_label_dict = {v: k for k, v in label_dict.items()}

    for class_idx in range(shap_values.shape[2]):
        # Get top features for this class
        mean_abs_shap = np.abs(shap_values[:, :, class_idx]).mean(axis=0)
        top_features = X.columns[np.argsort(-mean_abs_shap)[:top_n]]

        # Create subplots for each top feature
        fig, axes = plt.subplots(top_n, 1, figsize=(12, 4 * top_n))
        if top_n == 1:
            axes = [axes]

        for idx, feature in enumerate(top_features):
            # Plot feature distribution
            sns.kdeplot(data=X, x=feature, ax=axes[idx], fill=True)

            # Get the true label for the feature if it exists in mapping
            feature_label = feature
            if feature in mapping:
                feature_label = mapping[feature]

            axes[idx].set_title(
                f'Distribution of {feature_label} (SHAP Importance: {mean_abs_shap[X.columns.get_loc(feature)]:.3f})')
            axes[idx].set_xlabel(feature_label)
            axes[idx].set_ylabel('Density')

        plt.suptitle(f'Top {top_n} Feature Distributions for {reversed_label_dict[class_idx]}', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.savefig(f'figures/top_features_distribution_{reversed_label_dict[class_idx]}.png', dpi=300,
                    bbox_inches='tight')
        plt.close()
        safe_gpu_clear()


def plot_shap_dependence_plots(shap_values, X, label_dict, mapping, top_n=3):
    """Create detailed dependence plots for top features"""
    reversed_label_dict = {v: k for k, v in label_dict.items()}
    fig = plt.figure(figsize=(10, 6))

    for class_idx in range(shap_values.shape[2]):
        # Get top features for this class
        mean_abs_shap = np.abs(shap_values[:, :, class_idx]).mean(axis=0)
        top_features = X.columns[np.argsort(-mean_abs_shap)[:top_n]]

        for feature in top_features:
            fig.clear()
            # Get the true label for the feature if it exists in mapping
            feature_label = feature
            if feature in mapping:
                feature_label = mapping[feature]

            shap.dependence_plot(
                feature,
                shap_values[:, :, class_idx],
                X,
                show=False,
                alpha=0.8,
                dot_size=50
            )
            plt.title(f'SHAP Dependence Plot for {feature_label} - {reversed_label_dict[class_idx]}', pad=20)
            plt.xlabel(feature_label)
            plt.tight_layout()
            plt.savefig(f'figures/shap_dependence_{feature}_{reversed_label_dict[class_idx]}.png', dpi=300,
                        bbox_inches='tight')
            plt.close()
            safe_gpu_clear()

# def plot_decision_plots(explainer, X, y_true, y_pred, label_dict, model=None, sample_indices=None, max_samples=5,
#                         save_path=None):
#     """
#     Create SHAP multioutput decision plots for cancer type prediction with proper label handling.
#     """
#     # Create save directory if needed
#     if save_path and not os.path.exists(save_path):
#         os.makedirs(save_path)
#
#     # Create reversed label dictionary for displaying class names
#     reversed_label_dict = {v: k for k, v in label_dict.items()}
#     class_names = [reversed_label_dict[i] for i in range(len(label_dict))]
#
#     # Get base values from the explainer
#     base_values = explainer.expected_value
#     if not isinstance(base_values, list):
#         base_values = [base_values] * len(label_dict)
#
#     # Ensure we have prediction probabilities if needed for label annotation
#     pred_probs = None
#     if model is not None:
#         try:
#             pred_probs = model.predict_proba(X)
#         except:
#             print("Could not get prediction probabilities from model.")
#
#     # Select samples to visualize
#     if sample_indices is None:
#         # Find indices where predictions are correct
#         correct_indices = (y_pred == y_true)
#         correct_sample_indices = np.where(correct_indices)[0]
#
#         # Randomly select samples to visualize
#         if len(correct_sample_indices) > max_samples:
#             sample_indices = np.random.choice(correct_sample_indices, size=max_samples, replace=False)
#         else:
#             sample_indices = correct_sample_indices
#
#     # Convert to list if it's not already
#     if not isinstance(sample_indices, list):
#         sample_indices = sample_indices.tolist()
#
#     # Get feature names
#     feature_names = list(X.columns)
#
#     # Calculate SHAP values for selected samples
#     samples_to_explain = X.iloc[sample_indices]
#     shap_values = explainer.shap_values(samples_to_explain)
#
#     # Handle different SHAP value formats
#     if isinstance(shap_values, list):
#         # For some models like XGBoost, shap_values is a list of arrays
#         if len(shap_values) == 2 and len(label_dict) > 2:
#             # Binary classification case but with multiclass model
#             print("Detected binary format SHAP values for multiclass model")
#             # We need to recalculate with multi-output format
#             try:
#                 shap_values = explainer(samples_to_explain)
#             except:
#                 print("Could not recalculate SHAP values. Using available format.")
#
#     # Process each sample
#     for i, idx in enumerate(sample_indices):
#         sample_idx = i  # Index in the samples_to_explain dataframe
#         true_class = y_true[idx]
#         pred_class = y_pred[idx]
#
#         # Create legend labels with probabilities if available
#         if pred_probs is not None:
#             legend_labels = [
#                 f"{class_names[j]} ({pred_probs[idx, j]:.2f})"
#                 for j in range(len(class_names))
#             ]
#         else:
#             legend_labels = class_names
#
#         # Create the decision plot
#         plt.figure(figsize=(15, 10))
#
#         try:
#             shap.multioutput_decision_plot(
#                 base_values,
#                 shap_values,
#                 row_index=sample_idx,
#                 feature_names=feature_names,
#                 highlight=true_class,  # Highlight the true class
#                 legend_labels=legend_labels,
#                 legend_location="lower right",
#                 show=False
#             )
#
#             # Add informative title
#             plt.title(
#                 f"Cancer Type Decision Plot - Sample {idx}\n"
#                 f"True: {reversed_label_dict[true_class]} | "
#                 f"Predicted: {reversed_label_dict[pred_class]} "
#                 f"({'Correct' if true_class == pred_class else 'Incorrect'})",
#                 pad=20
#             )
#
#             plt.tight_layout()
#
#             # Save or show the plot
#             if save_path:
#                 plt.savefig(f"{save_path}/decision_plot_sample_{idx}.png", dpi=300, bbox_inches='tight')
#                 plt.close()
#             else:
#                 plt.show()
#
#         except Exception as e:
#             print(f"Error creating decision plot for sample {idx}: {e}")
#             plt.close()
#
#         # Clean up
#         safe_gpu_clear()

def plot_decision_plots(explainer, X, y_true, y_pred, label_dict, model=None, sample_indices=None, max_samples=5,
                        save_path=None):
    """
    Create SHAP multioutput decision plots for cancer type prediction with proper label handling.
    Fixed to handle the "base_values and shap_values args expect lists" error.

    Parameters:
    -----------
    explainer : shap.TreeExplainer
        The SHAP explainer object.
    X : pandas.DataFrame
        Feature data.
    y_true : numpy.ndarray
        True class labels (numeric).
    y_pred : numpy.ndarray
        Predicted class labels (numeric).
    label_dict : dict
        Dictionary mapping cancer type names to numeric indices.
    model : optional, trained model
        Used to get prediction probabilities if needed.
    sample_indices : list, optional
        Specific indices to plot. If None, will choose random correct predictions.
    max_samples : int, default=5
        Maximum number of samples to plot if sample_indices is None.
    save_path : str, optional
        Directory to save the plots. If None, plots will be displayed.
    """
    import shap
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # Create save directory if needed
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    # Create reversed label dictionary for displaying class names
    reversed_label_dict = {v: k for k, v in label_dict.items()}
    class_names = [reversed_label_dict[i] for i in range(len(label_dict))]

    # Select samples to visualize
    if sample_indices is None:
        # Find indices where predictions are correct
        correct_indices = (y_pred == y_true)
        correct_sample_indices = np.where(correct_indices)[0]

        # Randomly select samples to visualize
        if len(correct_sample_indices) > max_samples:
            sample_indices = np.random.choice(correct_sample_indices, size=max_samples, replace=False)
        else:
            sample_indices = correct_sample_indices

    # Convert to list if it's not already
    if not isinstance(sample_indices, list):
        sample_indices = sample_indices.tolist()

    # Get feature names
    feature_names = list(X.columns)

    # Get prediction probabilities if the model supports it
    pred_probs = None
    if model is not None:
        try:
            pred_probs = model.predict_proba(X.iloc[sample_indices])
        except Exception as e:
            print(f"Could not get prediction probabilities: {e}")

    # Process each sample individually
    for i, idx in enumerate(sample_indices):
        print(f"Processing sample {idx}...")
        # Extract the single sample to explain
        sample_to_explain = X.iloc[[idx]]

        try:
            # Get SHAP values specifically for this sample
            # This ensures we get the correct format
            sample_shap_values = explainer.shap_values(sample_to_explain)

            # Get base values (expected values)
            base_values = explainer.expected_value

            # Convert to lists if not already
            if not isinstance(sample_shap_values, list):
                # For one class per array format (3D array)
                if len(sample_shap_values.shape) == 3:
                    # Convert to list of arrays, one per class
                    sample_shap_values = [sample_shap_values[0, :, i] for i in range(sample_shap_values.shape[2])]
                else:
                    # For binary case, convert to list with two entries
                    sample_shap_values = [sample_shap_values[0], -sample_shap_values[0]]
            else:
                # If already a list but contains 2D arrays (multiple samples), take only first sample
                sample_shap_values = [sv[0] for sv in sample_shap_values]

            # Ensure base_values is a list
            if not isinstance(base_values, list):
                if len(label_dict) == 2:  # Binary case
                    base_values = [base_values, -base_values]
                else:  # Multi-class case
                    base_values = [base_values] * len(label_dict)

            # Create the decision plot
            plt.figure(figsize=(15, 10))

            # Create legend labels with probabilities if available
            if pred_probs is not None:
                legend_labels = [
                    f"{class_names[j]} ({pred_probs[i, j]:.2f})"
                    for j in range(len(class_names))
                ]
            else:
                legend_labels = class_names

            # True class for this sample
            true_class = y_true[idx]
            pred_class = y_pred[idx]

            # Debug print
            print(f"  Base values: {base_values}")
            print(f"  SHAP values shapes: {[sv.shape for sv in sample_shap_values]}")
            print(f"  Legend labels: {legend_labels}")

            shap.decision_plot(
                base_values,
                sample_shap_values,
                feature_names=feature_names,
                highlight=true_class,
                legend_labels=legend_labels,
                show=False
            )

            plt.title(
                f"Cancer Type Decision Plot - Sample {idx}\n"
                f"True: {reversed_label_dict[true_class]} | "
                f"Predicted: {reversed_label_dict[pred_class]} "
                f"({'Correct' if true_class == pred_class else 'Incorrect'})",
                pad=20
            )

            plt.tight_layout()

            # Save or show the plot
            if save_path:
                plt.savefig(f"{save_path}/decision_plot_sample_{idx}.png", dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

        except Exception as e:
            print(f"Error creating decision plot for sample {idx}: {e}")
            plt.close()

        # Try multioutput decision plot as an alternative if available
        try:
            plt.figure(figsize=(15, 10))

            # Get SHAP values in the format for multioutput_decision_plot
            # For multioutput_decision_plot, we often need to restructure the SHAP values
            multi_shap_values = explainer.shap_values(sample_to_explain)

            # Debug print
            print("  Attempting multioutput_decision_plot as alternative...")

            # Prepare base values correctly
            multi_base_values = explainer.expected_value
            if not isinstance(multi_base_values, list):
                multi_base_values = [multi_base_values] * len(label_dict)

            shap.multioutput_decision_plot(
                multi_base_values,
                multi_shap_values,
                row_index=0,  # Since we're only using one sample
                feature_names=feature_names,
                highlight=y_true[idx],
                legend_labels=legend_labels,
                legend_location="lower right",
                show=False
            )

            plt.title(
                f"Multioutput Decision Plot - Sample {idx}\n"
                f"True: {reversed_label_dict[true_class]} | "
                f"Predicted: {reversed_label_dict[pred_class]}",
                pad=20
            )

            plt.tight_layout()

            # Save or show the plot
            if save_path:
                plt.savefig(f"{save_path}/multi_decision_plot_sample_{idx}.png", dpi=300, bbox_inches='tight')
            else:
                plt.show()

        except Exception as e:
            print(f"  Multioutput decision plot also failed: {e}")

        plt.close('all')
        safe_gpu_clear()


def manual_decision_plot(explainer, X, y_true, label_dict, sample_idx, save_path=None):
    """
    A simplified version that uses direct SHAP approaches for a single sample.
    """
    import shap
    import numpy as np
    import matplotlib.pyplot as plt

    # Get reversed label dictionary
    reversed_label_dict = {v: k for k, v in label_dict.items()}
    class_names = [reversed_label_dict[i] for i in range(len(label_dict))]

    # Extract single sample
    sample = X.iloc[[sample_idx]]

    # Calculate SHAP values for this sample
    shap_values = explainer.shap_values(sample)

    # Get base/expected values
    base_values = explainer.expected_value

    # Convert to appropriate format if needed
    if not isinstance(shap_values, list):
        print("Converting SHAP values to list format...")
        if len(shap_values.shape) == 3:  # (samples, features, classes)
            shap_list = []
            for class_idx in range(shap_values.shape[2]):
                shap_list.append(shap_values[0, :, class_idx])
            shap_values = shap_list

    if not isinstance(base_values, list):
        base_values = [base_values] * len(label_dict)

    # Print the shapes for debugging
    print(f"Base values: {base_values}")
    if isinstance(shap_values, list):
        print(f"SHAP values shapes: {[sv.shape for sv in shap_values]}")
    else:
        print(f"SHAP values shape: {shap_values.shape}")

    # Create the plot
    plt.figure(figsize=(15, 10))

    try:
        # Try decision_plot first (works better with some models)
        shap.decision_plot(
            base_values,
            shap_values,
            feature_names=list(X.columns),
            legend_labels=class_names,
            show=False
        )
        plt.title(f"Decision Plot for Sample {sample_idx} - Cancer Type: {reversed_label_dict[y_true[sample_idx]]}")

        if save_path:
            plt.savefig(f"{save_path}/manual_decision_plot_{sample_idx}.png", dpi=300, bbox_inches='tight')
        else:
            plt.show()
    except Exception as e:
        print(f"Regular decision plot failed: {e}")
        plt.close()

        # Try waterfall plot as fallback
        try:
            plt.figure(figsize=(12, 8))
            # For waterfall, just use the predicted class
            pred_class = np.argmax([sv.sum() for sv in shap_values])
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[pred_class],
                    base_values=base_values[pred_class],
                    data=sample.values[0],
                    feature_names=list(X.columns)
                ),
                show=False
            )
            plt.title(f"SHAP Waterfall Plot - Sample {sample_idx} - Class: {class_names[pred_class]}")

            if save_path:
                plt.savefig(f"{save_path}/waterfall_plot_{sample_idx}.png", dpi=300, bbox_inches='tight')
            else:
                plt.show()
        except Exception as e2:
            print(f"Waterfall plot also failed: {e2}")

    plt.close('all')


# Specific function to fix the multioutput decision plot issue
def fix_multioutput_decision_plot(explainer, X, y_true, label_dict, sample_idx=0, save_path=None):
    """
    A specialized function to fix the multioutput decision plot issue
    with "base_values and shap_values args expect lists" error.
    """
    import shap
    import numpy as np
    import matplotlib.pyplot as plt

    # Get reversed label dictionary
    reversed_label_dict = {v: k for k, v in label_dict.items()}
    class_names = [reversed_label_dict[i] for i in range(len(label_dict))]

    # Extract single sample
    sample = X.iloc[[sample_idx]]
    true_class = y_true[sample_idx]

    print(f"Creating decision plot for sample {sample_idx}")
    print(f"True class: {true_class} ({reversed_label_dict[true_class]})")

    # Get base values
    base_values = explainer.expected_value
    print(f"Base values original type: {type(base_values)}")
    print(f"Base values original: {base_values}")

    # Convert base_values to a list of python floats if needed
    if isinstance(base_values, np.ndarray):
        base_values = base_values.tolist()
    elif not isinstance(base_values, list):
        base_values = [float(base_values)] * len(label_dict)

    print(f"Base values after conversion: {base_values}")

    # Calculate SHAP values specifically for this sample
    try:
        shap_values = explainer.shap_values(sample)
        print(f"SHAP values type: {type(shap_values)}")

        if isinstance(shap_values, list):
            print(f"SHAP values is a list of length {len(shap_values)}")
            print(f"First element shape: {shap_values[0].shape}")
        else:
            print(f"SHAP values shape: {shap_values.shape}")

            # Convert to the expected list format if needed
            if len(shap_values.shape) == 3:  # (samples, features, classes)
                print("Converting 3D array to list of arrays...")
                shap_list = []
                for class_idx in range(shap_values.shape[2]):
                    shap_list.append(shap_values[0, :, class_idx])
                shap_values = shap_list
                print(f"After conversion: SHAP values is a list of length {len(shap_values)}")
            elif len(shap_values.shape) == 2:  # (samples, features)
                print("Converting 2D array to list for binary case...")
                shap_values = [shap_values[0], -shap_values[0]]  # For binary classification

        # Create the plot
        plt.figure(figsize=(15, 10))

        # For multioutput_decision_plot, ensure shap_values is in the right format
        # The function expects a list of arrays, one per class
        try:
            # Try the regular decision plot first (more compatible)
            shap.decision_plot(
                base_values,
                shap_values,
                feature_names=list(X.columns),
                legend_labels=class_names,
                highlight=true_class,
                show=False
            )
            plt.title(f"Decision Plot for {reversed_label_dict[true_class]} (Sample {sample_idx})")

            if save_path:
                plt.savefig(f"{save_path}/fixed_decision_plot_{sample_idx}.png", dpi=300, bbox_inches='tight')
            else:
                plt.show()

            print("Successfully created decision plot!")

        except Exception as e:
            print(f"Regular decision plot failed: {e}")
            plt.close()

            # Try as a last resort
            try:
                plt.figure(figsize=(15, 10))

                # If SHAP values is a 3D array with dimensions (samples, features, classes)
                # we need to restructure
                if not isinstance(shap_values, list) and len(shap_values.shape) == 3:
                    class_list = []
                    for c in range(shap_values.shape[2]):
                        class_list.append(shap_values[0, :, c])
                    shap_values = class_list

                if isinstance(shap_values, list):
                    # For XGBoost, the first element might be for negative class (binary case)
                    # Just use the positive class for visualization
                    if len(shap_values) == 2 and len(label_dict) > 2:
                        # This is a binary output from a multiclass problem
                        print("Binary SHAP output detected for multiclass problem")
                        # Try to use force plot instead as fallback
                        shap.force_plot(base_values[1], shap_values[1], sample.iloc[0],
                                        matplotlib=True, show=False,
                                        feature_names=list(X.columns))
                    else:
                        # Normal multiclass case
                        print("Trying with individual SHAP plots for each class")
                        for i, sv in enumerate(shap_values):
                            if i >= len(class_names):
                                break
                            plt.subplot(len(shap_values), 1, i + 1)
                            shap.force_plot(base_values[i], sv, sample.iloc[0],
                                            matplotlib=True, show=False,
                                            feature_names=list(X.columns))
                            plt.title(f"Class: {class_names[i]}")

                    plt.tight_layout()
                    if save_path:
                        plt.savefig(f"{save_path}/fallback_force_plot_{sample_idx}.png", dpi=300, bbox_inches='tight')
                    else:
                        plt.show()

                else:
                    print("Could not create appropriate visualization")

            except Exception as e2:
                print(f"All visualization attempts failed: {e2}")

    except Exception as e:
        print(f"Failed to calculate SHAP values: {e}")

    plt.close('all')


# def plot_decision_plots(shap_values, X, label_dict, mapping, explainer, max_display=20, n_samples=10):
#     reversed_label_dict = {v: k for k, v in label_dict.items()}
#
#     num_rows = shap_values.shape[0]
#     sample_indices = np.random.choice(num_rows, size=min(n_samples, num_rows), replace=False)
#     shap_subset = shap_values[sample_indices, :, :]
#     X_subset = X.iloc[sample_indices]
#
#     feature_names = list(X.columns)
#
#     if shap_subset.shape[1] > max_display:
#         mean_abs = np.abs(shap_subset).mean(axis=(0, 2))
#         top_indices = np.argsort(-mean_abs)[:max_display]
#         shap_subset = shap_subset[:, top_indices, :]
#         X_subset = X_subset.iloc[:, top_indices]
#         feature_names = [feature_names[i] for i in top_indices]
#
#     class_names = [reversed_label_dict[i] for i in range(len(label_dict))]
#
#     # Get base_values from explainer
#     base_values = explainer.expected_value
#     if isinstance(base_values, np.ndarray):
#         base_values = base_values.tolist()
#     elif not isinstance(base_values, list):
#         base_values = [base_values]
#
#     # Ensure base_values has the same length as the number of classes
#     if len(base_values) != len(class_names):
#         base_values = [base_values[0]] * len(class_names)
#
#     # Reshape SHAP values for multioutput decision plot
#     # Shape should be (n_samples, n_features, n_classes)
#     shap_values_reshaped = np.transpose(shap_subset, (0, 1, 2))
#
#     plt.figure(figsize=(24, 14))
#     shap.multioutput_decision_plot(
#         base_values,
#         shap_values_reshaped,
#         X_subset,
#         feature_names=feature_names,
#         class_names=class_names,
#         show=False,
#         ignore_warnings=True
#     )
#     plt.title("Multioutput Decision Plot", pad=20)
#     plt.tight_layout()
#     plt.savefig("figures/multioutput_decision_plot.png", dpi=300, bbox_inches='tight')
#     plt.close()
#     plt.close('all')
#     safe_gpu_clear()


# === Hypotheses Generation ===
def generate_hypotheses_db(explainer, model, X, y_true, label_dict, mapping,
                           min_features=3, relative_threshold_percent=10, min_support=3):
    hypotheses_db = _generate_raw_hypotheses_df(X, explainer, label_dict, min_features,
                                                min_support, model, relative_threshold_percent, y_true)
    hypotheses_db = apply_category_mappings(hypotheses_db, mapping)
    hypotheses_db = hypotheses_db.sort_values(["cancer_type", 'support'], ascending=[True, False])
    return generate_sentences(hypotheses_db)


def _generate_raw_hypotheses_df(X, explainer, label_dict, min_features, min_support,
                                model, relative_threshold_percent, y_true):
    reversed_label_dict = {v: k for k, v in label_dict.items()}
    y_pred = model.predict(X)
    correct_indices = y_pred == y_true
    correct_X = X[correct_indices]
    correct_y = y_true[correct_indices]
    shap_values = calculate_shap_in_batches(explainer, correct_X)
    hypotheses, top_feat = [], set()

    for i, sample_idx in tqdm(enumerate(correct_X.index), desc="Generating hypotheses", unit="sample", colour='green'):
        cancer_type = reversed_label_dict[correct_y[i]]
        sample_shap = shap_values[i, :, :].T[correct_y[i]]
        total_contribution = np.sum(sample_shap)
        contribution_cutoff = (relative_threshold_percent / 100) * total_contribution
        significant_indices = np.where(sample_shap >= contribution_cutoff)[0]
        top_k_features = max(len(significant_indices), min_features)
        top_features_indices = sample_shap.argsort()[-(top_k_features + 1):][::-1]
        top_features = [(X.columns[idx], correct_X.iloc[i, idx]) for idx in top_features_indices]
        if ("Smoke Status", 2) in top_features:
            top_features.remove(("Smoke Status", 2))
        else:
            top_features = top_features[:-1]
        for f in top_features:
            top_feat.add(f[0])
        hypothesis = {f"{feature}": value for feature, value in top_features}
        hypothesis["cancer_type"] = cancer_type
        hypotheses.append(frozenset(hypothesis.items()))

    hypothesis_counts = Counter(hypotheses)
    filtered = [dict(hypo) for hypo, count in hypothesis_counts.items() if count >= min_support]
    df = pd.DataFrame(filtered).fillna("")
    df["support"] = [hypothesis_counts[frozenset(hypo.items())] for hypo in filtered]
    return df


def generate_sentences(df):
    sentences = []
    for _, row in df.iterrows():
        parts = []
        for col in df.columns:
            if col in ["cancer_type", "support"]:
                continue
            value = row[col]
            if pd.isna(value) or value == "":
                continue
            elif value in [0, 1]:
                parts.append(f"{'is' if value == 1 else 'is NOT'} {col}")
            else:
                parts.append(f"{col} value is {value}")
        sentence = " AND ".join(parts).replace("_", " ")
        sentences.append(sentence)
    df["hypothesis"] = sentences
    return df


# === Main SHAP Analysis Pipeline ===
def shap_analysis(model, X, y_true, label_dict, mapping, batch_size=256):
    print("Building SHAP explainer...")
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

    print("Filtering for model features...")
    X = X.loc[:, model.get_booster().feature_names]

    print("Filtering for correct predictions...")
    y_pred = model.predict(X)
    correct_indices = y_pred == y_true
    correct_X = X[correct_indices]
    correct_y = y_true[correct_indices]

    print("Loading SHAP values for correct predictions...")
    # Either load from file or calculate
    try:
        shap_values = np.load("models_and_explainers/shap_values.npy")
        # Filter SHAP values for correct predictions
        shap_values = shap_values[correct_indices]
    except:
        print("Calculating SHAP values...")
        shap_values = calculate_shap_in_batches(explainer, correct_X)
        # Optionally save for future use
        os.makedirs("models_and_explainers", exist_ok=True)
        np.save("models_and_explainers/shap_values.npy", shap_values)

    safe_gpu_clear()

    print("Creating comprehensive SHAP visualizations...")

    # Create decision plots with improved function
    print("Creating decision plots...")
    plot_decision_plots(
        explainer=explainer,
        X=X,  # Use all data, not just correct predictions
        y_true=y_true,
        y_pred=y_pred,
        label_dict=label_dict,
        model=model,  # Pass model to get prediction probabilities
        sample_indices=None,  # Will randomly select correct predictions
        max_samples=5,
        save_path="figures/decision_plots"
    )

    # For debugging with a single sample
    fix_multioutput_decision_plot(
        explainer=explainer,
        X=X,
        y_true=y_true,
        label_dict=label_dict,
        sample_idx=456,  # One of the samples that was failing
        save_path="figures/debug_plots"
    )

    # Or try the manual approach
    manual_decision_plot(
        explainer=explainer,
        X=X,
        y_true=y_true,
        label_dict=label_dict,
        sample_idx=456,
        save_path="figures/manual_plots"
    )

    # Continue with the rest of your visualization functions...
    print("Creating summary plots...")
    efficient_shap_summary_plots(shap_values, correct_X, label_dict)

    print("Creating feature importance heatmap...")
    plot_feature_importance_heatmap(shap_values, correct_X, label_dict, mapping)

    print("Creating feature distribution plots...")
    plot_top_feature_distributions(shap_values, correct_X, label_dict, mapping)

    print("Creating dependence plots...")
    plot_shap_dependence_plots(shap_values, correct_X, label_dict, mapping)

    print("Generating top feature SHAP table...")
    df_raw = generate_raw_df(np.array(shap_values), correct_X, correct_y, label_dict, top_n=10)
    df_raw.to_csv("figures/shap_raw_top_features.csv", index=False)

    print("Computing SHAP interaction values and plots...")
    try:
        interactions = np.load("models_and_explainers/shap_interactions.npy")
        # Filter interactions for correct predictions
        interactions = interactions[correct_indices]
    except:
        print("Interaction values not found. Calculate them with calculate_shap_interactions_in_batches()")

    try:
        for class_idx in range(len(label_dict)):
            generate_shap_interaction_heatmap(interactions, correct_X, class_idx, label_dict)
            shap_dependence_top_interactions(interactions, correct_X, class_idx, label_dict, top_k=3)
    except:
        print("Could not generate interaction plots.")

    return df_raw