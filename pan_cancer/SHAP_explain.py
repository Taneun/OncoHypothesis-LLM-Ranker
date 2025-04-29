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
        plt.clf()
        shap.summary_plot(shap_values[:, :, class_idx], X, plot_type='bar', show=False, max_display=max_display)
        plt.title(f"SHAP Summary (Bar) - {reversed_label_dict[class_idx]}")
        plt.savefig(f"figures/shap_summary_bar_class_{reversed_label_dict[class_idx]}.png")
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
                shap.dependence_plot(X.columns[i], interactions[:, :, :, class_idx], X,
                                     interaction_index=X.columns[j], show=False)
                plt.title(f"Dependence Plot ({X.columns[i]} & {X.columns[j]}) - {reversed_label_dict[class_idx]}")
                plt.savefig(f"figures/shap_dependence_{X.columns[i]}_{X.columns[j]}_{reversed_label_dict[class_idx]}.png")
                plt.close()
                safe_gpu_clear()
                count += 1
            except Exception as e:
                print(f"Could not plot dependence for {X.columns[i]} & {X.columns[j]}: {e}")

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
def shap_analysis(model, X, y_true, label_dict, batch_size=256):
    print("Building SHAP explainer...")
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

    print("Calculating SHAP values...")
    X = X.loc[:, model.get_booster().feature_names]
    shap_values = explainer.shap_values(X)
    safe_gpu_clear()

    print("Plotting summary plots...")
    efficient_shap_summary_plots(shap_values, X, label_dict)

    print("Generating top feature SHAP table...")
    df_raw = generate_raw_df(np.array(shap_values), X, y_true, label_dict, top_n=10)
    df_raw.to_csv("figures/shap_raw_top_features.csv", index=False)

    print("Computing SHAP interaction values and plots...")
    interactions = calculate_shap_interactions_in_batches(explainer, X, y_true, batch_size=batch_size, save_path="models_and_explainers/shap_interactions.npy")

    for class_idx in range(len(label_dict)):
        generate_shap_interaction_heatmap(interactions, X, class_idx, label_dict)
        shap_dependence_top_interactions(interactions, X, class_idx, label_dict, top_k=3)

    return df_raw
