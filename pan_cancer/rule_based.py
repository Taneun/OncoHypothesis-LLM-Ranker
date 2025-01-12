import pickle
from imodels.rule_set.skope_rules import SkopeRulesClassifier
import pandas as pd
from model_metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier


def rule_based(X_train, X_test, y_train, y_test, label_dictionary, is_plot_run=False):
    rules_dict = {}
    rules_count = {cancer_type: 0 for cancer_type in label_dictionary.values()}
    y_pred_list = []
    y_proba_list = []

    for cancer_type, idx in label_dictionary.items():
        # Best parameters: {'estimator__max_depth': [3, 5, 7], 'estimator__max_depth_duplication': 3,
        # 'estimator__max_features': 'sqrt', 'estimator__min_samples_split': 10, 'estimator__n_estimators': 50}
        rule_model = SkopeRulesClassifier(random_state=39,
                                          n_estimators=50,
                                          max_depth=[3, 5, 7],
                                          max_depth_duplication=3,
                                          max_features='sqrt',
                                          min_samples_split=10,
                                          n_jobs=-1)


        # Verify the one-vs-all transformation
        one_vs_all_y_train = (y_train == idx).astype(int)
        one_vs_all_y_test = (y_test == idx).astype(int)
        assert one_vs_all_y_train.shape[0] == X_train.shape[0], "Mismatch in rows for one-vs-all y_train."
        assert one_vs_all_y_test.shape[0] == X_test.shape[0], "Mismatch in rows for one-vs-all y_test."

        # Fit and evaluate the model
        fitted, y_pred, y_proba = fit_and_evaluate(f"Skope Rules {idx}: {cancer_type} vs. All", rule_model, X_train,
                                                   X_test,
                                                   one_vs_all_y_train, one_vs_all_y_test, label_dictionary,
                                                   is_multiclass=False, print_eval=True)

        mask = y_proba[:, 1] > 0
        print(y_proba[mask].shape)
        print(y_proba[mask][:5])
        if fitted.rules_:
            rules_dict[cancer_type] = fitted.rules_
            rules_count[cancer_type] = len(fitted.rules_)

        y_pred_list.append((idx, y_pred))
        y_proba_list.append((idx, y_proba))

    if is_plot_run:
        plot_rules_roc_curves_by_cancer(y_proba_list, y_test, label_dictionary, rules_count)

    return rules_dict


def process_rule_results(y_pred_list, y_proba_list, y_test):
    n_samples = len(y_pred_list[0][1])
    y_pred = np.zeros(n_samples)
    y_proba = np.zeros((n_samples, len(y_pred_list)))
    has_prediction = np.zeros(n_samples, dtype=bool)

    for i, (idx, predictions) in enumerate(y_pred_list):
        mask = predictions == 1
        y_pred[mask] = idx
        y_proba[:, i] = y_proba_list[i][1][:, 1]
        has_prediction |= mask

    # Keep only samples with predictions
    y_pred = y_pred[has_prediction]
    y_proba = y_proba[has_prediction]
    y_test_filtered = y_test[has_prediction]

    # Normalize probabilities
    row_sums = y_proba.sum(axis=1)
    y_proba = y_proba / np.maximum(row_sums[:, np.newaxis], np.finfo(float).eps)

    return y_pred, y_proba, y_test_filtered


def plot_rules_roc_curves_by_cancer(y_proba_list, y_test, label_dictionary, rules_count):
    """
    Plot ROC curves for each cancer type using Plotly, using only samples where predictions were made.

    Args:
        y_proba_list: List of tuples (cancer_idx, probabilities)
        y_test: True labels
        label_dictionary: Dictionary mapping cancer types to indices
    """
    # Create figure
    fig = go.Figure()

    # Create reverse dictionary for index to label mapping
    idx_to_label = {v: k for k, v in label_dictionary.items()}

    # Plot ROC curve for each cancer type
    for idx, predictions in y_proba_list:
        # Create mask for samples where this model made predictions
        prediction_mask = predictions[:, 1] > 0

        # Get relevant samples
        filtered_proba = predictions[prediction_mask]
        filtered_true = (y_test[prediction_mask] == idx).astype(int)

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(filtered_true, filtered_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        # Count samples used for this curve
        n_samples = np.sum(prediction_mask)
        # Count how many of the samples predicted originated from the cancer type
        true_type_count = np.sum(y_test[prediction_mask] == idx)

        # Add ROC curve for this cancer type
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                name=f'{idx_to_label[idx]} (n={n_samples} ({true_type_count} from this type), Rules={rules_count[idx_to_label[idx]]} AUC={roc_auc:.2f})',
                mode='lines',
                hovertemplate=(
                    'False Positive Rate: %{x:.3f}<br>'
                    'True Positive Rate: %{y:.3f}<br>'
                    '<extra></extra>'
                )
            )
        )

    # Add diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Random Classifier',
            hovertemplate=(
                'Random Classifier<br>'
                'False Positive Rate: %{x:.3f}<br>'
                'True Positive Rate: %{y:.3f}<br>'
                '<extra></extra>'
            )
        )
    )

    # Update layout
    fig.update_layout(
        title='Skope Rules Classifier - ROC Curves by Cancer Type<br>(Only samples with predictions)',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        hovermode='closest',
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        ),
        template='plotly_white'
    )
    fig.show()
    return fig

def optimize_skope_rules(X_train, X_test, y_train, y_test):
    # Initialize the base classifier (SkopeRulesClassifier)
    base_classifier = SkopeRulesClassifier(random_state=39)

    # Wrap it in a OneVsRestClassifier
    ovr_classifier = OneVsRestClassifier(base_classifier)

    # Define a parameter grid for SkopeRulesClassifier and OneVsRestClassifier
    param_grid = {
        'estimator__max_depth_duplication': [1, 2, 3],
        'estimator__n_estimators': [10, 20, 50],
        'estimator__max_depth': [[3, 5, 7], [5, 7], [3, 5]],
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__max_features': [None, 'sqrt', 'log2'],
    }

    # Perform grid search
    grid_search = GridSearchCV(ovr_classifier, param_grid, cv=3, n_jobs=-1, verbose=2)

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    print("Best parameters:", grid_search.best_params_)

    # Evaluate the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # You can also print classification metrics or confusion matrix
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))
    # Best parameters: {'estimator__max_depth': [3, 5, 7], 'estimator__max_depth_duplication': 3,
    # 'estimator__max_features': 'sqrt', 'estimator__min_samples_split': 10, 'estimator__n_estimators': 50}


def create_rules_dataframe(rules_dict):
    """
    Create a DataFrame from a dictionary of rules.

    Parameters:
        rules_dict (dict): Dictionary of rules for each cancer type.

    Returns:
        pd.DataFrame: DataFrame with columns "Cancer Type" and "Rule".
    """
    rules_list = []
    for cancer_type, rules in rules_dict.items():
        for rule in rules:
            rules_list.append({"Cancer Type": cancer_type, "Rule": rule})

    return pd.DataFrame(rules_list)


def convert_rules_to_readable(rules_dict, mapping):
    """
    Convert encoded rules back to readable format using the mapping dictionary.

    Args:
        rules_dict: Dictionary where keys are cancer types and values are lists of rule strings
        mapping: Dictionary containing category mappings for each feature

    Returns:
        Dictionary with converted rules in natural language
    """
    def get_age_description(comparison, value):
        if comparison == '>':
            return f"AND Diagnosis Age is older than {value}"
        else:
            return f"AND Diagnosis Age is younger than {value}"

    def convert_single_rule(rule):
        parts = rule.split(' and ')
        converted_parts = []

        # First check if rule contains Unknown smoking status
        for part in parts:
            if 'Smoke Status' in part:
                components = part.strip().split(' ')
                comparison = components[-2]
                value = float(components[-1])
                if (comparison == '<=' and value >= 1.5) or (comparison == '>' and value >= 1.5):
                # if unknown smoking status, remove only this part
                    parts.remove(part)


        for part in parts:
            components = part.strip().split(' ')
            feature_name = ' '.join(components[:-2])
            comparison = components[-2]
            value = float(components[-1])

            # Handle dummy variables
            dummy_vars = ['3_prime_UTR_variant', '5_prime_UTR_variant', 'NMD_transcript_variant',
                          'coding_sequence_variant', 'downstream_gene_variant', 'frameshift_variant',
                          'inframe_deletion', 'inframe_insertion', 'intergenic_variant', 'intron_variant',
                          'mature_miRNA_variant', 'missense_variant', 'non_coding_transcript_exon_variant',
                          'non_coding_transcript_variant', 'protein_altering_variant', 'splice_acceptor_variant',
                          'splice_donor_variant', 'splice_region_variant', 'start_lost', 'start_retained_variant',
                          'stop_gained', 'stop_lost', 'stop_retained_variant', 'synonymous_variant',
                          'upstream_gene_variant', 'chr_1', 'chr_10', 'chr_11', 'chr_12', 'chr_13', 'chr_14', 'chr_15',
                           'chr_16', 'chr_17', 'chr_18', 'chr_19', 'chr_2', 'chr_20', 'chr_21',
                           'chr_22', 'chr_3', 'chr_4', 'chr_5', 'chr_6', 'chr_7', 'chr_8', 'chr_9',
                           'chr_X']

            if feature_name in dummy_vars:
                if feature_name.startswith('chr_'):
                    if '<= 0.5' in part:
                        converted_parts.append(f"AND Chromosome is NOT {feature_name[4:]}")
                    elif '> 0.5' in part:
                        converted_parts.append(f"AND Chromosome is {feature_name[4:]}")
                else:
                    if '<= 0.5' in part:
                        converted_parts.append(f"AND NOT {feature_name}")
                    elif '> 0.5' in part:
                        converted_parts.append(f"AND {feature_name}")
                continue

            elif feature_name == 'Diagnosis Age':
                converted_parts.append(get_age_description(comparison, value))

            elif feature_name == 'Sex':
                sex_value = 'male' if value > 0.5 else 'female'
                converted_parts.append(f"AND Sex is {sex_value}")

            elif feature_name == 'Smoke Status':
                smoke_value = 'Nonsmoker' if value <= 0.5 else 'Smoker'
                converted_parts.append(f"AND smoking status is {smoke_value}")

            elif feature_name == "VAR_TYPE_SX":
                var_class = 'Substitution/Indel' if value == 0 else 'Truncation'
                converted_parts.append(f"AND variant type is {var_class}")

            elif feature_name in mapping:
                cat_value = int(value)
                if comparison == '<=':
                    converted_parts.append(f"AND {feature_name} is {mapping[feature_name][cat_value]}")
                elif comparison == '>':
                    converted_parts.append(f"AND {feature_name} is not {mapping[feature_name][cat_value]}")

            else:
                converted_parts.append("AND " + part)

        return ' '.join(converted_parts)

    converted_rules = {}
    for cancer_type, rules in rules_dict.items():
        converted_rules[cancer_type] = [rule[4:].replace("_", " ") for rule in
                                        [convert_single_rule(str(rule)) for rule in rules]
                                        if rule is not None]

    for cancer_type, rules in converted_rules.items():
        print(f"\n***** {cancer_type} *****")
        for rule in rules:
            print(rule)

    return converted_rules


# plot_multiclass(label_dictionary.values(),
#                 label_dictionary,
#                 f"Skope Rules (n = {len(y_pred)})",
#                 show_auc=True,
#                 show_cm=True,
#                 show_precision_recall=True,
#                 y_pred=y_pred,
#                 y_proba=y_proba,
#                 y_test=y_test,
#                 y_test_bin=y_test_bin)