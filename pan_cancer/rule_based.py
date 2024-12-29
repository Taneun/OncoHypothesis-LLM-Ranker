import matplotlib
from imodels.rule_set.skope_rules import SkopeRulesClassifier
import pandas as pd
from model_metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier


def rule_based(X_train, X_test, y_train, y_test, label_dictionary):
    rules_dict = {}
    y_pred_list = []
    y_proba_list = []
    # Convert y to pandas Series
    y_train = pd.Series(y_train, index=X_train.index)
    y_test = pd.Series(y_test, index=X_test.index)

    # Combine X and y for both training and test sets
    train_combined = pd.concat([X_train, y_train], axis=1)
    test_combined = pd.concat([X_test, y_test], axis=1)

    # Drop columns and rows with NAs
    train_combined = train_combined.drop(['Site1_Hugo_Symbol', 'Site2_Hugo_Symbol', 'Event_Info'], axis=1).dropna()
    test_combined = test_combined.drop(['Site1_Hugo_Symbol', 'Site2_Hugo_Symbol', 'Event_Info'], axis=1).dropna()

    # Split back into X and y
    X_train_cleaned = train_combined.iloc[:, :-1]  # All columns except the last one
    y_train_cleaned = train_combined.iloc[:, -1]  # The last column

    X_test_cleaned = test_combined.iloc[:, :-1]
    y_test_cleaned = test_combined.iloc[:, -1]

    # Assertions to verify shapes
    assert X_train_cleaned.shape[0] == y_train_cleaned.shape[0], "Mismatch in training set rows between X and y."
    assert X_test_cleaned.shape[0] == y_test_cleaned.shape[0], "Mismatch in test set rows between X and y."
    assert X_train_cleaned.shape[1] == X_test_cleaned.shape[1], "Mismatch in feature count between train and test."

    for cancer_type, idx in label_dictionary.items():
        rule_model = SkopeRulesClassifier(random_state=39,
                                          n_estimators=50,
                                          max_depth=[3, 5, 7],
                                          max_depth_duplication=3,
                                          max_features='sqrt',
                                          min_samples_split=10)

        # Verify the one-vs-all transformation
        one_vs_all_y_train = (y_train_cleaned == idx).astype(int)
        one_vs_all_y_test = (y_test_cleaned == idx).astype(int)
        assert one_vs_all_y_train.shape[0] == X_train_cleaned.shape[0], "Mismatch in rows for one-vs-all y_train."
        assert one_vs_all_y_test.shape[0] == X_test_cleaned.shape[0], "Mismatch in rows for one-vs-all y_test."

        # Fit and evaluate the model
        fitted, y_pred, y_proba = fit_and_evaluate(f"{cancer_type} vs. All", rule_model, X_train_cleaned,
                                                   X_test_cleaned,
                                                   one_vs_all_y_train, one_vs_all_y_test, label_dictionary,
                                                   is_multiclass=False, print_eval=False)
        if fitted.rules_:
            rules_dict[cancer_type] = fitted.rules_

        y_pred_list.append(y_pred)
        y_proba_list.append(y_proba)

    # plot_auc_curves(y_pred_list, y_proba_list, y_test_cleaned, label_dictionary)

    # for cancer_type, rules in rules_dict.items():
    #     print(f"\n***** {cancer_type} *****")
    #     for rule in rules:
    #         print(rule)

    return rules_dict


def plot_auc_curves(y_pred_list, y_proba_list, y_test, label_dict):
    """
    Plot ROC AUC and Precision-Recall AUC for multiple cancer types.

    Parameters:
        y_pred_list (list of np.ndarray): List of binary predictions for each cancer type.
        y_proba_list (list of np.ndarray): List of probabilities for each cancer type.
        y_test (pd.Series): True labels for the test set.
        label_dict (dict): Mapping of cancer types to integer labels.
    """
    reversed_label_dict = {v: k for k, v in label_dict.items()}
    cancer_types = list(label_dict.keys())
    matplotlib.use('TkAgg')
    # Initialize figures
    plt.figure(figsize=(10, 6))
    plt.title("ROC AUC for Each Cancer Type")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    for i, cancer_type in enumerate(cancer_types):
        fpr, tpr, _ = roc_curve((y_test == label_dict[cancer_type]).astype(int), y_proba_list[i][:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cancer_type} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label="Chance")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Precision-Recall AUC plot
    plt.figure(figsize=(10, 6))
    plt.title("Precision-Recall AUC for Each Cancer Type")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    for i, cancer_type in enumerate(cancer_types):
        precision, recall, _ = precision_recall_curve(
            (y_test == label_dict[cancer_type]).astype(int), y_proba_list[i],)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{cancer_type} (AUC = {pr_auc:.2f})")

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def optimize_skope_rules(X_train, X_test, y_train, y_test):
    y_train = pd.Series(y_train, index=X_train.index)
    y_test = pd.Series(y_test, index=X_test.index)

    # Combine X and y for both training and test sets
    train_combined = pd.concat([X_train, y_train], axis=1)
    test_combined = pd.concat([X_test, y_test], axis=1)

    # Drop columns and rows with NAs
    train_combined = train_combined.drop(['Site1_Hugo_Symbol', 'Site2_Hugo_Symbol', 'Event_Info'], axis=1).dropna()
    test_combined = test_combined.drop(['Site1_Hugo_Symbol', 'Site2_Hugo_Symbol', 'Event_Info'], axis=1).dropna()

    # Split back into X and y
    X_train_cleaned = train_combined.iloc[:, :-1]  # All columns except the last one
    y_train_cleaned = train_combined.iloc[:, -1]  # The last column

    X_test_cleaned = test_combined.iloc[:, :-1]
    y_test_cleaned = test_combined.iloc[:, -1]
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
    grid_search.fit(X_train_cleaned, y_train_cleaned)

    # Get the best parameters
    print("Best parameters:", grid_search.best_params_)

    # Evaluate the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_cleaned)

    # You can also print classification metrics or confusion matrix
    from sklearn.metrics import classification_report
    print(classification_report(y_test_cleaned, y_pred))
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

    def get_chromosome_range(comparison, value):
        chrom_map = mapping['Chromosome']
        value_idx = int(float(value))

        if comparison == '<=':
            # Include all chromosomes with index less than or equal to value_idx
            valid_chroms = [chrom for idx, chrom in chrom_map.items() if idx <= value_idx]
        elif comparison == '>':
            # Include all chromosomes with index greater than value_idx
            valid_chroms = [chrom for idx, chrom in chrom_map.items() if idx > value_idx]

        # Sort chromosomes naturally (1,2,3,...,22,X)
        sorted_chroms = sorted(valid_chroms,
                               key=lambda x: float('inf') if x == 'X' else int(x))

        return f"AND Chromosome is one of ({', '.join(sorted_chroms)})"

    def get_age_description(comparison, value):
        age_map = mapping['Diagnosis Age']
        age_value = int(float(value))
        age_range = age_map[age_value]
        if comparison == '>':
            return f"AND Diagnosis Age is older than {age_range}"
        else:
            return f"AND Diagnosis Age is younger than {age_range}"

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
                    return None  # Skip rules with unknown smoking status

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
                          'upstream_gene_variant']

            if feature_name in dummy_vars:
                if '<= 0.5' in part:
                    converted_parts.append(f"NOT {feature_name}")
                elif '> 0.5' in part:
                    converted_parts.append(feature_name)
                continue

            # Special handling for different features
            if feature_name == 'Chromosome':
                converted_parts.append(get_chromosome_range(comparison, value))

            elif feature_name == 'Sex':
                sex_value = 'male' if value > 0.5 else 'female'
                converted_parts.append(f"AND is {sex_value}")

            elif feature_name == 'Diagnosis Age':
                converted_parts.append(get_age_description(comparison, value))

            elif feature_name == 'Smoke Status':
                smoke_value = 'Nonsmoker' if value <= 0.5 else 'Smoker'
                converted_parts.append(f"AND smoking status is {smoke_value}")

            elif feature_name in ['TMB (nonsynonymous)', 'Start_Position', 'End_Position', 'Protein_position']:
                converted_parts.append(part)

            elif feature_name in mapping:
                cat_value = int(value)
                if comparison == '<=':
                    converted_parts.append(f"AND {feature_name} is {mapping[feature_name][cat_value]}")
                elif comparison == '>':
                    converted_parts.append(f"AND {feature_name} is not {mapping[feature_name][cat_value]}")

            else:
                converted_parts.append(part)

        # Remove the first "AND" from the first part
        if converted_parts:
            if converted_parts[0].startswith("AND "):
                converted_parts[0] = converted_parts[0][4:]

        return ' '.join(converted_parts)

    converted_rules = {}
    for cancer_type, rules in rules_dict.items():
        converted_rules[cancer_type] = [rule for rule in
                                        [convert_single_rule(str(rule)) for rule in rules]
                                        if rule is not None]

    for cancer_type, rules in converted_rules.items():
        print(f"\n***** {cancer_type} *****")
        for rule in rules:
            print(rule)

    return converted_rules