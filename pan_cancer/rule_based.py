from imodels.rule_set.skope_rules import SkopeRulesClassifier
import pandas as pd
from model_metrics import *

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
        rule_model = SkopeRulesClassifier(random_state=39)

        # Verify the one-vs-all transformation
        one_vs_all_y_train = (y_train_cleaned == idx).astype(int)
        one_vs_all_y_test = (y_test_cleaned == idx).astype(int)
        assert one_vs_all_y_train.shape[0] == X_train_cleaned.shape[0], "Mismatch in rows for one-vs-all y_train."
        assert one_vs_all_y_test.shape[0] == X_test_cleaned.shape[0], "Mismatch in rows for one-vs-all y_test."

        # Fit and evaluate the model
        fitted, y_pred, y_proba = fit_and_evaluate(f"{cancer_type} vs. All", rule_model, X_train_cleaned, X_test_cleaned,
                         one_vs_all_y_train, one_vs_all_y_test, label_dictionary, is_multiclass=False, print_eval=False)
        if fitted.rules_:
            rules_dict[cancer_type] = fitted.rules_

        y_pred_list.append(y_pred)
        y_proba_list.append(y_proba)

    plot_auc_curves(y_pred_list, y_proba_list, y_test_cleaned, label_dictionary)

    for cancer_type, rules in rules_dict.items():
        print(f"\n***** {cancer_type} *****")
        for rule in rules:
            print(rule)

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
            (y_test == label_dict[cancer_type]).astype(int), y_proba_list[i]
        )
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{cancer_type} (AUC = {pr_auc:.2f})")

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
