"""
The main script for the pan_cancer project.
"""
from SHAP_explain import *
from XGBoost_Model import *
from model_metrics import *
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from rule_based import rule_based
from sklearn import tree
from sklearn.tree import export_text
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import graphviz


def cancer_type_correlations(df):
    """
    Generate a list of cancer types and non-null feature-value pairs for each row in the DataFrame.

    Parameters:
    - df: DataFrame with columns "cancer_type", feature columns, and "support".

    Returns:
    - A list of formatted strings for each row.
    """
    corr_list = []
    # Iterate through each row
    for _, row in df.iterrows():
        # Extract cancer type and support
        cancer_type = row["cancer_type"]
        support = row["support"]

        # Get feature-value pairs where the feature value is not null or empty
        features = [
            f"{feature}={row[feature]}"
            for feature in df.columns
            if feature not in {"cancer_type", "support"} and pd.notnull(row[feature]) and row[feature] != ""
        ]

        # Format the output
        if features:
            features_str = ", ".join(features)
            corr_list.append(f"{cancer_type}: {features_str}, Support: {support}")
        else:
            corr_list.append(f"{cancer_type}: Support: {support}")  # No features available

    return corr_list

import pandas as pd


def tree_to_sentences(tree, feature_names, class_names, node_id=0, path_dict=None, label_name=None, results_df=None):
    """
    Recursively traverse the decision tree and convert each rule path into a sentence.
    Keeps only the tightest condition for each feature and handles dummy variables.

    Parameters:
        tree (sklearn.tree._tree.Tree): The decision tree object.
        feature_names (list): Names of features used in the decision tree.
        class_names (list): Class labels corresponding to the target variable.
        node_id (int): The current node being processed (default is 0 for the root node).
        path_dict (dict): Accumulated decisions for features leading to the current node.
        label_name (str): The label (cancer type) for the current tree.
        results_df (pd.DataFrame): The DataFrame to store the results.

    Returns:
        pd.DataFrame: The DataFrame containing the rules (label and sentence).
    """
    if path_dict is None:
        path_dict = {}

    if results_df is None:
        results_df = pd.DataFrame(columns=["Cancer Type", "Sentence"])

    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    # If this is a leaf node
    if left_child == -1 and right_child == -1:
        class_value = tree.value[node_id].argmax()
        class_name = class_names[class_value]
        path_str = " AND ".join(
            [f"{feature} {condition}" for feature, condition in path_dict.items()]
        )
        sentence = f"If {path_str}, then the class is {class_name}."
        # print(sentence)

        # Create a temporary DataFrame for this result
        temp_df = pd.DataFrame({"Cancer Type": [class_name], "Sentence": [sentence]})

        # Concatenate the new result to the existing DataFrame
        results_df = pd.concat([results_df, temp_df], ignore_index=True)
        return results_df

    # Get feature and threshold for the current node
    feature = feature_names[tree.feature[node_id]]
    threshold = tree.threshold[node_id]

    # Handle dummy variables (binary features)
    if threshold == 0.5:
        left_condition = "is 0"
        right_condition = "is 1"
    else:
        # Handle numeric features
        left_condition = f"<= {threshold}"
        right_condition = f"> {threshold}"

    # Create copies of the path dictionary for each branch
    left_path_dict = path_dict.copy()
    right_path_dict = path_dict.copy()

    # Update conditions for numeric or dummy features
    if feature in left_path_dict:
        # Tighten the condition: keep the stricter condition for <= on the left
        if ">" in left_path_dict[feature]:
            del left_path_dict[feature]  # Remove any previous ">" condition
        left_path_dict[feature] = left_condition  # Update with the latest <= condition

    if feature in right_path_dict:
        # Tighten the condition: keep the stricter condition for > on the right
        if "<=" in right_path_dict[feature]:
            del right_path_dict[feature]  # Remove any previous "<=" condition
        right_path_dict[feature] = right_condition  # Update with the latest > condition

    # Add new conditions if the feature hasn't been encountered yet
    if feature not in left_path_dict:
        left_path_dict[feature] = left_condition
    if feature not in right_path_dict:
        right_path_dict[feature] = right_condition

    # Recurse on children nodes
    results_df = tree_to_sentences(tree, feature_names, class_names, left_child, left_path_dict, label_name, results_df)
    results_df = tree_to_sentences(tree, feature_names, class_names, right_child, right_path_dict, label_name,
                                   results_df)

    return results_df

if __name__ == "__main__":
    # Load the data
    filepath = "data_for_rules.csv"
    X, y, label_dict, mapping = load_data(filepath)
    X_train, X_test, y_train, y_test, X_test_with_id = stratified_split_by_patient(X, y)
    # xtra_cheese = xgb.XGBClassifier(n_estimators=250, objective='multi:softmax',
    #                                 tree_method='hist', enable_categorical=True)
    # # rand_forest = RandomForestClassifier(random_state=39)
    # decision_tree = DecisionTreeClassifier(random_state=39)
    # model_dict = {"XGBoost": xtra_cheese, "Random Forest": rand_forest, "Decision Tree": decision_tree}
    # decision_tree.fit(X_train, y_train)
    # for model_type, model in model_dict.items():
    #     # Fit and evaluate the model
    #     model, y_pred = fit_and_evaluate(model_type, model, X_train, X_test, y_train, y_test, label_dict,
    #                      show_auc=True, show_cm=True, show_precision_recall=True)
    #     classify_patients(X_test_with_id, y_pred, y_test, label_dict, model_type)

        # Save the model
        # pickle.dump(model, open(f"{model_type}_model.pkl", "wb"))

    # Export rules from the tree
    # tree_rules = export_text(decision_tree, feature_names=list(X.columns[1:]))
    # print(tree_rules)
    # plt.figure(figsize=(20, 10))
    # reversed_label_dict = {v: k for k, v in label_dict.items()}
    # class_names = [reversed_label_dict[cat] for cat in sorted(reversed_label_dict.keys())]
    # tree.plot_tree(decision_tree=decision_tree, feature_names=list(X.columns[1:]), class_names=class_names, filled=True, max_depth=3)
    # plt.savefig("decision_tree.png", dpi=300)
    reversed_label_dict = {v: k for k, v in label_dict.items()}
    class_names = [reversed_label_dict[cat] for cat in sorted(reversed_label_dict.keys())]

    results_df = pd.DataFrame(columns=["Cancer Type", "Sentence"])

    # Generate sentences from the global decision tree
    global_tree_model = DecisionTreeClassifier( random_state=39)
    global_tree_model.fit(X_train, y_train)

    tree = global_tree_model.tree_
    feature_names = list(X_train.columns)
    results_df = tree_to_sentences(tree, feature_names, class_names, node_id=0, path_dict=None, results_df=results_df)

    print(results_df.head())
    results_df.to_csv("decision_tree_sentences.csv", index=False)

    # for label_name, label_value in label_dict.items():
    #     print(f"Training tree for label: {label_name}")
    #
    #     # Create a binary target for the current label
    #     binary_target = (y_train == label_value).astype(int)
    #
    #     # Train the decision tree for the current label
    #     tree_model = DecisionTreeClassifier(random_state=42)
    #     tree_model.fit(X_train, binary_target)
    #
    #     # tree = tree_model.tree_
    #     feature_names = list(X_train.columns)
    #
    #     # Export rules for the decision tree
    #     tree_rules = export_text(tree_model, feature_names=list(X_train.columns))
    #     print(f"Rules for {label_name}:\n{tree_rules}\n")
    #
    #     # Plot the decision tree
    #     plt.figure(figsize=(20, 10))
    #     plot_tree(
    #         decision_tree=tree_model,
    #         feature_names=feature_names,
    #         class_names=["Not " + label_name, label_name],
    #         filled=True,
    #         rounded=True
    #     )
    #     plt.title(f"Decision Tree for {label_name}")
    #     plt.savefig(f"decision_tree_{label_name.replace(' ', '_')}.png", dpi=300)
    #     plt.close()


    # Export the decision tree to DOT format
    # dot_data = export_graphviz(decision_tree,
    #                            out_file=None,
    #                            feature_names=list(X.columns[1:]),
    #                            class_names=class_names,
    #                            filled=True,
    #                            rounded=True,  # Makes nodes rounded for better readability
    #                            special_characters=True)  # Handles special characters in feature names
    #
    # # Render the DOT data to a PNG file
    # graph = graphviz.Source(dot_data, format="png")
    # graph.render("decision_tree_graphviz", cleanup=True)

    # rule_based(X_train, X_test, y_train, y_test, label_dict)

    # explainer = shap.TreeExplainer(model)
    # get_shap_interactions(explainer, X, y, label_dict)

    # model = pickle.load(open("model.pkl", "rb"))
    # explainer = pickle.load(open("explainer.pkl", "rb"))
    # # Perform SHAP analysis
    # feature_importance = shap_analysis(explainer, X_val, y_val, predictions, label_dict)

    # Generate hypotheses database
    # initial_db = generate_hypotheses_db(explainer, model, X_test, y_test, label_dict)
    # # print("initial" + str(len(list(initial_db.columns))))
    # hypotheses = apply_category_mappings(initial_db, mapping)
    # print("hypotheses" + str(len(list(hypotheses.columns))))
    # #
    # hypotheses.to_csv("hypotheses.csv", index=False)
    # sorted_hypotheses = hypotheses.sort_values(["cancer_type", 'support'], ascending=[True, False])
    # corr_list = cancer_type_correlations(sorted_hypotheses)
    # print(corr_list)
    # # Print results
    # print(f"Validation Accuracy: {validation_accuracy}")
