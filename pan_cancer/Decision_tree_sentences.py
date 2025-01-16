from XGBoost_Model import *
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.tree import export_text
from sklearn.tree import plot_tree
from rule_based import *

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


def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            if threshold == 0.5:
                p1 += [f"({name} == 0)"]
                recurse(tree_.children_left[node], p1, paths)
                p2 += [f"({name} == 1)"]
                recurse(tree_.children_right[node], p2, paths)
            else:
                p1 += [f"({name} <= {np.round(threshold, 3)})"]
                recurse(tree_.children_left[node], p1, paths)
                p2 += [f"({name} > {np.round(threshold, 3)})"]
                recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: " + str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]

    return rules


if __name__ == "__main__":
    # Load the data
    data_for_dt = "data_for_rules.csv"
    X, y, label_dict, mapping = load_data(data_for_dt)
    X_train, X_test, y_train, y_test, X_test_with_id = stratified_split_by_patient(X, y)
    decision_tree = DecisionTreeClassifier(random_state=39)

    reversed_label_dict = {v: k for k, v in label_dict.items()}
    class_names = [reversed_label_dict[cat] for cat in sorted(reversed_label_dict.keys())]

    # results_df = pd.DataFrame(columns=["Cancer Type", "Sentence"])

    # Generate sentences from the global decision tree
    global_tree_model = DecisionTreeClassifier(max_depth=10, random_state=39)
    global_tree_model.fit(X_train, y_train)

    tree = global_tree_model.tree_
    feature_names = list(X_train.columns)
    # results_df = tree_to_sentences(tree, feature_names, class_names, node_id=0, path_dict=None, results_df=results_df)
    #
    # print(results_df.head())
    # results_df.to_csv("decision_tree_sentences.csv", index=False)

    rules = get_rules(global_tree_model, feature_names, class_names)
    for r in rules:
        print(r)

    for label_name, label_value in label_dict.items():
        print(f"Training tree for label: {label_name}")

        # Create a binary target for the current label
        binary_target = (y_train == label_value).astype(int)

        # Train the decision tree for the current label
        tree_model = DecisionTreeClassifier(random_state=42)
        tree_model.fit(X_train, binary_target)

        # tree = tree_model.tree_
        feature_names = list(X_train.columns)

        # Export rules for the decision tree
        tree_rules = export_text(tree_model, feature_names=list(X_train.columns))
        print(f"Rules for {label_name}:\n{tree_rules}\n")

        # Plot the decision tree
        plt.figure(figsize=(20, 10))
        plot_tree(
            decision_tree=tree_model,
            feature_names=feature_names,
            class_names=["Not " + label_name, label_name],
            filled=True,
            rounded=True
        )
        plt.title(f"Decision Tree for {label_name}")
        plt.savefig(f"figures/decision_tree_{label_name.replace(' ', '_')}.png", dpi=300)
        plt.close()