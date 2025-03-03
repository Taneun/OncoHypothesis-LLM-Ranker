from XGBoost_Model import *
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.tree import _tree

def get_readable_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = np.round(tree_.threshold[node], 2)
            p1, p2 = list(path), list(path)

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

            if name in dummy_vars:
                if name.startswith('chr_'):
                    p1.append((name, f"Chromosome is NOT {name[4:]}"))
                    recurse(tree_.children_left[node], p1, paths)
                    p2.append((name, f"AND Chromosome is {name[4:]}"))
                    recurse(tree_.children_right[node], p2, paths)
                else:
                    p1.append((name, f"NOT {name}"))
                    recurse(tree_.children_left[node], p1, paths)
                    p2.append((name, f"{name}"))
                    recurse(tree_.children_right[node], p2, paths)
            elif name == "Sex":
                p1.append((name, "Sex is Female"))
                recurse(tree_.children_left[node], p1, paths)
                p2.append((name, "Sex is Male"))
                recurse(tree_.children_right[node], p2, paths)
            elif name == "VAR_TYPE_SX":
                p1.append((name, f"Variant type is Substitution/Indel"))
                recurse(tree_.children_left[node], p1, paths)
                p2.append((name, f"Variant type is Truncation"))
                recurse(tree_.children_right[node], p2, paths)
            elif threshold == 0.5:
                p1.append((name, f"{name} is 0 (False)"))
                recurse(tree_.children_left[node], p1, paths)
                p2.append((name, f"{name} is 1 (True)"))
                recurse(tree_.children_right[node], p2, paths)
            else:
                p1.append((name, f"{name} ≤ {threshold}"))
                recurse(tree_.children_left[node], p1, paths)
                p2.append((name, f"{name} > {threshold}"))
                recurse(tree_.children_right[node], p2, paths)
        else:
            path.append((None, (tree_.value[node], tree_.n_node_samples[node])))
            paths.append(path)

    recurse(0, path, paths)

    # Sort by sample count
    paths.sort(key=lambda x: x[-1][1][1], reverse=True)

    readable_rules = []
    for path in paths:
        feature_conditions = {}

        # Process conditions to keep only meaningful ones
        for feature, condition in path[:-1]:
            if feature not in feature_conditions:
                feature_conditions[feature] = []
            feature_conditions[feature].append(condition)

        # Merge conditions for the same feature
        merged_conditions = []
        for feature, conditions in feature_conditions.items():
            if len(conditions) == 1:
                merged_conditions.append(conditions[0])
            else:
                lower, upper = None, None
                for cond in conditions:
                    if "≤" in cond:
                        val = float(cond.split("≤")[1].strip())
                        upper = val if upper is None else min(upper, val)
                    elif ">" in cond:
                        val = float(cond.split(">")[1].strip())
                        lower = val if lower is None else max(lower, val)

                if lower is not None and upper is not None:
                    merged_conditions.append(f"{lower} < {feature} < {upper}")
                elif lower is not None:
                    merged_conditions.append(f"{feature} > {lower}")
                elif upper is not None:
                    merged_conditions.append(f"{feature} ≤ {upper}")

        # Extract class prediction and probability
        class_prediction = path[-1][1][0][0]
        total_samples = np.sum(class_prediction)
        probabilities = (100.0 * class_prediction / total_samples).round(2)
        predicted_class_index = np.argmax(class_prediction)
        predicted_class = class_names[predicted_class_index]
        predicted_proba = probabilities[predicted_class_index]

        # Build final rule
        rule_sentence = (
                "If " + ", ".join(merged_conditions) +
                f", then the predicted class is **{predicted_class}** (Probability: {predicted_proba}%). " +
                f"(Based on {path[-1][1][1]} samples)"
        )
        readable_rules.append(rule_sentence)

    return readable_rules


if __name__ == "__main__":
    # Load the data
    data_for_dt = "data_for_rules.csv"
    X, y, label_dict, mapping = load_data(data_for_dt)
    X_train, X_test, y_train, y_test, X_test_with_id = stratified_split_by_patient(X, y)

    reversed_label_dict = {v: k for k, v in label_dict.items()}
    class_names = [reversed_label_dict[cat] for cat in sorted(reversed_label_dict.keys())]

    decision_tree = DecisionTreeClassifier(random_state=39, min_samples_leaf=10, max_depth=10)
    decision_tree.fit(X_train, y_train)

    feature_names = list(X_train.columns)

    rules = get_readable_rules(decision_tree, feature_names, class_names)

    for rule in rules:
        print(rule)

# def get_rules(tree, feature_names, class_names):
#     tree_ = tree.tree_
#     feature_name = [
#         feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
#         for i in tree_.feature
#     ]
#
#     paths = []
#     path = []
#
#     def recurse(node, path, paths):
#
#         if tree_.feature[node] != _tree.TREE_UNDEFINED:
#             name = feature_name[node]
#             threshold = tree_.threshold[node]
#             p1, p2 = list(path), list(path)
#             if threshold == 0.5:
#                 p1 += [f"({name} == 0)"]
#                 recurse(tree_.children_left[node], p1, paths)
#                 p2 += [f"({name} == 1)"]
#                 recurse(tree_.children_right[node], p2, paths)
#             else:
#                 p1 += [f"({name} <= {np.round(threshold, 3)})"]
#                 recurse(tree_.children_left[node], p1, paths)
#                 p2 += [f"({name} > {np.round(threshold, 3)})"]
#                 recurse(tree_.children_right[node], p2, paths)
#         else:
#             path += [(tree_.value[node], tree_.n_node_samples[node])]
#             paths += [path]
#
#     recurse(0, path, paths)
#
#     # sort by samples count
#     samples_count = [p[-1][1] for p in paths]
#     ii = list(np.argsort(samples_count))
#     paths = [paths[i] for i in reversed(ii)]
#
#     rules = []
#     for path in paths:
#         rule = "if "
#
#         for p in path[:-1]:
#             if rule != "if ":
#                 rule += " and "
#             rule += str(p)
#         rule += " then "
#         if class_names is None:
#             rule += "response: " + str(np.round(path[-1][0][0][0], 3))
#         else:
#             classes = path[-1][0][0]
#             l = np.argmax(classes)
#             rule += f"class: {class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
#         rule += f" | based on {path[-1][1]:,} samples"
#         rules += [rule]
#
#     return rules

# if __name__ == "__main__":
#     # Load the data
#     data_for_dt = "data_for_rules.csv"
#     X, y, label_dict, mapping = load_data(data_for_dt)
#     # print(mapping)
#     X_train, X_test, y_train, y_test, X_test_with_id = stratified_split_by_patient(X, y)
#
#     reversed_label_dict = {v: k for k, v in label_dict.items()}
#     class_names = [reversed_label_dict[cat] for cat in sorted(reversed_label_dict.keys())]
#
#     decision_tree = DecisionTreeClassifier(random_state=39)
#     decision_tree.fit(X_train, y_train)
#
#     tree = decision_tree.tree_
#     feature_names = list(X_train.columns)
#     # results_df = tree_to_sentences(tree, feature_names, class_names, node_id=0, path_dict=None, results_df=None)
#     #
#     # for result in results_df["Sentence"]:
#     #     print(result)
#     # results_df.to_csv("decision_tree_sentences.csv", index=False)
#
#     rules = get_rules(decision_tree, feature_names, class_names)
#     for r in rules:
#         print(r)


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
    #     plt.savefig(f"figures/decision_tree_{label_name.replace(' ', '_')}.png", dpi=300)
    #     plt.close()