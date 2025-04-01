from XGBoost_Model import *
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.tree import _tree

def get_readable_sentences(tree, feature_names, class_names, mapping):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths, smoking_status_conditions=None):
        if smoking_status_conditions is None:
            smoking_status_conditions = []

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
                          'chr_X', 'Unknown', 'Smoker', 'Nonsmoker']

            if name in dummy_vars:
                if name.startswith('chr_'):
                    p1.append((name, f"Chromosome is NOT {name[4:]}"))
                    recurse(tree_.children_left[node], p1, paths, smoking_status_conditions)
                    p2.append((name, f"Chromosome is {name[4:]}"))
                    recurse(tree_.children_right[node], p2, paths, smoking_status_conditions)
                elif name in ['Unknown', 'Smoker', 'Nonsmoker']:
                    if name == 'Unknown':
                        smoking_status_conditions.append("Smoking status is Unknown")
                        p1.append((name, f"Smoking status is Smoker or Nonsmoker"))
                        recurse(tree_.children_left[node], p1, paths, smoking_status_conditions)
                        p2.append((name, f"Smoking status is {name}"))
                        recurse(tree_.children_right[node], p2, paths, smoking_status_conditions)
                    elif name == 'Smoker':
                        smoking_status_conditions.append("Smoking status is Smoker")
                        p1.append((name, f"Smoking status is Nonsmoker or Unknown"))
                        recurse(tree_.children_left[node], p1, paths, smoking_status_conditions)
                        p2.append((name, f"Smoking status is {name}"))
                        recurse(tree_.children_right[node], p2, paths, smoking_status_conditions)
                    elif name == 'Nonsmoker':
                        smoking_status_conditions.append("Smoking status is Nonsmoker")
                        p1.append((name, f"Smoking status is Smoker or Unknown"))
                        recurse(tree_.children_left[node], p1, paths, smoking_status_conditions)
                        p2.append((name, f"Smoking status is {name}"))
                        recurse(tree_.children_right[node], p2, paths, smoking_status_conditions)
                else:
                    p1.append((name, f"NOT {name}"))
                    recurse(tree_.children_left[node], p1, paths, smoking_status_conditions)
                    p2.append((name, f"{name}"))
                    recurse(tree_.children_right[node], p2, paths, smoking_status_conditions)
            elif name == "Sex":
                p1.append((name, "Sex is Female"))
                recurse(tree_.children_left[node], p1, paths, smoking_status_conditions)
                p2.append((name, "Sex is Male"))
                recurse(tree_.children_right[node], p2, paths, smoking_status_conditions)
            elif name == "VAR_TYPE_SX":
                p1.append((name, f"Variant type is Substitution/Indel"))
                recurse(tree_.children_left[node], p1, paths, smoking_status_conditions)
                p2.append((name, f"Variant type is Truncation"))
                recurse(tree_.children_right[node], p2, paths, smoking_status_conditions)
            elif threshold == 0.5:
                p1.append((name, f"{name} is 0 (False)"))
                recurse(tree_.children_left[node], p1, paths, smoking_status_conditions)
                p2.append((name, f"{name} is 1 (True)"))
                recurse(tree_.children_right[node], p2, paths, smoking_status_conditions)
            elif name in mapping:
                p1.append(f"{name} is {mapping[name][threshold]}")
                p2.append(f"{name} is not {mapping[name][threshold]}")
            else:
                p1.append((name, f"{name} ≤ {threshold}"))
                recurse(tree_.children_left[node], p1, paths, smoking_status_conditions)
                p2.append((name, f"{name} > {threshold}"))
                recurse(tree_.children_right[node], p2, paths, smoking_status_conditions)
        else:
            path.append((None, (tree_.value[node], tree_.n_node_samples[node])))
            paths.append(path)

    recurse(0, path, paths)

    # Sort by sample count
    paths.sort(key=lambda x: x[-1][1][1], reverse=True)

    readable_sentences = []
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

        # Handle the smoking status merging based on your updated conditions:
        if "Smoking status is Unknown" in merged_conditions:
            if "Smoking status is Smoker" in merged_conditions:
                # Remove "Unknown" and only keep "Smoker"
                merged_conditions = [cond for cond in merged_conditions if "Smoking status" not in cond]
                merged_conditions.insert(0, "Smoking status is Smoker")
            elif "Smoking status is Nonsmoker" in merged_conditions:
                # Remove "Unknown" and only keep "Nonsmoker"
                merged_conditions = [cond for cond in merged_conditions if "Smoking status" not in cond]
                merged_conditions.insert(0, "Smoking status is Nonsmoker")
        elif "Smoking status is Smoker" in merged_conditions and "Smoking status is Nonsmoker" in merged_conditions:
            # Keep both Smoker or Nonsmoker
            merged_conditions = [cond for cond in merged_conditions if "Smoking status" not in cond]
            merged_conditions.insert(0, "Smoking status is Smoker or Nonsmoker")

        # New logic to handle "Smoking status is X or Y" and "Smoking status is X"
        if ("Smoking status is Smoker or Nonsmoker" in merged_conditions
                or "Smoking status is Nonsmoker or Unknown" in merged_conditions
                or "Smoking status is Smoker or Unknown" in merged_conditions):
            if "Smoking status is Smoker" in merged_conditions:
                # Keep only "Smoking status is Smoker"
                merged_conditions = [cond for cond in merged_conditions if "Smoking status" not in cond]
                merged_conditions.insert(0, "Smoking status is Smoker")
            elif "Smoking status is Nonsmoker" in merged_conditions:
                # Keep only "Smoking status is Nonsmoker"
                merged_conditions = [cond for cond in merged_conditions if "Smoking status" not in cond]
                merged_conditions.insert(0, "Smoking status is Nonsmoker")
        if "Smoking status is Smoker or Nonsmoker" in merged_conditions:
            if "Smoking status is Nonsmoker or Unknown" in merged_conditions:
                merged_conditions = [cond for cond in merged_conditions if "Smoking status" not in cond]
                merged_conditions.insert(0, "Smoking status is Nonsmoker")
            if "Smoking status is Smoker or Unknown" in merged_conditions:
                merged_conditions = [cond for cond in merged_conditions if "Smoking status" not in cond]
                merged_conditions.insert(0, "Smoking status is Smoker")
        if "Smoking status is Nonsmoker or Unknown" in merged_conditions:
            if "Smoking status is Smoker or Unknown" in merged_conditions:
                merged_conditions = [cond for cond in merged_conditions if "Smoking status" not in cond]
                merged_conditions.insert(0, "Smoking status is Unknown")

        # Handle removal of "Smoking status is Unknown"
        merged_conditions = [cond for cond in merged_conditions if "Smoking status is Unknown" not in cond]

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
        readable_sentences.append(rule_sentence)

    return readable_sentences

