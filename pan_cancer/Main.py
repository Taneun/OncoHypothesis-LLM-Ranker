"""
The main script for the pan_cancer project.
"""
import time
from SHAP_explain import *
from XGBoost_Model import *
from model_metrics import *
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.tree import DecisionTreeClassifier, _tree
from rule_based import rule_based
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.tree import export_text
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from rule_based import *


if __name__ == "__main__":
    # Load the data
    all_cancers = "all_cancers_data.csv"
    partial_cancers = "narrowed_cancers_data.csv"
    data_for_rules = "data_for_rules.csv"
    X, y, label_dict, mapping = load_data(data_for_rules)
    print(mapping)
    X_train, X_test, y_train, y_test, X_test_with_id = stratified_split_by_patient(X, y)
    # xtra_cheese = xgb.XGBClassifier(n_estimators=250, objective='multi:softmax',
    #                                 tree_method='hist', enable_categorical=True)
    # rand_forest = RandomForestClassifier(random_state=39)
    decision_tree = DecisionTreeClassifier(random_state=39)
    # model_dict = {"XGBoost": xtra_cheese, "Random Forest": rand_forest, "Decision Tree": decision_tree}
    # for model_type, model in model_dict.items():
    #     # time model and explainer run time
    #     start_time = time.time()
    #     # fancy print the model type
    #     print(f"\n\n{'*' * 10} {model_type} {'*' * 10}\n")
        # Fit and evaluate the model
        # model, y_pred = fit_and_evaluate(model_type, model, X_train, X_test, y_train, y_test, label_dict,
        #                                  show_auc=True, show_cm=True, show_precision_recall=True)
        # classify_patients(X_test_with_id, y_pred, y_test, label_dict, model_type)
        # explainer = shap.TreeExplainer(model)

        # Save the model
        # pickle.dump(model, open(f"models_and_explainers/{model_type}_model.pkl", "wb"))
        # pickle.dump(explainer, open(f"models_and_explainers/{model_type}_explainer.pkl", "wb"))

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
        # load the model
        # model = pickle.load(open(f"models_and_explainers/{model_type}_model.pkl", "rb"))
        # explainer = pickle.load(open(f"models_and_explainers/{model_type}_explainer.pkl", "rb"))
        #
        # # feature_importance = shap_analysis(explainer, X_test, y_test, y_pred, label_dict)
        # # Generate hypotheses database
        # hypotheses = generate_hypotheses_db(explainer, model, X_test, y_test, label_dict, mapping)
        # hypotheses.to_csv(f"models_hypotheses/{model_type}_hypotheses_as_sentences.csv", index=False)
        #
        # # Print the run time
        # print(f"{model_type} - Run time: {time.time() - start_time} seconds\n\n")

    # optimize_skope_rules(X_train, X_test, y_train, y_test)
    # plot_only = True
    # rules = rule_based(X_train, X_test, y_train, y_test, label_dict, is_plot_run=True)
    # if not plot_only:
    #     rules = convert_rules_to_readable(rules, mapping)
    #     rules_df = create_rules_dataframe(rules)
    #     rules_df.to_csv("models_hypotheses/rules_df_nonsmoker_fixed.csv", index=False)

    # get_shap_interactions(explainer, X, y, label_dict) # very slow currently
