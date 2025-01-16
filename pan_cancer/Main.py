"""
The main script for the pan_cancer project.
"""
import argparse
import time
from SHAP_explain import *
from XGBoost_Model import *
from model_metrics import *
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from rule_based import *


def main():
    parser = argparse.ArgumentParser(description='Cancer Classification Model')
    subparsers = parser.add_subparsers(dest='mode', help='Choose mode: rules or regular')

    # Rules parser
    rules_parser = subparsers.add_parser('rules')
    rules_parser.add_argument('--save_df', type=bool, default=False,
                              help='Save dataframe flag')
    rules_parser.add_argument('--is_plot_run', type=bool, default=False,
                              help='Plot run flag')

    # Regular parser
    regular_parser = subparsers.add_parser('regular')
    regular_parser.add_argument('--model_name', choices=['forest', 'tree', 'xgb'],
                                required=True, help='Model type to use')
    regular_parser.add_argument('--use_pickled', type=bool, default=False,
                                help='Use pickled model flag')
    regular_parser.add_argument('--show_plots', type=bool, default=False,
                                help='Show plots flag')
    regular_parser.add_argument('--print_eval', type=bool, default=False,
                                help='Print evaluation flag')
    regular_parser.add_argument('--generate_hypotheses_db', type=bool, default=False,
                                help='Generate hypotheses database flag')
    regular_parser.add_argument('--get_shap_interactions', type=bool, default=False,
                                help='Get SHAP interactions flag')
    args = parser.parse_args()

    # Load the data
    all_cancers = "all_cancers_data.csv"
    partial_cancers = "narrowed_cancers_data.csv"
    data_for_rules = "data_for_rules.csv"
    X, y, label_dict, mapping = load_data(partial_cancers)
    X_train, X_test, y_train, y_test, X_test_with_id = stratified_split_by_patient(X, y)

    if args.mode == 'rules':
        run_rules(X_train, X_test, y_train, y_test, label_dict, mapping, args)
    elif args.mode == 'regular':
        run_regular(X, y, X_train, X_test, y_train, y_test, X_test_with_id, label_dict, mapping, args)


def run_regular(X, y, X_train, X_test, y_train, y_test, X_test_with_id, label_dict, mapping, args):
    # Initialize models
    model_dict = {
        "xgb": xgb.XGBClassifier(n_estimators=250, objective='multi:softmax',
                                 tree_method='hist', enable_categorical=True),
        "forest": RandomForestClassifier(random_state=39),
        "tree": DecisionTreeClassifier(random_state=39)
    }

    model = model_dict[args.model_name]
    model_type = {"xgb": "XGBoost", "forest": "Random Forest", "tree": "Decision Tree"}[args.model_name]

    start_time = time.time()
    print(f"\n\n{'*' * 10} {model_type} {'*' * 10}\n")

    if args.use_pickled:
        model = pickle.load(open(f"models_and_explainers/{model_type}_model.pkl", "rb"))
        explainer = pickle.load(open(f"models_and_explainers/{model_type}_explainer.pkl", "rb"))
    else:
        # Fit and evaluate the model
        model, y_pred = fit_and_evaluate(
            model_type, model, X_train, X_test, y_train, y_test, label_dict,
            print_eval=args.print_eval,
            show_auc=args.show_plots,
            show_cm=args.show_plots,
            show_precision_recall=args.show_plots
        )
        classify_patients(X_test_with_id, y_pred, y_test, label_dict, model_type)
        explainer = shap.TreeExplainer(model)

        # Save the model
        pickle.dump(model, open(f"models_and_explainers/{model_type}_model.pkl", "wb"))
        pickle.dump(explainer, open(f"models_and_explainers/{model_type}_explainer.pkl", "wb"))

    if args.generate_hypotheses_db:
        hypotheses = generate_hypotheses_db(explainer, model, X_test, y_test, label_dict, mapping)
        hypotheses.to_csv(f"models_hypotheses/{model_type}_hypotheses_as_sentences.csv", index=False)

    if args.get_shap_interactions:
        get_shap_interactions(explainer, X, y, label_dict)

    print(f"{model_type} - Run time: {time.time() - start_time} seconds\n\n")


def run_rules(X_train, X_test, y_train, y_test, label_dict, mapping, args):
    rules = rule_based(X_train, X_test, y_train, y_test, label_dict, is_plot_run=args.is_plot_run)

    if args.save_df:
        rules = convert_rules_to_readable(rules, mapping)
        rules_df = create_rules_dataframe(rules)
        rules_df.to_csv("models_hypotheses/rules_df.csv", index=False)


if __name__ == "__main__":
    main()
