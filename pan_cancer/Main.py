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
import lightgbm as lgb
from Decision_tree_sentences import *
from rule_based import *


def main():
    """
    Main entry point for the cancer classification program. Handles command-line argument parsing
    and orchestrates the execution of either rules-based or model-based classification.

    Command Line Arguments:
    For rules mode:
        rules
            --save_df (bool): Whether to save the rules dataframe to CSV
            --is_plot_run (bool): Whether to generate and display plots

    For regular mode:
        regular
            --model_name (str): Type of model to use ('forest', 'tree', 'xgb')
            --use_pickled (bool): Whether to use a previously saved model
            --gpu (bool): Whether the model should use GPU acceleration
            --show_plots (bool): Whether to display performance plots
            --print_eval (bool): Whether to print evaluation metrics
            --generate_db (bool): Whether to generate hypotheses database
            --get_shap_interactions (bool): Whether to calculate SHAP interactions

    Examples:
        Rules mode:
        $ python Main.py rules --save_df True --is_plot_run True

        Regular mode:
        $ python Main.py regular --model_name forest --show_plots True --print_eval True

    Returns:
        None

    Raises:
        FileNotFoundError: If required data files are not found
        ValueError: If invalid model_name is provided

    Dependencies:
        - argparse: For command-line argument parsing
        - Required data files:
            - all_cancers_data.csv
            - narrowed_cancers_data.csv
            - data_for_rules.csv
    """
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
    regular_parser.add_argument('--model_name', choices=['forest', 'tree', 'xgb', 'lgb'],
                                required=True, help='Model type to use')
    regular_parser.add_argument('--use_pickled', type=bool, default=False,
                                help='Use pickled model flag')
    regular_parser.add_argument('--show_plots', type=bool, default=False,
                                help='Show plots flag')
    regular_parser.add_argument('--print_eval', type=bool, default=False,
                                help='Print evaluation flag')
    regular_parser.add_argument('--generate_db', type=bool, default=False,
                                help='Generate hypotheses database flag')
    regular_parser.add_argument('--get_shap_interactions', type=bool, default=False,
                                help='Get SHAP interactions flag')
    regular_parser.add_argument('--by_id', type=bool, default=False,
                                help='Get predictions by patient ID flag')
    regular_parser.add_argument('--gpu', type=bool, default=False,
                                help='Use GPU flag')
    args = parser.parse_args()

    # Load the data
    all_cancers = "all_cancers_data.csv"
    partial_cancers = "narrowed_cancers_data.csv"
    data_for_rules = "data_for_rules.csv"
    data_for_decision = "data_for_decision.csv"
    if args.mode == 'rules':
        X, y, label_dict, mapping = load_data(data_for_rules)
    elif args.model_name == "tree":
        X, y, label_dict, mapping = load_data_non_categorical(data_for_decision)
    else:
        X, y, label_dict, mapping = load_data(partial_cancers)
    X_train, X_test, y_train, y_test, X_test_with_id = stratified_split_by_patient(X, y)

    if args.mode == 'rules':
        run_rules(X_train, X_test, y_train, y_test, label_dict, mapping, args)
    elif args.mode == 'regular':
        run_regular(X, y, X_train, X_test, y_train, y_test, X_test_with_id, label_dict, mapping, args)


def run_regular(X, y, X_train, X_test, y_train, y_test, X_test_with_id, label_dict, mapping, args):
    # Initialize models
    model_dict = {
        "xgb": xgb.XGBClassifier(n_estimators=192, objective='multi:softmax',
                                 tree_method='hist', enable_categorical=True, booster='dart',
                                 learning_rate=0.08596994335468267, max_depth=12, max_leaves=80, max_bin=508,
                                min_child_weight=2.89592834170114, gamma=0.6679875084005478,
                                 subsample=0.9972816061667134, colsample_bytree=0.9104275817039963,
                                 colsample_bylevel=0.5696232331004479, colsample_bynode=0.8743322819039548,
                                reg_alpha=0.19962668333554703, reg_lambda=0.3945177492732007,
                                 sample_type='weighted', normalize_type='tree', rate_drop=0.0004722716577317265,
                                 skip_drop=0.32192704523754445),
        "forest": RandomForestClassifier(random_state=39),
        "tree": DecisionTreeClassifier(random_state=39, min_samples_leaf=10, max_depth=10),
        "lgb": lgb.LGBMClassifier(n_estimators=250, objective='multiclass', metric='multi_logloss', learning_rate=0.05332631440912483,
                                  num_leaves=137, max_depth=20, min_child_samples=25, subsample=0.7645079558344258,
                                  colsample_bytree=0.8380036267505455)
    }

    model = model_dict[args.model_name]
    model_type = {"xgb": "XGBoost", "forest": "Random_Forest", "tree": "Decision_Tree", "lgb": "LightGBM"}[args.model_name]

    start_time = time.time()
    print(f"\n\n{'*' * 10} {model_type} {'*' * 10}\n")

    if args.model_name == "tree":
        feature_names = list(X_train.columns)
        model_dict["tree"].fit(X_train, y_train)
        sentences = get_readable_sentences(model_dict["tree"], feature_names, label_dict, mapping)

        for sentence in sentences:
            print(sentence)

        print(f"{model_type} - Run time: {time.time() - start_time} seconds\n\n")
        return

    elif args.use_pickled:
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
        if args.by_id:
            classify_patients(X_test_with_id, y_pred, y_test, label_dict, model_type)
        if args.gpu:
            explainer = shap.explainers.GPUTreeExplainer(model)
            pickle.dump(explainer, open(f"models_and_explainers/{model_type}_gpu_explainer.pkl", "wb"))
        else:
            explainer = shap.TreeExplainer(model)
            pickle.dump(explainer, open(f"models_and_explainers/{model_type}_explainer.pkl", "wb"))

        pickle.dump(model, open(f"models_and_explainers/{model_type}_model.pkl", "wb"))

    if args.generate_db:
        hypotheses = generate_hypotheses_db(explainer, model, X_test, y_test, label_dict, mapping)
        hypotheses.to_csv(f"models_hypotheses/{model_type}_hypotheses_as_sentences.csv", index=False)

    if args.get_shap_interactions:
        interation_vals = get_shap_interactions(explainer, X, y, label_dict)
        pickle.dump(interation_vals, open(f"models_and_explainers/{model_type}_shap_interactions.pkl", "wb"))

    print(f"{model_type} - Run time: {time.time() - start_time} seconds\n\n")


def run_rules(X_train, X_test, y_train, y_test, label_dict, mapping, args):
    rules = rule_based(X_train, X_test, y_train, y_test, label_dict, is_plot_run=args.is_plot_run)

    if args.save_df:
        rules = convert_rules_to_readable(rules, mapping)
        rules_df = create_rules_dataframe(rules)
        rules_df.to_csv("models_hypotheses/rules_df.csv", index=False)

if __name__ == "__main__":
    main()
