"""
The main script for the pan_cancer project.
"""
from SHAP_explain import *
from XGBoost_Model import *
from model_metrics import *
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.tree import DecisionTreeClassifier
from imodels.rule_set.skope_rules import SkopeRuleClassifier

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


if __name__ == "__main__":
    # Load the data
    filepath = "pan_cancer_data_for_model.csv"
    X, y, label_dict, mapping = load_data(filepath)
    X_train, X_test, y_train, y_test, X_test_with_id = stratified_split_by_patient(X, y)
    # xtra_cheese = xgb.XGBClassifier(n_estimators=250, objective='multi:softmax',
    #                                 tree_method='hist', enable_categorical=True)
    # rand_forest = RandomForestClassifier(random_state=39)
    # decision_tree = DecisionTreeClassifier(random_state=39)
    # model_dict = {"XGBoost": xtra_cheese, "Random Forest": rand_forest, "Decision Tree": decision_tree}
    # for model_type, model in model_dict.items():
    #     # Fit and evaluate the model
    #     model, y_pred = fit_and_evaluate(model_type, model, X_train, X_test, y_train, y_test, label_dict,
    #                      show_auc=True, show_cm=True, show_precision_recall=True)
    #     classify_patients(X_test_with_id, y_pred, y_test, label_dict, model_type)

        # Save the model
        # pickle.dump(model, open(f"{model_type}_model.pkl", "wb"))

    feature_names = list(X.columns)
    for cancer_type, idx in label_dict:
        rule_model = SkopeRuleClassifier()
        one_vs_all_y_train = y_train.apply(lambda x: 1 if x == idx else 0)
        one_vs_all_y_test = y_test.apply(lambda x: 1 if x == idx else 0)
        fit_and_evaluate(f"{cancer_type} vs. All", rule_model, X_train, X_test,
                         one_vs_all_y_train, one_vs_all_y_test, label_dict,
                         show_auc=True, show_cm=True, show_precision_recall=True)




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
