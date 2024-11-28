"""
The main script for the pan_cancer project.
"""
from SHAP_explain import *
from XGBoost_Model import *


if __name__ == "__main__":
    # Load the data
    filepath = "pan_cancer_data_for_model.csv"
    X, y, label_dict, mapping = load_data(filepath)
    X_train, X_val, X_test, y_train, y_val, y_test, X_val_with_id = stratified_split_by_patient(X, y)

    # Fit and evaluate model
    validation_accuracy, model, predictions \
        = fit_and_evaluate_model(X_train, X_val, y_train, y_val, label_dict, show_plots=False)

    # Per Patient prediction
    classify_patients(X_val_with_id, predictions, y_val, label_dict)

    # Perform SHAP analysis
    feature_importance = shap_analysis(model, X_val, y_val, predictions, label_dict)

    # Generate hypotheses database
    hypotheses = apply_category_mappings(generate_hypotheses_db(model, X_val, y_val, label_dict), mapping)

    hypotheses.to_csv("hypotheses.csv", index=False)
    # Print results
    print(f"Validation Accuracy: {validation_accuracy}")

