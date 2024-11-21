"""
The main script for the pan_cancer project.
"""
from SHAP_explain import *
from XGBoost_Model import *


if __name__ == "__main__":
    # Load the data
    filepath = "pan_cancer_data_for_model.csv"
    X_train, X_val, X_test, y_train, y_val, y_test, label_dict = load_and_split_data(filepath)

    # Fit and evaluate model
    validation_accuracy, model, predictions = fit_and_evaluate_model(X_train, X_val, y_train, y_val, label_dict)

    y_val = np.array(y_val).flatten()  # Ensure 1D array

    # Extract predicted class indices
    # predicted_classes = np.argmax(predictions, axis=0)

    # Map label dictionary to get class names
    class_names = list(label_dict.values())

    # Analyze SHAP contributions and plot
    contributions_summary = analyze_shap_contributions_with_plots(
        model, X_val, y_val, predictions, class_names
    )

    # Print results
    print(f"Validation Accuracy: {validation_accuracy}")
    for class_name, summary in contributions_summary.items():
        print(f"\nFor {class_name}, contributions:")
        print(summary.head())

