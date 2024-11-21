import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_shap_contributions_with_plots(model, X, y, predicted_classes, class_names):
    """
    Analyze and visualize SHAP contributions for correct predictions, per class.

    Parameters:
    model : Trained XGBoost model
        The trained multiclass classifier.
    X : pandas.DataFrame
        Feature dataset used for predictions.
    y : array-like
        True class labels corresponding to X.
    class_names : list
        List of class names corresponding to model outputs.

    Returns:
    contributions_summary : dict
        A dictionary summarizing feature value contributions by class.
    """
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)  # Predicted class indices

    # Track contributions per class
    contributions_summary = {class_name: [] for class_name in class_names}

    # Iterate over each instance
    for idx, (true_class, pred_class) in enumerate(zip(y, predicted_classes)):
        if true_class == pred_class:  # Correct prediction
            shap_contributions = shap_values[pred_class][idx]  # SHAP for predicted class
            instance = X.iloc[idx]

            # Identify features with significant contributions
            significant_features = [
                (feature, value, shap_contributions[i])
                for i, (feature, value) in enumerate(instance.items())
                if abs(shap_contributions[i]) > 0.1  # Threshold for significance todo: IndexError: index 25 is out of bounds for axis 0 with size 25
            ]

            # Append significant feature values for this class
            contributions_summary[class_names[pred_class]].extend(significant_features)

    # Summarize contributions
    summarized_contributions = {}
    for class_name in class_names:
        summary_df = pd.DataFrame(contributions_summary[class_name],
                                  columns=["Feature", "Value", "Contribution"])
        summarized = (
            summary_df
            .groupby(["Feature", "Value"])
            .size()
            .reset_index(name="Count")
            .sort_values(by="Count", ascending=False)
        )
        summarized_contributions[class_name] = summarized

        # Plot top 10 feature-value pairs for this class
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=summarized.head(10),
            x="Count",
            y=summarized["Feature"].astype(str) + " = " + summarized["Value"].astype(str),
            palette="viridis"
        )
        plt.title(f"Top Feature Contributions for {class_name}")
        plt.xlabel("Frequency of Contribution")
        plt.ylabel("Feature = Value")
        plt.tight_layout()
        plt.show()

    return summarized_contributions

