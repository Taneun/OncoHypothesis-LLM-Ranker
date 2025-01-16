import pandas as pd
from itertools import combinations
from tqdm import tqdm

def data_prep_lift(df):
    # Select a subset of columns to analyze (e.g., most relevant ones)
    columns_to_combine = ['Sex', 'Smoke Status', 'Chromosome', 'Hugo_Symbol', 'SNP_event', "Consequence", 'Exon_Number',
                          "Diagnosis Age", "TMB (nonsynonymous)", "Position", "Protein_position", "Codons", "VAR_TYPE_SX"]

    cancer_probabilities = {cancer_type: data_for_lift[cancer_type].mean() for cancer_type in list(df["Cancer Type"].unique())}
    feature_combinations = list(combinations(columns_to_combine, 5))

    return cancer_probabilities, feature_combinations


def calculate_lift(data_for_lift, cancer_probabilities, feature_combinations):
    lifts = []

    for cancer_type, P_B in tqdm(cancer_probabilities.items(), desc="Cancer Types", unit="type"):
        for feature in tqdm(feature_combinations, desc="Feature Combinations", unit="comb", leave=False):
            # Combine the selected features into a single feature
            combined_feature = data_for_lift[list(feature)].astype(str).agg('_'.join, axis=1)

            # Compute value counts for the combined feature
            combined_counts = combined_feature.value_counts()
            valid_features = combined_counts[combined_counts >= 100].index

            if len(valid_features) == 0:  # Skip if no valid combined features
                continue

            # Filter the combined feature to include only valid entries
            filtered_data = combined_feature[combined_feature.isin(valid_features)]

            # Reset index to align with the original DataFrame for filtering the cancer type
            filtered_data = filtered_data.reset_index(drop=True)
            cancer_data = data_for_lift[cancer_type].reset_index(drop=True)  # Make sure cancer_data has the same index

            # Compute joint probabilities for cancer type
            joint_prob = (
                filtered_data[cancer_data == 1]
                .value_counts(normalize=True)
                .reindex(filtered_data.value_counts(normalize=True).index, fill_value=0)
            )

            # Calculate lift
            P_A = filtered_data.value_counts(normalize=True)
            lift = (joint_prob / (P_A * P_B)).round(2)

            # Store results as a list of tuples
            lifts.append((cancer_type, feature, lift))
    counter+=1
    # Return lifts as a DataFrame
    lift_data = []
    for cancer_type, feature, lift in lifts:
        for idx, lift_value in lift.items():
            lift_data.append(
                {"Cancer Type": cancer_type, "Feature Combination": feature, "Lift Value": lift_value, "Feature": idx})

    lift_df = pd.DataFrame(lift_data)
    return lift_df


if __name__ == "__main__":
    data_for_lift = pd.read_csv("./pan_cancer/data_for_lift.csv", index_col=0)
    cancer_prob, features_comb = data_prep_lift(data_for_lift)
    lifts_df = calculate_lift(data_for_lift, cancer_prob, features_comb)
    lifts_df.to_csv("lifts.csv", index=False)



