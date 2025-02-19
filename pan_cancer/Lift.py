# import pandas as pd
# from itertools import combinations
# from tqdm import tqdm
#
# def data_prep_lift(df):
#     # Select a subset of columns to analyze (e.g., most relevant ones)
#     columns_to_combine = ['Sex', 'Smoke Status', 'Chromosome', 'Hugo_Symbol', 'SNP_event', "Consequence", 'Exon_Number',
#                           "Diagnosis Age", "TMB (nonsynonymous)", "Position", "Protein_position", "Codons", "VAR_TYPE_SX"]
#
#     cancer_probabilities = {cancer_type: df[cancer_type].mean() for cancer_type in list(df["Cancer Type"].unique())}
#     feature_combinations = list(combinations(columns_to_combine, 5))
#
#     return cancer_probabilities, feature_combinations
#
#
# def calculate_lift(data_for_lift, cancer_probabilities, feature_combinations):
#     lifts = []
#
#     for cancer_type, P_B in tqdm(cancer_probabilities.items(), desc="Cancer Types", unit="type"):
#         for feature in tqdm(feature_combinations, desc="Feature Combinations", unit="comb", leave=False):
#
#             # Combine the selected features into a single feature
#             combined_feature = data_for_lift[list(feature)].astype(str).agg('_'.join, axis=1)
#             combined_feature = combined_feature.reset_index().drop_duplicates().set_index("PATIENT_ID")
#
#             # Compute value counts for the combined feature
#             combined_counts = combined_feature.value_counts()
#             valid_features = combined_counts[combined_counts >= 50].index
#
#             if len(valid_features) == 0:  # Skip if no valid combined features
#                 continue
#
#             # Filter the combined feature to include only valid entries
#             filtered_data = combined_feature[combined_feature.isin(valid_features)]
#             filtered_data = data_for_lift.loc[combined_feature.isin(valid_features), feature].astype(str).agg('_'.join,
#                                                                                                               axis=1)
#
#             # Reset index to align with the original DataFrame for filtering the cancer type
#             filtered_data = filtered_data.reset_index(drop=True)
#             cancer_data = data_for_lift[cancer_type].reset_index(drop=True)  # Make sure cancer_data has the same index
#             cancer_data = data_for_lift.loc[filtered_data.index, cancer_type]
#
#             # Compute joint probabilities for cancer type
#             joint_prob = (
#                 filtered_data[cancer_data == 1]
#                 .value_counts(normalize=True)
#                 .reindex(filtered_data.value_counts(normalize=True).index, fill_value=0)
#             )
#
#             # Calculate lift
#             P_A = filtered_data.value_counts(normalize=True)
#             lift = (joint_prob / (P_A * P_B)).round(2)
#
#             # Store results as a list of tuples
#             lifts.append((cancer_type, feature, lift))
#     # Return lifts as a DataFrame
#     lift_data = []
#     for cancer_type, feature, lift in lifts:
#         for idx, lift_value in lift.items():
#             lift_data.append(
#                 {"Cancer Type": cancer_type, "Feature Combination": feature, "Lift Value": lift_value, "Feature": idx})
#
#     lift_df = pd.DataFrame(lift_data)
#     return lift_df
#
#
# if __name__ == "__main__":
#     data_for_lift = pd.read_csv("data_for_lift.csv", index_col=0)
#     cancer_prob, features_comb = data_prep_lift(data_for_lift)
#     lifts_df = calculate_lift(data_for_lift, cancer_prob, features_comb)
#     lifts_df.to_csv("lifts.csv", index=False)

import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import warnings

warnings.filterwarnings('ignore')


def data_prep_lift(data_for_lift):
    columns_to_combine = ['Sex', 'Smoke Status', 'Chromosome', 'Hugo_Symbol', 'SNP_event',
                          "Consequence", 'Exon_Number', "Diagnosis Age", "TMB (nonsynonymous)",
                          "Position", "Protein_position", "Codons", "VAR_TYPE_SX"]

    # Calculate cancer probabilities using the correct dataframe
    cancer_probabilities = {
        cancer_type: data_for_lift[cancer_type].mean()
        for cancer_type in data_for_lift.select_dtypes(include=['bool', 'int']).columns
    }

    # Create feature combinations
    feature_combinations = list(combinations(columns_to_combine, 5))

    return cancer_probabilities, feature_combinations


def process_combination(args):
    data_for_lift, cancer_type, P_B, feature = args

    # Combine features efficiently
    combined_feature = data_for_lift[list(feature)].fillna('missing').astype(str).agg('_'.join, axis=1)

    # Filter features with sufficient occurrences
    feature_counts = combined_feature.value_counts()
    valid_features = feature_counts[feature_counts >= 50].index

    if len(valid_features) == 0:
        return None

    # Create mask for valid features
    valid_mask = combined_feature.isin(valid_features)
    filtered_data = combined_feature[valid_mask]
    cancer_data = data_for_lift.loc[valid_mask, cancer_type]

    # Compute probabilities
    P_A = filtered_data.value_counts(normalize=True)

    # Calculate joint probability more efficiently
    joint_counts = pd.Series(0, index=P_A.index)
    positive_counts = filtered_data[cancer_data == 1].value_counts()
    joint_counts.update(positive_counts)
    joint_prob = joint_counts / len(filtered_data)

    # Calculate lift
    lift = (joint_prob / (P_A * P_B)).round(2)

    # Return results for valid calculations
    return [(cancer_type, feature, idx, lift_val)
            for idx, lift_val in lift.items()
            if not np.isnan(lift_val) and not np.isinf(lift_val)]


def calculate_lift(data_for_lift, cancer_probabilities, feature_combinations):
    # Prepare arguments for parallel processing
    args_list = [
        (data_for_lift, cancer_type, P_B, feature)
        for cancer_type, P_B in cancer_probabilities.items()
        for feature in feature_combinations
    ]

    # Process combinations in parallel
    lift_data = []
    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(process_combination, args_list),
                           total=len(args_list),
                           desc="Processing combinations"):
            if result:
                lift_data.extend([{
                    "Cancer Type": cancer_type,
                    "Feature Combination": feature,
                    "Feature": idx,
                    "Lift Value": lift_val
                } for cancer_type, feature, idx, lift_val in result])

    return pd.DataFrame(lift_data)


if __name__ == "__main__":
    data_for_lift = pd.read_csv("data_for_lift.csv", index_col=0)
    cancer_prob, features_comb = data_prep_lift(data_for_lift)
    lifts_df = calculate_lift(data_for_lift, cancer_prob, features_comb)
    lifts_df.to_csv("lifts.csv", index=False)