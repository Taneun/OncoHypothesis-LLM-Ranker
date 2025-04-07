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

    # Count the number of unique PATIENT_IDs per feature combination
    feature_counts = (
        data_for_lift.reset_index()
        .groupby(list(feature))['PATIENT_ID']
        .nunique()
        .reset_index(name='patient_count')
    )

    # Create the feature combination column
    feature_counts["feature_combination"] = feature_counts[list(feature)].apply(tuple, axis=1)

    feature_counts = feature_counts[feature_counts["patient_count"] >= 50]
    # Filter valid feature combinations with at least 50 unique patients
    valid_features = set(feature_counts["feature_combination"].astype(str).unique())

    # Combine features in the original dataset
    combined_feature = data_for_lift[list(feature)].apply(tuple, axis=1)

    # Apply mask for valid features
    valid_mask = combined_feature.astype(str).isin(valid_features)
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
                    "Feature": tuple(idx),  # Ensure "Feature" is stored as a tuple
                    "Lift Value": lift_val
                } for cancer_type, feature, idx, lift_val in result])

    return pd.DataFrame(lift_data)


if __name__ == "__main__":
    data_for_lift = pd.read_csv("data_for_lift.csv", index_col=0)
    cancer_prob, features_comb = data_prep_lift(data_for_lift)
    lifts_df = calculate_lift(data_for_lift, cancer_prob, features_comb)
    # lifts_df.to_csv("lifts.csv", index=False)
