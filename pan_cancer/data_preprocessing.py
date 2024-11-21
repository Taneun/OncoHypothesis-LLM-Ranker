import pandas as pd


# reading all the data files
data_clinical_patient = pd.read_csv('pan_origimed_2020/data_clinical_patient.txt', sep="\t")
data_clinical_sample = pd.read_csv('pan_origimed_2020/data_clinical_sample.txt', sep="\t")
data_cna_log2 = pd.read_csv('pan_origimed_2020/data_cna_log2.txt', sep="\t")
data_cna = pd.read_csv('pan_origimed_2020/data_cna.txt', sep="\t")
data_mutations = pd.read_csv('pan_origimed_2020/data_mutations.txt', sep="\t", header=2, dtype={"Exon_Number": "string"})
data_sv = pd.read_csv('pan_origimed_2020/data_sv.txt', sep="\t")

# removing bad rows
data_clinical_sample = data_clinical_sample[4:]

# matching the sample id to match other tables
data_clinical_patient["SAMPLE_ID"] = data_clinical_patient["PATIENT_ID"].apply(lambda x: "P-" + x[7:])

# make all sample id header name the same - "SAMPLE_ID"
data_clinical_sample.rename(columns={"Sample Identifier": 'SAMPLE_ID'}, inplace=True)
data_mutations.rename(columns={"Tumor_Sample_Barcode": 'SAMPLE_ID'}, inplace=True)
data_sv.rename(columns={"Sample_Id": 'SAMPLE_ID'}, inplace=True)

merged_clinical_data = data_clinical_patient.merge(data_clinical_sample, on="SAMPLE_ID", how='outer')
merged_mutations_data = merged_clinical_data.merge(data_mutations, on="SAMPLE_ID", how='outer')
merged_all_data = merged_mutations_data.merge(data_sv, on="SAMPLE_ID", how='outer')

merged_all_data.to_csv("pan_cancer_db_merged.csv")

merged_all_data["SNP_event"] = merged_all_data["Reference_Allele"].fillna("").astype(str) + ">" \
                               + merged_all_data["Tumor_Seq_Allele2"].fillna("").astype(str)

data_for_model = merged_all_data[["PATIENT_ID", "SEX", "DIAGNOSIS AGE", "SMOKE STATUS", "TMB (nonsynonymous)",
                                "Hugo_Symbol", "Chromosome", "Start_Position", "End_Position",
                                "Consequence", "Variant_Type", "SNP_event", "Protein_position", "Codons",
                                "Exon_Number","VAR_TYPE_SX", "Site1_Hugo_Symbol", "Site2_Hugo_Symbol","Event_Info"]]

data_for_model.to_csv("pan_cancer_db_for_model.csv")


def convert_exon_number(val):
    """
    checking if the dates in Exon number are dd/mm or mm/dd by comparing the dates and not dates for gene APC
    data_for_model[data_for_model["Hugo_Symbol"].str.contains("APC", na=False)]
    result - the format of the dates are total exons-exon number
    :param val:
    :return:
    """
    try:
        # First, try to convert to 'Month-Year' format (e.g., 'Sep-89' -> '09/89')
        return pd.to_datetime(val, format='%b-%y').strftime('%m/%y')
    except ValueError:
        pass

    try:
        # Then, try to convert to 'DD-Mon' format (e.g., '14-Sep' -> '09/14')
        date_obj = pd.to_datetime(val, format='%d-%b', errors='raise')
        return date_obj.strftime('%m/%d')
    except ValueError:
        # If neither format matches, return the value as is (non-date-like string)
        return val

# Apply the function to the column
data_for_model['Exon_Number'] = data_for_model['Exon_Number'].apply(convert_exon_number)

data_for_model['Consequence'].str.split(',')
dummy_vars = data_for_model['Consequence'].str.split(',').explode().str.get_dummies().groupby(level=0).sum()
data_for_model = data_for_model.join(dummy_vars)

data_for_model = pd.get_dummies(data_for_model, columns=['SMOKE STATUS', 'SEX', 'VAR_TYPE_SX', 'Chromosome', 'Variant_Type'], drop_first=False)
data_for_model.drop('SMOKE STATUS_Unknown', axis=1, inplace=True)