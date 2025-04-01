import math
import pandas as pd
import numpy as np
import requests
import time
import random
import re
import logging
import os
from pathlib import Path
import csv

# Configuration
class Config:
    RESULTS_DIR = Path("novelty_results")
    LOGS_DIR = Path("novelty_logs")
    CSV_PATH = Path("novelty_results/novelty_scores.csv")

# Initialize directories
Config.RESULTS_DIR.mkdir(exist_ok=True)
Config.LOGS_DIR.mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TEXT_FEATURES = ['Smoke Status','Site2_Hugo_Symbol','Event_Info','Site1_Hugo_Symbol','Sex','Hugo_Symbol']
DUMMIE_FEATURES = ['missense_variant', 'frameshift_variant',
       'stop_gained,splice_region_variant', 'stop_gained',
       'intron_variant', 'splice_region_variant,intron_variant',
       'inframe_deletion', 'intron_variant,non_coding_transcript_variant',
       'missense_variant,splice_region_variant',
       'downstream_gene_variant',
       'splice_region_variant,synonymous_variant', 'synonymous_variant',
       'splice_acceptor_variant', 'upstream_gene_variant',
       'inframe_insertion', 'splice_donor_variant',
       'splice_acceptor_variant,intron_variant',
       'inframe_deletion,splice_region_variant',
       'frameshift_variant,splice_region_variant', '5_prime_UTR_variant',
       'splice_acceptor_variant,coding_sequence_variant,intron_variant',
       'start_lost', 'protein_altering_variant', 'stop_lost',
       'splice_donor_variant,coding_sequence_variant,intron_variant',
       'splice_acceptor_variant,coding_sequence_variant',
       'start_lost,5_prime_UTR_variant',
       'splice_region_variant,5_prime_UTR_variant',
       'splice_donor_variant,intron_variant',
       'frameshift_variant,start_lost,start_retained_variant',
       'stop_gained,inframe_insertion',
       'splice_donor_variant,coding_sequence_variant',
       'stop_lost,3_prime_UTR_variant', '3_prime_UTR_variant',
       'stop_gained,frameshift_variant',
       'inframe_insertion,splice_region_variant',
       'frameshift_variant,start_lost', 'frameshift_variant,stop_lost',
       'start_lost,start_retained_variant,5_prime_UTR_variant',
       'splice_region_variant,3_prime_UTR_variant',
       'non_coding_transcript_exon_variant',
       'stop_gained,inframe_deletion',
       'splice_acceptor_variant,coding_sequence_variant,3_prime_UTR_variant,intron_variant',
       'stop_retained_variant',
       'stop_gained,frameshift_variant,splice_region_variant',
       'stop_gained,protein_altering_variant', 'mature_miRNA_variant',
       'intron_variant,NMD_transcript_variant',
       'splice_region_variant,non_coding_transcript_exon_variant',
       'intergenic_variant',
       'splice_acceptor_variant,non_coding_transcript_variant',
       'splice_donor_variant,5_prime_UTR_variant,intron_variant',
       'start_lost,splice_region_variant']

class HypothesisNoveltyAssessor:
    def __init__(self, input_csv_path, output_csv_path, search_delay=(2, 5)):
        """
        Initialize the Hypothesis Novelty Assessor

        Parameters:
        -----------
        input_csv_path : str
            Path to input CSV file containing hypotheses
        output_csv_path : str
            Path to save output CSV with novelty scores
        search_delay : tuple
            Range of random delay between searches (min, max) in seconds
        """
        self.input_csv_path = input_csv_path
        self.output_csv_path = output_csv_path
        self.search_delay = search_delay

    def format_hypothesis_query(self, row):
        """
        Format a hypothesis into a PubMed query

        Parameters:
        -----------
        row : pandas.Series
            Row containing hypothesis data

        Returns:
        --------
        str
            Formatted query for PubMed
        """
        # Extract relevant parts from the hypothesis
        cancer_type = row['cancer_type']

        factor_terms = []
        # add textual features
        for col in TEXT_FEATURES:
            if col in row and pd.notna(row[col]):
                factor_terms.extend(str(row[col]).split('-'))
        # add dummies features
        for col in DUMMIE_FEATURES:
            if col in row and pd.notna(row[col]) and row[col] == 1:
                factor_terms.append(str(col).replace('_', ' '))

        to_remove = ['Intragenic', 'ETV62']
        # remove unwanted terms
        factor_terms = [term for term in factor_terms if term not in to_remove]
        # remove duplicates
        factor_terms = ' '.join(sorted(set(factor_terms), key=lambda x: factor_terms.index(x)))

        # Combine with cancer type
        query = f"{factor_terms} {cancer_type}"
        return query.strip()

    def search_pubmed(self, query):
        """
        Search PubMed for related papers

        Parameters:
        -----------
        query : str
            Query string for PubMed

        Returns:
        --------
        list
            List of dictionaries containing paper information
        """
        logger.info(f"Searching PubMed for: {query}")
        article_count = 0
        try:
            # Construct the PubMed API URL
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query.replace(' ', '+')}&retmode=json"

            # Send the request
            response = requests.get(url)

            # Parse the response as JSON
            data = response.json()

            # Extract the total number of results
            article_count = data["esearchresult"]["count"]

        except Exception as e:
            logger.error(f"Error in PubMed search: {str(e)}")

        # Add random delay to avoid being blocked
        # time.sleep(random.uniform(self.search_delay[0], self.search_delay[1]))

        return article_count

    def calculate_novelty_score(self, papers_found, base=10):
        """
        Convert similarity score to novelty score

        Parameters:
        -----------
        papers_found : int
            Number of papers found for this hypothesis

        Returns:
        --------
        int
            Novelty score from 1-10 (higher means more novel) or -1 if no papers found
        """
        # If no papers found, return -1
        if papers_found == 0:
            return -1

        count_score =  max(1, min(10, 1 + 9 * math.log1p(papers_found) / math.log1p(base * 1000)))
        novelty = 10 - count_score

        # Ensure score is between 1-10
        return max(1, min(10, round(novelty)))

    def assess_novelty(self):
        """
        Main method to assess novelty of all hypotheses
        """
        # Read input CSV
        logger.info(f"Reading hypotheses from {self.input_csv_path}")
        df = pd.read_csv(self.input_csv_path)

        results = []

        # Process each hypothesis
        for idx, row in df.iterrows():
            if row['method'] == 'RULE':
                # Skip rules-based hypotheses
                continue
            hypo_id = row['hypo_id']
            logger.info(f"Processing hypothesis {hypo_id} ({idx + 1}/{len(df)})")

            # Create search query
            query = self.format_hypothesis_query(row)

            # Search for related papers
            article_count = self.search_pubmed(query)

            if article_count == 0:
                logger.warning(f"No papers found for hypothesis {hypo_id}. Assigning novelty score of -1.")
            else:
                logger.info(f"Found {article_count} papers for hypothesis {hypo_id}")

            results.append({
                'hypo_id': hypo_id,
                'num_papers_found': article_count
            })

        # Create results DataFrame
        results_df = pd.DataFrame(results)
        results_df['num_papers_found'] = results_df['num_papers_found'].astype(int)

        # Calculate novelty scores with Logarithmic Scaling
        results_df['novelty_score'] = results_df['num_papers_found'].apply(lambda x: self.calculate_novelty_score(x))

        # Save to CSV
        results_df.to_csv(self.output_csv_path, index=False)
        logger.info(f"Novelty assessment complete. Results saved to {self.output_csv_path}")

        # # Additional detailed output for analysis
        # detailed_output_path = os.path.splitext(self.output_csv_path)[0] + "_detailed.csv"
        # results_df.to_csv(detailed_output_path, index=False)
        # logger.info(f"Detailed results saved to {detailed_output_path}")

        return results_df


# Example usage
if __name__ == "__main__":
    # Example paths
    input_csv = "models_hypotheses/combined_hypotheses.csv"

    # Create and run the assessor
    assessor = HypothesisNoveltyAssessor(input_csv, Config.CSV_PATH)
    results = assessor.assess_novelty()

    print(f"Summary of novelty scores:")
    print(results['novelty_score'].value_counts().sort_index())