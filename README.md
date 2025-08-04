# Cancer Hypothesis Generation And Ranking Pipeline

## TL;DR
By leveraging interpretable ML models and LLM evaluation, we aim to accelerate hypothesis discovery and reduce the manual bottleneck in biomedical research.

## Abstract

Prioritizing biologically meaningful hypotheses in large-scale cancer genomics remains a major bottleneck, with manual approaches proving slow, biased, and unscalable. We present a novel pipeline that combines supervised learning and large language models (LLMs) to generate and prioritize interpretable hypotheses grounded in cancer biology. Using a multi-cancer somatic mutation dataset from over 10,000 solid tumor patients, we trained a multiclass classifier to predict cancer type from genomic and medical features. Model interpretation via SHAP values enabled the generation of over 1,800 natural language hypotheses describing key mutation-phenotype associations. Three independent LLMs then evaluated and scored each hypothesis based on contextual knowledge from the biomedical literature. Analysis of top-ranked hypotheses revealed multiple possible novel and biologically plausible insights, supported by known oncogenic mechanisms.

## Methodology

### Data and Models
- **Dataset**: Multi-cancer somatic mutation data from 10,000+ Chinese cancer patients
- **Data Source**: Processed from [Nature Communications - Pan-cancer analysis of Chinese cancer patients](https://www.nature.com/articles/s41467-022-31780-9)
- **Cancer Types**: 24 distinct cancer types, 105,000+ individual samples
- **Features**: Genomic markers (Hugo gene symbols, exon numbers, mutation types) and clinical variables (age, smoking status, sex)
- **Models**: XGBoost and LightGBM classifiers for multi-class cancer type prediction

### Hypothesis Generation
- **Interpretability**: SHAP (SHapley Additive exPlanations) values to identify influential features or Lift metric to statistically evaluate
associations between specific feature combinations and cancer type.
- **Hypothesis Formation**: Structured natural language sentences describing mutation-phenotype relationships
- **Scale**: Generated 1,900+ candidate biological hypotheses

### LLM Evaluation
- **Models Used**: OpenAI o3-mini, o4-mini, and Anthropic Claude 3.7 Sonnet
- **Scoring Metrics**: 
  - **Novelty**: How unique or previously unexplored the hypothesis is (1-10 scale)
  - **Plausibility**: How well the hypothesis fits with known biomedical mechanisms (1-10 scale)
- **Consistency**: Average standard deviations of 0.8 for novelty and 0.7 for plausibility across five evaluations

## Key Results

### Model Performance
- **XGBoost**: 0.92 average multiclass ROC-AUC, 53.35% test accuracy
- **LightGBM**: 0.91 average multiclass ROC-AUC, 52.36% test accuracy
- Strong predictive power across 12 retained cancer types

### Hypothesis Evaluation
- 1,900 hypotheses evaluated across three LLMs
- Z-score normalization applied to account for model-specific scoring biases
- Some top-ranked hypotheses revealed biologically plausible yet underexplored associations
- Multiple coherent patterns identified between mutational signatures and clinical attributes

### Key Findings
- Pipeline successfully identifies possibly novel mutation-phenotype associations
- LLM scoring provides consistent and interpretable hypothesis prioritization
- Approach offers scalable alternative to manual hypothesis generation
- Top hypotheses span diverse cancer types with coherent biological rationale

## Impact and Applications

This integrative, interpretable approach enhances the scalability and objectivity of hypothesis generation, offering a powerful tool for:
- Accelerating discovery in cancer genomics
- Identifying potential therapeutic avenues
- Reducing bias in hypothesis prioritization
- Focusing experimental efforts on promising leads

## Research Question

*How can LLMs be leveraged to identify, evaluate and prioritize biological hypotheses in terms of their scientific profile and novelty?*

Our work demonstrates that LLMs can effectively act as scientific evaluators, ranking hypotheses for both novelty and biological relevance, contributing a new automated layer to the scientific discovery pipeline.
