# README

A machine learning project that predicts mutual romantic matches using the Columbia University Speed Dating Dataset\.

This project develops a leakage\-safe prediction framework combining participant\-level normalization, relationship\-aware feature engineering, and three from\-scratch machine learning models implemented entirely in NumPy:

- Logistic Regression

- AdaBoost with Decision Stumps

- RBF\-Kernel Support Vector Machine \(SVM\)

The framework addresses participant\-level data leakage through a wave\-level participant\-disjoint split and introduces GPB \(Global Persona Baseline\) normalization to remove individual scoring bias\.

---

## Project Overview

### Task

Given information about two participants and their interaction during a speed\-dating event, predict whether they will form a mutual match \(`match = 1`\)\.

### Dataset

**Columbia University Speed Dating Dataset**

- 8,378 dating interactions

- 195 original attributes

- Approximately 552 unique participants

- Binary target variable: `match`

- Positive class ratio: 16\.9%

After preprocessing:

- 8,156 valid interactions

- 54 engineered features

- Leakage\-safe wave\-level train/test split

- Training samples: 5,707

- Test samples: 2,449

---

## Project Design Philosophy

This repository separates **hyperparameter search** from **final model training**\.

During development, extensive hyperparameter tuning was performed using cross\-validation and model selection procedures\. The best\-performing configurations were then fixed and used to train the final models reported in the project\.

This design ensures:

- Full reproducibility of the model selection process

- Fair training\-time comparison across models

- Fast execution during demonstrations and grading

- Clear separation between experimentation and final deployment

As a result, users can either reproduce the entire tuning pipeline or directly train the final models using the selected hyperparameters\.

---

## Quick Start

### Hyperparameter Search \(Optional\)

The hyperparameter search stage was completed during development and is included for reproducibility\.

For demonstration, grading, or quick reproduction of the reported results, users may safely skip this step and directly run the final training scripts\.

```bash
python3 lr_grid.py
python3 adaboost_grid.py
python3 svm_grid.py
```

### Data Preprocessing

```bash
python3 ML_data.py
```

### Train Final Models

```bash
python3 lr.py
python3 adaboost.py
python3 svm.py
```

### Generate Experimental Results

```bash
python3 experimental_analysis.py
```

### Generate Report Figures

```bash
python3 generate_all_figures.py
```

> Recommended for demonstrations and grading:
> Run only the preprocessing and final training pipeline below\. Hyperparameter search is not required because the optimal configurations have already been selected and fixed\.
> 
> 

### One\-Command Pipeline

```bash
python3 ML_data.py && \
python3 lr.py && \
python3 adaboost.py && \
python3 svm.py && \
python3 experimental_analysis.py && \
python3 generate_all_figures.py
```

---

## Repository Structure

```text
.
├── Speed Dating Data.csv
│
├── ML_data.py
│
├── lr_grid.py
├── lr.py
│
├── adaboost_grid.py
├── adaboost.py
│
├── svm_grid.py
├── svm.py
│
├── experimental_analysis.py
├── generate_all_figures.py
│
├── data_outputs/
├── model_outputs/
├── final_figures/
│
├── experiment_run_log.md
└── README.md
```

---

## Hyperparameter Search vs Final Training

To maintain reproducibility while keeping demonstration time reasonable, each model is provided in two versions\.

### Hyperparameter Search Scripts

```text
lr_grid.py
adaboost_grid.py
svm_grid.py
```

These scripts were used during the development stage to perform extensive hyperparameter optimization\.

Examples include:

- Logistic Regression: grid search with 5\-fold cross\-validation

- AdaBoost: repeated cross\-validation with Top\-K feature selection

- SVM: candidate model search, ensemble selection, and threshold optimization

The tuning procedures can require substantial computation time and therefore are not necessary for routine execution\.

### Final Training Scripts

```text
lr.py
adaboost.py
svm.py
```

These scripts contain the final hyperparameter configurations selected during the tuning stage and train only the final models used in the report\.

Training times reported in the paper are measured using these final training scripts only\.

For demonstration, grading, and quick reproduction, users are encouraged to run the final training scripts directly\.

---

## Methodology

### 1\. Leakage\-Safe Wave\-Level Splitting

Traditional row\-level random splitting causes participant\-level leakage because the same individual appears in multiple dating interactions\.

This project uses a wave\-level participant\-disjoint split:

- No participant overlap between train and test sets

- Prevents information leakage

- Produces more realistic generalization estimates

---

### 2\. GPB \(Global Persona Baseline\) Normalization

Participants use rating scales differently\.

For participant *i*:

```math
z_{i,k} = \frac{x_{i,k} - \mu_i}{\sigma_i}
```

where:

- $x_{i,k}$ = raw rating

- $\mu_i$ = participant mean rating

- $\sigma_i$ = participant rating standard deviation

This removes individual scoring bias while preserving relative preferences\.

---

### 3\. Relationship\-Aware Feature Engineering

Five categories of engineered features are constructed\.

#### Gender Interaction Features

```text
gender × attractiveness
gender × intelligence
gender × sincerity
...
```

#### Reciprocity Features

```text
like × guess_prob_liked
```

#### Pairwise Compatibility Features

```text
attractive_pair
funny_pair
trait_pair
```

#### Evaluation Gap Features

```text
attractive_gap
age_diff
...
```

#### Interest Similarity Features

- Standardized Interest Similarity \(SIS\)

- Cosine Similarity

- Variance\-weighted Euclidean Similarity

---

## Models

### Logistic Regression

Implemented entirely from scratch using NumPy\.

Features:

- Mini\-batch SGD

- Momentum optimization

- L1/L2 regularization

- Class\-balanced weighting

- Threshold optimization

---

### AdaBoost

Implemented from scratch\.

Features:

- Weighted Decision Stumps

- AdaBoost\.SAMME

- Top\-K feature selection

- Class imbalance handling

---

### Support Vector Machine \(SVM\)

Implemented from scratch\.

Features:

- SMO optimization

- RBF kernel

- Platt scaling

- Validation\-based ensemble selection

---

## Experimental Results

|Model|Accuracy|Precision|Recall|F1|AUC|
|---|---|---|---|---|---|
|Logistic Regression|0\.8024|0\.4496|0\.7536|**0\.5632**|**0\.8474**|
|AdaBoost|0\.8085|0\.4568|0\.7029|0\.5538|0\.8410|
|SVM Ensemble|0\.8195|0\.4741|0\.6184|0\.5367|0\.8229|

### Key Findings

- Logistic Regression achieved the best overall performance\.

- Engineered relationship\-level features were more important than model complexity\.

- Participant\-disjoint splitting produced realistic performance estimates\.

- Reciprocal attraction and bilateral compatibility were the strongest predictive signals\.

---

## Top Predictive Features

|Rank|Feature|Correlation|
|---|---|---|
|1|attractive\_pair|0\.3973|
|2|funny\_pair|0\.3891|
|3|like\_x\_shared\_interests\_o|0\.3876|
|4|trait\_pair|0\.3731|
|5|like\_x\_guess\_prob\_liked|0\.3247|
|6|like\_iid\_centered|0\.3164|
|7|like|0\.3047|

Most of the strongest predictors are engineered interaction features rather than raw participant ratings\.

---

## Key Contributions

- Leakage\-safe participant\-disjoint data splitting

- GPB participant\-level normalization

- Relationship\-aware feature engineering

- Three from\-scratch machine learning models

- Unified experimental evaluation framework

- Fully reproducible pipeline

---

## Dependencies

```bash
numpy >= 1.20
pandas >= 1.3
matplotlib >= 3.3
seaborn >= 0.11
```

No scikit\-learn, TensorFlow, or PyTorch is used\.
All machine learning models are implemented from scratch using NumPy\.

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn
```

---

## Reproducibility

To reproduce all experiments:

```bash
python3 ML_data.py
python3 lr.py
python3 adaboost.py
python3 svm.py
python3 experimental_analysis.py
python3 generate_all_figures.py
```

Generated outputs:

```text
data_outputs/
model_outputs/
final_figures/
```

---

## Future Work

Potential future directions include:

- Learning\-to\-Rank \(LTR\) recommendation systems

- Graph Neural Network \(GNN\) based relationship modeling

- Cross\-cultural validation studies

- Modern online dating platform datasets

- Temporal generalization analysis

---

## Acknowledgements

### Dataset

Columbia University Speed Dating Experiment Dataset
\(Fisman et al\., 2006\)

### Course

AI3013 Machine Learning

### Authors

- Linchen Wang

- Feixuan Wang

- Yucheng Li

- Yingyao Li

- Yunzhen Chen

---

**Version:** v2\.1
**Last Updated:** May 2026

