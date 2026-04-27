# Plastizymes: Multimodal Machine Learning for Plastic-Degrading Enzyme Prediction

This repository contains the full machine learning pipeline for the Plastizymes project. The goal is to predict plastic-degrading enzyme activity using multimodal features that combine protein sequence representations, polymer chemistry information, molecular fingerprints, and graph-based sequence embeddings.

The project includes data cleaning, benchmark dataset construction, protein language model embeddings, polymer embeddings, multimodal feature construction, model training, uncertainty quantification, modality contribution analysis, and multiple biologically meaningful validation strategies.

---

## Project Overview

Plastic-degrading enzymes act on chemically diverse polymer substrates. A strong predictive model should therefore represent both:

1. the enzyme sequence, and  
2. the polymer or plastic substrate.

This repository builds a multimodal model using:

- protein sequence embeddings from ProtBERT
- protein sequence embeddings from ESM-2
- Prot2vec Embeddings
- polymer representations from Transpolymer2
- handcrafted sequence descriptors
- polymer chemistry descriptors
- SMILES/Morgan fingerprints
- fixed untrained graph attention network embeddings
- LightGBM and XGBoost meta-learners

The repository also includes evaluation under:

- random split
- MMseqs2 sequence-similarity-aware split
- leave-one-polymer-family-out split
- modality ablation analysis
- SHAP interpretability
- uncertainty quantification and calibration

---

## Repository Structure

```text
Plastizymes/
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data_processing.py
│   ├── protbert_embeddings.py
│   ├── esm2_embeddings.py
│   ├── transpolymer_embeddings.py
│   ├── multimodal_feature_builder.py
│   ├── random_split_model.py
│   ├── mmseqs_split_model.py
│   ├── leave_one_polymer_family_split.py
│   ├── modality_contribution_analysis.py
│   └── uncertainty_analysis.py
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_embedding_generation.ipynb
│   ├── 03_multimodal_features.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_modality_contribution.ipynb
│   └── 06_uncertainty_analysis.ipynb
│
├── results/
│   ├── figures/
│   └── tables/
│
├── models/
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Data

The project expects raw and processed data to be organized as follows:

```text
data/
├── raw/
│   ├── CLEAN_V4.csv
│   ├── Cleaned_Combined_filled.csv
│   └── additional_source_files.csv
│
└── processed/
    ├── plastizyme_cleaned_dataset.csv
    ├── plastizyme_protbert_esm2_model_input.csv
    └── plastizyme_multimodal_final.csv
```

Recommended practice:

- keep original files in `data/raw/`
- save cleaned files in `data/processed/`
- do not overwrite raw data
- save model outputs in `results/`

---

## Installation

Clone the repository:

```bash
git clone https://github.com/jwasswa2023/Plastizymes.git
cd Plastizymes
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Additional Installation Notes

Some scripts require large deep learning or chemistry packages.

For Google Colab, install packages inside the notebook environment as needed:

```bash
pip install transformers torch pandas numpy tqdm sentencepiece
pip install biopython lightgbm xgboost shap rdkit torch-geometric
```

For MMseqs2-based splitting, install MMseqs2:

```bash
wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
tar -xzf mmseqs-linux-avx2.tar.gz
cp mmseqs/bin/mmseqs /usr/local/bin/mmseqs
chmod +x /usr/local/bin/mmseqs
mmseqs version
```

---

## Pipeline Summary

The overall workflow is:

```text
Raw data
   ↓
Data cleaning and deduplication
   ↓
Protein embedding generation
   ├── ProtBERT
   └── ESM-2
   ↓
Polymer representation
   ├── Transpolymer2
   ├── RDKit descriptors
   └── Morgan fingerprints
   ↓
Multimodal feature construction
   ↓
Model training
   ├── Random split
   ├── MMseqs2 split
   └── Leave-one-polymer-family-out split
   ↓
Interpretability
   ├── Modality ablation
   └── SHAP contribution analysis
   ↓
Uncertainty quantification
   ├── ensemble variance
   ├── predictive entropy
   ├── calibration metrics
   └── reliability diagram
```

---

## 1. Data Processing

Script:

```text
src/data_processing.py
```

Purpose:

- load original plastizyme datasets
- standardize column names
- clean enzyme sequences
- normalize polymer names
- remove duplicate polymer-sequence pairs
- save cleaned dataset

Expected output:

```text
data/processed/plastizyme_cleaned_dataset.csv
```

Run:

```bash
python src/data_processing.py
```

---

## 2. ProtBERT Embeddings

Script:

```text
src/protbert_embeddings.py
```

Purpose:

- load cleaned enzyme sequences
- clean and validate amino acid sequences
- generate ProtBERT embeddings
- apply mean pooling across residue tokens
- save sequence-level embedding features

Expected output:

```text
data/processed/plastizyme_with_protbert.csv
```

Run:

```bash
python src/protbert_embeddings.py
```

---

## 3. ESM-2 Embeddings

Script:

```text
src/esm2_embeddings.py
```

Purpose:

- generate ESM-2 protein language model embeddings
- clean invalid sequences
- use mean pooling over valid residue tokens
- save ESM-2 embedding features

Expected output:

```text
data/processed/plastizyme_with_esm2.csv
```

Run:

```bash
python src/esm2_embeddings.py
```

---

## 4. Transpolymer2 Embeddings

Script:

```text
src/transpolymer_embeddings.py
```

Purpose:

- represent polymer repeat units using SMILES strings
- compute ChemBERTa-based polymer embeddings
- compute RDKit descriptors and Morgan fingerprints
- generate Transpolymer2 polymer embeddings

Expected input column:

```text
repeat_unit_smiles
```

Expected output:

```text
data/processed/plastizyme_with_transpolymer2.csv
```

Run:

```bash
python src/transpolymer_embeddings.py
```

---

## 5. Multimodal Feature Construction

Script:

```text
src/multimodal_feature_builder.py
```

Purpose:

Build the final multimodal model input by combining:

- ProtBERT embeddings
- ESM-2 embeddings
- handcrafted sequence descriptors
- polymer chemistry descriptors
- SMILES/Morgan fingerprints
- fixed untrained GAT sequence graph embeddings

Expected output:

```text
data/processed/plastizyme_multimodal_final.csv
```

Run:

```bash
python src/multimodal_feature_builder.py
```

---

## 6. Random Split Modeling

Script:

```text
src/random_split_model.py
```

Purpose:

Train and evaluate multimodal models using random stratified splitting.

Models:

- LightGBM
- XGBoost

Validation:

- 5-fold outer cross-validation
- 3-fold inner randomized hyperparameter search
- 60 randomized hyperparameter iterations

Metrics:

- AUROC
- AUPRC
- F1 score
- balanced accuracy
- Matthews correlation coefficient

Expected outputs:

```text
results/random_split_results.csv
```

Run:

```bash
python src/random_split_model.py
```

---

## 7. MMseqs2 Sequence-Similarity Split

Script:

```text
src/mmseqs_split_model.py
```

Purpose:

Evaluate biological generalization by preventing highly similar sequences from appearing in both training and test sets.

This script uses MMseqs2 to cluster enzyme sequences and then applies group-aware cross-validation.

MMseqs2 settings:

```text
Minimum sequence identity: 0.40
Coverage: 0.80
Sensitivity: 7.5
```

Models:

- LightGBM
- XGBoost

Validation:

- 5-fold outer StratifiedGroupKFold
- 3-fold inner StratifiedGroupKFold
- 60 randomized hyperparameter iterations

Expected outputs:

```text
results/final_all_modalities_mmseqs2_codeA_fold_results.csv
results/final_all_modalities_mmseqs2_codeA_summary.csv
```

Run:

```bash
python src/mmseqs_split_model.py
```

---

## 8. Leave-One-Polymer-Family-Out Split

Script:

```text
src/leave_one_polymer_family_split.py
```

Purpose:

Test how well the model generalizes to unseen polymer families.

Outer split:

- one polymer family is held out at a time

Inner split:

- grouped cross-validation on the remaining polymer families

Required column:

```text
polymer_family
```

Models:

- LightGBM
- XGBoost

Expected outputs:

```text
results/final_all_modalities_leave1polyfamilyout_codeA_fold_results.csv
results/final_all_modalities_leave1polyfamilyout_codeA_summary.csv
```

Run:

```bash
python src/leave_one_polymer_family_split.py
```

---

## 9. Modality Contribution Analysis

Script:

```text
src/modality_contribution_analysis.py
```

Purpose:

Quantify the contribution of each feature modality.

Modalities tested:

1. sequence descriptors
2. ProtBERT + ESM-2 sequence embeddings
3. polymer chemistry descriptors
4. SMILES fingerprint features
5. fixed untrained GAT embeddings

Analysis includes:

- full model
- leave-one-modality-out ablation
- nested hyperparameter tuning
- SHAP feature importance
- modality-level SHAP aggregation

Expected outputs:

```text
results/ablation_fold_results.csv
results/ablation_summary.csv
results/ablation_drop_vs_full.csv
results/shap_feature_values.csv
results/shap_modality_summary.csv
results/top_features_by_modality.csv
results/top15_features_overall.csv
```

Run:

```bash
python src/modality_contribution_analysis.py
```

---

## 10. Uncertainty Quantification

Script:

```text
src/uncertainty_analysis.py
```

Purpose:

Estimate predictive uncertainty and calibration for the final multimodal LightGBM model.

Uncertainty methods:

- bootstrap ensemble predictions
- predictive variance
- predictive entropy
- reliability diagram
- calibration error metrics

Calibration metrics:

- expected calibration error
- maximum calibration error
- Brier score
- negative log likelihood

Expected outputs:

```text
results/uq_metrics.csv
results/reliability.png
```

Run:

```bash
python src/uncertainty_analysis.py
```

---

## Models

This project evaluates:

### LightGBM

Gradient boosting decision tree model used as a primary meta-learner.

### XGBoost

Gradient boosting model used as a comparison meta-learner.

Both models are trained on the final multimodal feature table.

---

## Feature Modalities

### Protein Language Model Embeddings

ProtBERT and ESM-2 embeddings capture contextual protein sequence information learned from large-scale protein databases.

### Handcrafted Sequence Descriptors

These include amino acid composition, dipeptide composition, grouped residue statistics, terminal composition, CTD descriptors, and physicochemical properties.

### Polymer Chemistry Descriptors

RDKit descriptors capture polymer repeat-unit chemistry, including molecular weight, LogP, TPSA, hydrogen bond donors/acceptors, ring counts, aromaticity, and heteroatom composition.

### Molecular Fingerprints

Morgan fingerprints represent polymer repeat-unit substructure patterns.

### GAT Sequence Graph Embeddings

Protein sequences are represented as chain graphs, and fixed untrained GAT embeddings provide graph-structured sequence features.

---

## Evaluation Metrics

The classification models are evaluated using:

- AUROC
- AUPRC
- F1 score
- balanced accuracy
- Matthews correlation coefficient

Uncertainty analysis additionally reports:

- predictive variance
- predictive entropy
- expected calibration error
- maximum calibration error
- Brier score
- negative log likelihood

---

## Reproducibility

Most scripts use:

```text
RANDOM_STATE = 42
```

For full reproducibility, use the same:

- Python version
- package versions
- input data
- random seed
- hardware setting

Deep learning models and GPU operations may still produce small numerical differences across systems.

---

## Notes on Large Files

Generated embedding files can become large. Avoid committing very large CSVs unless needed.

Recommended `.gitignore` entries:

```text
__pycache__/
*.pyc
venv/
.env
data/raw/*
data/processed/*.csv
models/*.pkl
models/*.pt
results/*.csv
```

Keep `.gitkeep` files inside empty folders if you want GitHub to preserve folder structure.

---

## Suggested Citation

If using this repository, cite the associated Plastizymes manuscript when available.

---

## Acknowldgements
AI Use: Artificial intelligence models from OpenAI (ChatGPT), Anthropic (Claude), and Google (Gemini) were used as coding aids in the development of Plastizymes.


## Contact

Joseph Wasswa  
SUNY Polytechnic Institute  
wasswaj@sunypoly.edu
