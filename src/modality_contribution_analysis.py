# =========================================================
# MODALITY CONTRIBUTION ANALYSIS (REPO READY)
# Nested CV + Ablation + SHAP
# =========================================================

import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score, matthews_corrcoef

from lightgbm import LGBMClassifier

# =========================================================
# SETTINGS
# =========================================================
CSV_PATH = "data/processed/plastizyme_protbert_esm2_model_input.csv"
TARGET_COL = "degradation_label"

RANDOM_STATE = 42
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 3

OUTPUT_DIR = "results/"

# =========================================================
# REPRODUCIBILITY
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

seed_everything(RANDOM_STATE)

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(CSV_PATH)

df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df = df.dropna(subset=[TARGET_COL])
df[TARGET_COL] = df[TARGET_COL].astype(int)

# =========================================================
# FEATURE GROUPS
# =========================================================
feature_groups = {
    "seq_desc": [c for c in df.columns if c.startswith("aa_") or c.startswith("dp_")],
    "seq_embed": [c for c in df.columns if c.startswith("protbert_") or c.startswith("esm2_")],
    "polymer": [c for c in df.columns if c.startswith("poly_")],
    "fingerprint": [c for c in df.columns if c.startswith("fp_")],
    "gat": [c for c in df.columns if c.startswith("gat_")]
}

feature_groups = {k: v for k, v in feature_groups.items() if len(v) > 0}
all_features = sum(feature_groups.values(), [])

X = df[all_features]
y = df[TARGET_COL].values

# =========================================================
# MODEL + PARAM GRID
# =========================================================
param_dist = {
    "clf__n_estimators": [200, 500, 800],
    "clf__learning_rate": [0.01, 0.05, 0.1],
    "clf__num_leaves": [15, 31, 63],
    "clf__min_child_samples": [10, 20, 30],
    "clf__subsample": [0.7, 0.85, 1.0]
}

def make_pipeline(features):
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), features)
    ])

    return Pipeline([
        ("prep", preprocessor),
        ("clf", LGBMClassifier(objective="binary", random_state=RANDOM_STATE, verbose=-1))
    ])

# =========================================================
# METRICS
# =========================================================
def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "BalancedAcc": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

# =========================================================
# ABLATION CONFIGS
# =========================================================
configs = {"FULL": all_features}

for group in feature_groups:
    keep = [f for g, feats in feature_groups.items() if g != group for f in feats]
    configs[f"NO_{group.upper()}"] = keep

# =========================================================
# NESTED CV
# =========================================================
outer_cv = StratifiedKFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

results = []
shap_results = []

for config_name, features in configs.items():
    print(f"\nRunning {config_name}")

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):

        X_train = X.iloc[train_idx][features]
        X_test = X.iloc[test_idx][features]
        y_train = y[train_idx]
        y_test = y[test_idx]

        inner_cv = StratifiedKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

        search = RandomizedSearchCV(
            estimator=make_pipeline(features),
            param_distributions=param_dist,
            n_iter=10,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        y_prob = best_model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_prob)

        results.append({
            "config": config_name,
            "fold": fold,
            **metrics
        })

        print(f"{config_name} Fold {fold} AUROC: {metrics['AUROC']:.3f}")

        # SHAP only for FULL
        if config_name == "FULL" and fold == 1:
            X_trans = best_model.named_steps["prep"].transform(X_train)
            model = best_model.named_steps["clf"]

            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_trans[:200])

            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            mean_abs = np.abs(shap_vals).mean(axis=0)

            shap_df = pd.DataFrame({
                "feature": features,
                "importance": mean_abs
            }).sort_values("importance", ascending=False)

# =========================================================
# SAVE RESULTS
# =========================================================
results_df = pd.DataFrame(results)

results_df.to_csv(f"{OUTPUT_DIR}ablation_results.csv", index=False)
shap_df.to_csv(f"{OUTPUT_DIR}shap_results.csv", index=False)

print("\nSaved results in /results/")
