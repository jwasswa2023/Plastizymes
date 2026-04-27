# =========================================================
# UNCERTAINTY QUANTIFICATION (REPO READY)
# =========================================================

import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef,
    brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve

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

N_ENSEMBLE = 10
BOOTSTRAP = True
EPS = 1e-12
N_BINS = 10


# =========================================================
# SEED
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

exclude_cols = {
    TARGET_COL,
    "sequence",
    "repeat_unit_smiles",
    "polymer_name",
    "polymer_family"
}

feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols]
y = df[TARGET_COL].values


# =========================================================
# MODEL + PARAMS
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
        ("clf", LGBMClassifier(
            objective="binary",
            random_state=RANDOM_STATE,
            verbose=-1
        ))
    ])


# =========================================================
# UNCERTAINTY FUNCTIONS
# =========================================================
def predictive_entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p*np.log(p) + (1-p)*np.log(1-p))


def bootstrap_sample(X, y, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=len(y), replace=True)
    return X.iloc[idx], y[idx]


def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "BalancedAcc": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Brier": brier_score_loss(y_true, y_prob),
        "NLL": log_loss(y_true, np.clip(y_prob, EPS, 1 - EPS))
    }


# =========================================================
# OUTER CV
# =========================================================
outer_cv = StratifiedKFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

results = []
all_y_true = []
all_y_prob = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):

    print(f"\nFold {fold}")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    search = RandomizedSearchCV(
        make_pipeline(feature_cols),
        param_distributions=param_dist,
        n_iter=20,
        scoring="roc_auc",
        cv=N_INNER_SPLITS,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    ensemble_probs = []

    for i in range(N_ENSEMBLE):
        model = clone(best_model)

        if BOOTSTRAP:
            Xb, yb = bootstrap_sample(X_train, y_train, seed=RANDOM_STATE+i)
            model.fit(Xb, yb)
        else:
            model.fit(X_train, y_train)

        ensemble_probs.append(model.predict_proba(X_test)[:, 1])

    ensemble_probs = np.vstack(ensemble_probs)

    mean_prob = ensemble_probs.mean(axis=0)
    var_prob = ensemble_probs.var(axis=0)
    ent_prob = predictive_entropy(mean_prob)

    metrics = compute_metrics(y_test, mean_prob)

    results.append({
        "fold": fold,
        **metrics,
        "MeanVariance": var_prob.mean(),
        "MeanEntropy": ent_prob.mean()
    })

    all_y_true.extend(y_test)
    all_y_prob.extend(mean_prob)


# =========================================================
# SAVE
# =========================================================
results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUTPUT_DIR}uq_metrics.csv", index=False)

print("\nSaved results to results/uq_metrics.csv")


# =========================================================
# RELIABILITY DIAGRAM
# =========================================================
all_y_true = np.array(all_y_true)
all_y_prob = np.array(all_y_prob)

frac_pos, mean_pred = calibration_curve(all_y_true, all_y_prob, n_bins=N_BINS)

plt.figure(figsize=(6,6))
plt.plot([0,1],[0,1],'--')
plt.plot(mean_pred, frac_pos, marker='o')
plt.xlabel("Predicted")
plt.ylabel("Observed")
plt.title("Reliability Diagram")
plt.savefig(f"{OUTPUT_DIR}reliability.png")
plt.show()
