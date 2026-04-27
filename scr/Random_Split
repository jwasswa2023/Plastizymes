import random
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef
)

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


# =========================================================
# SETTINGS
# =========================================================
CSV_PATH = "data/processed/plastizyme_multimodal_final.csv"
TARGET_COL = "degradation_label"

RANDOM_STATE = 42
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 3
N_ITER_SEARCH = 60


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

if TARGET_COL not in df.columns:
    raise ValueError(f"Missing target column: {TARGET_COL}")

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

print("Samples:", len(df))
print("Positives:", int((y == 1).sum()), "| Negatives:", int((y == 0).sum()))
print("Total features:", len(feature_cols))


# =========================================================
# PREPROCESSOR
# =========================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), feature_cols)
    ]
)


# =========================================================
# MODEL PIPELINES
# =========================================================
lgbm_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", LGBMClassifier(
        objective="binary",
        random_state=RANDOM_STATE,
        verbose=-1
    ))
])

xgb_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        tree_method="hist",
        use_label_encoder=False
    ))
])


# =========================================================
# ORIGINAL HYPERPARAMETER SPACES (UNCHANGED)
# =========================================================
LGBM_PARAM_DIST = {
    "clf__n_estimators": [200, 500, 800],
    "clf__learning_rate": [0.01, 0.05, 0.1],
    "clf__num_leaves": [15, 31, 63],
    "clf__min_child_samples": [10, 20, 30],
    "clf__subsample": [0.7, 0.85, 1.0]
}

XGB_PARAM_DIST = {
    "clf__n_estimators": [200, 500, 800],
    "clf__learning_rate": [0.01, 0.05, 0.1],
    "clf__max_depth": [3, 5, 7],
    "clf__min_child_weight": [1, 3, 5],
    "clf__subsample": [0.7, 0.85, 1.0]
}

model_dict = {
    "LightGBM": {
        "pipe": lgbm_pipe,
        "param_dist": LGBM_PARAM_DIST
    },
    "XGBoost": {
        "pipe": xgb_pipe,
        "param_dist": XGB_PARAM_DIST
    }
}


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
# OUTER CV
# =========================================================
outer_cv = StratifiedKFold(
    n_splits=N_OUTER_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

all_results = []

for model_name, model_info in model_dict.items():

    print(f"\n===== MODEL: {model_name} =====")

    pipe = model_info["pipe"]
    param_dist = model_info["param_dist"]

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):

        print(f"\nFold {fold}/{N_OUTER_SPLITS}")

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        search = RandomizedSearchCV(
            estimator=clone(pipe),
            param_distributions=param_dist,
            n_iter=N_ITER_SEARCH,
            scoring="roc_auc",
            cv=N_INNER_SPLITS,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        y_prob = best_model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_prob)

        row = {
            "model": model_name,
            "fold": fold,
            **metrics
        }

        all_results.append(row)

        print(
            f"{model_name} | Fold {fold} | "
            f"AUROC={metrics['AUROC']:.3f} | "
            f"AUPRC={metrics['AUPRC']:.3f}"
        )


# =========================================================
# SAVE RESULTS
# =========================================================
results_df = pd.DataFrame(all_results)

results_df.to_csv("results/random_split_results.csv", index=False)

print("\nSaved results to results/random_split_results.csv")
