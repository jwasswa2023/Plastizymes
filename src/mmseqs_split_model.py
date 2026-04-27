# =========================================================
# FINAL MULTIMODAL MODEL — CODE A
# MMseqs2-based sequence-similarity split
#
# Modalities:
# - stronger handcrafted sequence descriptors
# - ProtBERT + ESM-2 concatenated sequence embeddings
# - polymer chemistry descriptors
# - SMILES fingerprint
# - fixed untrained standard GAT embeddings
#
# Models:
# - LightGBM
# - XGBoost
#
# 5-fold outer CV + 3-fold inner randomized search
# MMseqs2 cluster-aware split for both outer and inner CV
# GAT is NOT trained and NOT tuned
# =========================================================

# =========================================================
# INSTALLS
# =========================================================
!pip install -q biopython lightgbm xgboost rdkit-pypi
!python -m pip install -U rdkit
!pip install -q torch torchvision torchaudio
!pip install -q torch-geometric

# Install MMseqs2 official static binary
!wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
!tar -xzf mmseqs-linux-avx2.tar.gz
!cp mmseqs/bin/mmseqs /usr/local/bin/mmseqs
!chmod +x /usr/local/bin/mmseqs
!mmseqs version

# =========================================================
# IMPORTS
# =========================================================
import os
import random
import shutil
import subprocess
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski, AllChem

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
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
# USER SETTINGS
# =========================================================
CSV_PATH = "/content/plastizyme_protbert_esm2_model_input.csv"
SEQUENCE_COL = "sequence"
TARGET_COL = "degradation_label"
POLYMER_SMILES_COL = "repeat_unit_smiles"

RANDOM_STATE = 42
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 3

MMSEQS_MIN_SEQ_ID = 0.40
MMSEQS_COVERAGE = 0.80
MMSEQS_COV_MODE = 0
MMSEQS_CLUSTER_MODE = 0
MMSEQS_SENSITIVITY = 7.5
MMSEQS_TMP_DIR = "/content/mmseqs_tmp_final_multimodal"

NTERM_K = 10
CTERM_K = 10

FINGERPRINT_RADIUS = 2
FINGERPRINT_NBITS = 1024

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GAT_BATCH_SIZE = 64
GAT_HIDDEN_DIM = 64
GAT_NUM_LAYERS = 3
GAT_NUM_HEADS = 4
GAT_DROPOUT = 0.3
GAT_ATTN_DROPOUT = 0.0
GAT_CONCAT_HEADS = True
GAT_POOL = "mean"
GAT_RESIDUAL = True
GAT_USE_BATCHNORM = True

N_ITER_SEARCH = 60

# =========================================================
# REDUCED HYPERPARAMETER SPACES
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

# =========================================================
# REPRODUCIBILITY
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(RANDOM_STATE)

# =========================================================
# CONSTANTS
# =========================================================
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)
DIPEPTIDES = [a + b for a in AA_ORDER for b in AA_ORDER]

AA_GROUPS = {
    "aliphatic": set("GAVLMI"),
    "aromatic": set("FYW"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "uncharged": set("STCPNQ")
}
GROUP_NAMES = list(AA_GROUPS.keys())
GROUP_DIPEPTIDES = [g1 + "_" + g2 for g1 in GROUP_NAMES for g2 in GROUP_NAMES]

CTD_PROPERTIES = {
    "hydrophobicity": [set("RKEDQN"), set("GASTPHY"), set("CLVIMFW")],
    "normwaalsvolume": [set("GASTPDC"), set("NVEQIL"), set("MHKFRYW")],
    "polarity": [set("LIFWCMVY"), set("PATGS"), set("HQRKNED")],
    "polarizability": [set("GASDT"), set("CPNVEQIL"), set("KMHFRYW")],
    "charge": [set("KR"), set("ANCQGHILMFPSTWYV"), set("DE")],
    "secondarystruct": [set("EALMQKRH"), set("VIYCWFT"), set("GNPSD")],
    "solventaccess": [set("ALFCGIVW"), set("RKQEND"), set("MPSTHY")]
}

# =========================================================
# HELPERS
# =========================================================
def clean_seq(seq):
    seq = str(seq).upper().strip()
    seq = "".join([aa if aa in AA_SET else "A" for aa in seq])
    return seq if len(seq) > 0 else "A"

def shannon_entropy(seq):
    n = len(seq)
    if n == 0:
        return 0.0
    counts = Counter(seq)
    probs = np.array([v / n for v in counts.values()], dtype=float)
    return float(-(probs * np.log2(probs + 1e-12)).sum())

def longest_homopolymer_run(seq):
    if len(seq) == 0:
        return 0
    best, cur = 1, 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best

def aa_composition(seq):
    n = len(seq)
    return {f"aa_{aa}": seq.count(aa)/n if n > 0 else 0.0 for aa in AA_ORDER}

def dipeptide_comp(seq):
    total = max(len(seq)-1, 1)
    counts = dict.fromkeys(DIPEPTIDES, 0)
    for i in range(len(seq)-1):
        dp = seq[i:i+2]
        if dp in counts:
            counts[dp] += 1
    return {f"dp_{dp}": counts[dp]/total for dp in DIPEPTIDES}

def residue_to_group(aa):
    for g, aset in AA_GROUPS.items():
        if aa in aset:
            return g
    return "uncharged"

def grouped_aa_comp(seq):
    n = len(seq)
    out = {}
    for g, aset in AA_GROUPS.items():
        out[f"gaa_{g}"] = sum(seq.count(a) for a in aset) / n if n > 0 else 0.0
    return out

def grouped_dipeptide_comp(seq):
    total = max(len(seq)-1, 1)
    counts = dict.fromkeys(GROUP_DIPEPTIDES, 0)
    for i in range(len(seq)-1):
        g1 = residue_to_group(seq[i])
        g2 = residue_to_group(seq[i+1])
        counts[f"{g1}_{g2}"] += 1
    return {f"gdp_{k}": v/total for k, v in counts.items()}

def grouped_k_spaced_pairs(seq, k=1):
    total = max(len(seq)-k-1, 1)
    counts = dict.fromkeys(GROUP_DIPEPTIDES, 0)
    for i in range(len(seq)-k-1):
        g1 = residue_to_group(seq[i])
        g2 = residue_to_group(seq[i+k+1])
        counts[f"{g1}_{g2}"] += 1
    return {f"gsp{k}_{kk}": vv/total for kk, vv in counts.items()}

def terminal_composition(seq, k=10):
    nterm = seq[:k]
    cterm = seq[-k:]
    out = {}
    nk = max(len(nterm), 1)
    ck = max(len(cterm), 1)
    for aa in AA_ORDER:
        out[f"nterm_{k}_{aa}"] = nterm.count(aa) / nk
        out[f"cterm_{k}_{aa}"] = cterm.count(aa) / ck
    return out

def complexity_features(seq):
    n = len(seq)
    uniq = len(set(seq))
    counts = Counter(seq)
    most_common_frac = counts.most_common(1)[0][1] / n if n > 0 else 0.0
    small_res = sum(seq.count(a) for a in "AGSTP") / n if n > 0 else 0.0
    disorder_like = sum(seq.count(a) for a in "AGQSEKPR") / n if n > 0 else 0.0
    sulfur = sum(seq.count(a) for a in "CM") / n if n > 0 else 0.0
    amide = sum(seq.count(a) for a in "NQ") / n if n > 0 else 0.0
    return {
        "seq_entropy": shannon_entropy(seq),
        "unique_residue_fraction": uniq / 20.0,
        "most_common_residue_fraction": most_common_frac,
        "longest_homopolymer_run": longest_homopolymer_run(seq),
        "small_residue_fraction": small_res,
        "disorder_like_fraction": disorder_like,
        "sulfur_fraction": sulfur,
        "amide_fraction": amide
    }

def physicochem(seq):
    p = ProteinAnalysis(seq)
    aa = p.get_amino_acids_percent()
    helix, turn, sheet = p.secondary_structure_fraction()
    return {
        "length": len(seq),
        "log_length": np.log1p(len(seq)),
        "aromaticity": p.aromaticity(),
        "instability": p.instability_index(),
        "pI": p.isoelectric_point(),
        "gravy": p.gravy(),
        "mw": p.molecular_weight(),
        "charge_pH7": p.charge_at_pH(7.0),
        "charge_density_pH7": p.charge_at_pH(7.0) / max(len(seq), 1),
        "helix": helix,
        "turn": turn,
        "sheet": sheet,
        "hydrophobic_fraction": sum(aa.get(x, 0) for x in ["A","I","L","M","F","W","V","Y"]),
        "positive_fraction": sum(aa.get(x, 0) for x in ["K","R","H"]),
        "negative_fraction": sum(aa.get(x, 0) for x in ["D","E"]),
        "polar_fraction": sum(aa.get(x, 0) for x in ["N","C","Q","S","T","Y"]),
        "tiny_fraction": sum(aa.get(x, 0) for x in ["A","C","G","S","T"]),
        "branched_fraction": sum(aa.get(x, 0) for x in ["I","L","V"]),
        "proline_fraction": aa.get("P", 0),
        "glycine_fraction": aa.get("G", 0)
    }

def ctd_features(seq):
    out = {}
    n = len(seq)
    for prop_name, groups in CTD_PROPERTIES.items():
        labels = []
        for aa in seq:
            if aa in groups[0]:
                labels.append(1)
            elif aa in groups[1]:
                labels.append(2)
            else:
                labels.append(3)

        for c in [1, 2, 3]:
            out[f"ctd_{prop_name}_C{c}"] = labels.count(c) / n if n > 0 else 0.0

        total_trans = max(len(labels) - 1, 1)
        t12 = t13 = t23 = 0
        for i in range(len(labels) - 1):
            pair = {labels[i], labels[i + 1]}
            if pair == {1, 2}:
                t12 += 1
            elif pair == {1, 3}:
                t13 += 1
            elif pair == {2, 3}:
                t23 += 1

        out[f"ctd_{prop_name}_T12"] = t12 / total_trans
        out[f"ctd_{prop_name}_T13"] = t13 / total_trans
        out[f"ctd_{prop_name}_T23"] = t23 / total_trans

        for c in [1, 2, 3]:
            pos = [i + 1 for i, lab in enumerate(labels) if lab == c]
            if len(pos) == 0:
                vals = [0, 0, 0, 0, 0]
            else:
                vals = [np.percentile(pos, p) / n for p in [0, 25, 50, 75, 100]]
            for idx, v in enumerate(vals, start=1):
                out[f"ctd_{prop_name}_D{c}_{idx}"] = float(v)
    return out

def extract_features(seq):
    seq = clean_seq(seq)
    feat = {}
    feat.update(physicochem(seq))
    feat.update(complexity_features(seq))
    feat.update(aa_composition(seq))
    feat.update(dipeptide_comp(seq))
    feat.update(grouped_aa_comp(seq))
    feat.update(grouped_dipeptide_comp(seq))
    feat.update(grouped_k_spaced_pairs(seq, k=1))
    feat.update(grouped_k_spaced_pairs(seq, k=2))
    feat.update(terminal_composition(seq, k=NTERM_K))
    feat.update(ctd_features(seq))
    return feat

def compute_polymer_descriptor_dict(smiles):
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return {
            "poly_molwt": np.nan, "poly_exactmolwt": np.nan, "poly_logp": np.nan,
            "poly_tpsa": np.nan, "poly_hbd": np.nan, "poly_hba": np.nan,
            "poly_rotbonds": np.nan, "poly_rings": np.nan, "poly_aromatic_rings": np.nan,
            "poly_fraction_csp3": np.nan, "poly_heavy_atoms": np.nan, "poly_hetero_atoms": np.nan,
            "poly_valence_electrons": np.nan, "poly_mr": np.nan, "poly_nhohcount": np.nan,
            "poly_no_count": np.nan, "poly_aliphatic_rings": np.nan, "poly_saturated_rings": np.nan,
            "poly_aromatic_atoms_fraction": np.nan, "poly_hetero_atoms_fraction": np.nan
        }

    heavy_atoms = mol.GetNumHeavyAtoms()
    aromatic_atoms = sum(atom.GetIsAromatic() for atom in mol.GetAtoms())
    hetero_atoms = sum(atom.GetAtomicNum() not in [1, 6] for atom in mol.GetAtoms())

    return {
        "poly_molwt": Descriptors.MolWt(mol),
        "poly_exactmolwt": Descriptors.ExactMolWt(mol),
        "poly_logp": Crippen.MolLogP(mol),
        "poly_tpsa": rdMolDescriptors.CalcTPSA(mol),
        "poly_hbd": Lipinski.NumHDonors(mol),
        "poly_hba": Lipinski.NumHAcceptors(mol),
        "poly_rotbonds": Lipinski.NumRotatableBonds(mol),
        "poly_rings": Lipinski.RingCount(mol),
        "poly_aromatic_rings": Lipinski.NumAromaticRings(mol),
        "poly_fraction_csp3": Lipinski.FractionCSP3(mol),
        "poly_heavy_atoms": heavy_atoms,
        "poly_hetero_atoms": hetero_atoms,
        "poly_valence_electrons": Descriptors.NumValenceElectrons(mol),
        "poly_mr": Crippen.MolMR(mol),
        "poly_nhohcount": Lipinski.NHOHCount(mol),
        "poly_no_count": Lipinski.NOCount(mol),
        "poly_aliphatic_rings": Lipinski.NumAliphaticRings(mol),
        "poly_saturated_rings": Lipinski.NumSaturatedRings(mol),
        "poly_aromatic_atoms_fraction": aromatic_atoms / max(1, mol.GetNumAtoms()),
        "poly_hetero_atoms_fraction": hetero_atoms / max(1, mol.GetNumAtoms())
    }

def compute_fingerprint_array(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(str(smiles).strip())
    arr = np.zeros((n_bits,), dtype=float)
    if mol is None:
        return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def compute_fingerprint_df(smiles_array, radius=2, n_bits=1024):
    fps = [compute_fingerprint_array(s, radius=radius, n_bits=n_bits) for s in smiles_array]
    return pd.DataFrame(fps, columns=[f"fp_{i}" for i in range(n_bits)])

# =========================================================
# GAT UTILITIES
# =========================================================
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
UNK_INDEX = len(AMINO_ACIDS)
NUM_NODE_FEATURES = len(AMINO_ACIDS) + 1

def one_hot_encode_sequence(seq):
    x = np.zeros((len(seq), NUM_NODE_FEATURES), dtype=np.float32)
    for i, aa in enumerate(seq):
        idx = AA_TO_IDX.get(aa, UNK_INDEX)
        x[i, idx] = 1.0
    return x

def build_chain_edge_index(seq_len):
    if seq_len == 1:
        return torch.tensor([[0], [0]], dtype=torch.long)
    src, dst = [], []
    for i in range(seq_len):
        src.append(i)
        dst.append(i)
    for i in range(seq_len - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
    return torch.tensor([src, dst], dtype=torch.long)

def sequence_to_graph(sequence, label=0):
    seq = clean_seq(sequence)
    x = torch.tensor(one_hot_encode_sequence(seq), dtype=torch.float)
    edge_index = build_chain_edge_index(len(seq))
    y = torch.tensor(label, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

def dataframe_to_graphs(df_sub):
    return [sequence_to_graph(row[SEQUENCE_COL], int(row[TARGET_COL])) for _, row in df_sub.iterrows()]

def make_loader(graphs, batch_size, shuffle):
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)

class StandardGATEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim=64,
        num_gat_layers=3,
        num_heads=4,
        dropout=0.3,
        attn_dropout=0.0,
        concat_heads=True,
        pool="mean",
        residual=True,
        use_batchnorm=True,
    ):
        super().__init__()
        self.pool = pool
        self.residual = residual
        self.dropout = dropout
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        current_dim = hidden_dim

        for _ in range(num_gat_layers):
            conv = GATConv(
                in_channels=current_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                concat=concat_heads,
                dropout=attn_dropout,
                add_self_loops=False,
            )
            self.convs.append(conv)
            next_dim = hidden_dim * num_heads if concat_heads else hidden_dim
            self.bns.append(nn.BatchNorm1d(next_dim) if use_batchnorm else nn.Identity())
            current_dim = next_dim

        self.graph_dim = current_dim

    def encode_graph(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_proj(x)
        for conv, bn in zip(self.convs, self.bns):
            x_in = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=False)
            if self.residual and x.shape == x_in.shape:
                x = x + x_in
        if self.pool == "mean":
            return global_mean_pool(x, batch)
        return global_max_pool(x, batch)

@torch.no_grad()
def extract_gat_embeddings(model, loader, device):
    model.eval()
    all_emb = []
    for batch in loader:
        batch = batch.to(device)
        all_emb.append(model.encode_graph(batch).cpu().numpy())
    return np.vstack(all_emb)

def build_untrained_gat_embeddings(train_df, test_df, fold_seed):
    torch.manual_seed(fold_seed)
    torch.cuda.manual_seed_all(fold_seed)

    model = StandardGATEncoder(
        in_dim=NUM_NODE_FEATURES,
        hidden_dim=GAT_HIDDEN_DIM,
        num_gat_layers=GAT_NUM_LAYERS,
        num_heads=GAT_NUM_HEADS,
        dropout=GAT_DROPOUT,
        attn_dropout=GAT_ATTN_DROPOUT,
        concat_heads=GAT_CONCAT_HEADS,
        pool=GAT_POOL,
        residual=GAT_RESIDUAL,
        use_batchnorm=GAT_USE_BATCHNORM,
    ).to(DEVICE)

    train_loader = make_loader(dataframe_to_graphs(train_df), batch_size=GAT_BATCH_SIZE, shuffle=False)
    test_loader = make_loader(dataframe_to_graphs(test_df), batch_size=GAT_BATCH_SIZE, shuffle=False)

    gat_emb_train = extract_gat_embeddings(model, train_loader, DEVICE)
    gat_emb_test = extract_gat_embeddings(model, test_loader, DEVICE)
    return gat_emb_train, gat_emb_test

# =========================================================
# MMSEQS2 CLUSTERING
# =========================================================
def build_mmseqs2_groups(df, seq_col, work_dir, min_seq_id=0.4, coverage=0.8, cov_mode=0, cluster_mode=0, sensitivity=7.5):
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    fasta_path = os.path.join(work_dir, "seqs.fasta")
    db_path = os.path.join(work_dir, "seqdb")
    clu_path = os.path.join(work_dir, "clu")
    tsv_path = os.path.join(work_dir, "clusters.tsv")
    tmp_dir = os.path.join(work_dir, "tmp")

    with open(fasta_path, "w") as f:
        for i, seq in enumerate(df[seq_col].astype(str).tolist()):
            f.write(f">seq_{i}\n{clean_seq(seq)}\n")

    subprocess.run(["mmseqs", "createdb", fasta_path, db_path], check=True)
    subprocess.run([
        "mmseqs", "cluster",
        db_path, clu_path, tmp_dir,
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", str(cov_mode),
        "--cluster-mode", str(cluster_mode),
        "-s", str(sensitivity)
    ], check=True)
    subprocess.run(["mmseqs", "createtsv", db_path, db_path, clu_path, tsv_path], check=True)

    clu_df = pd.read_csv(tsv_path, sep="\t", header=None, names=["rep", "member"])
    member_to_rep = dict(zip(clu_df["member"], clu_df["rep"]))

    groups = []
    for i in range(len(df)):
        member_id = f"seq_{i}"
        groups.append(member_to_rep.get(member_id, member_id))

    return np.array(groups)

# =========================================================
# FEATURE TABLES
# =========================================================
def build_feature_table(index, handcrafted_df, seq_embeddings_df, polymer_df, fingerprint_df, gat_embeddings):
    desc_df = handcrafted_df.copy()
    desc_df.index = index

    emb_df = seq_embeddings_df.copy()
    emb_df.index = index
    emb_cols = list(emb_df.columns)

    poly_df = polymer_df.copy()
    poly_df.index = index

    fp_df = fingerprint_df.copy()
    fp_df.index = index

    gat_cols = [f"gat_emb_{i}" for i in range(gat_embeddings.shape[1])]
    gat_df = pd.DataFrame(gat_embeddings, columns=gat_cols, index=index)

    X_df = pd.concat([desc_df, emb_df, poly_df, fp_df, gat_df], axis=1)

    desc_cols = list(desc_df.columns)
    poly_cols = list(poly_df.columns)
    fp_cols = list(fp_df.columns)

    return X_df, desc_cols, emb_cols, poly_cols, fp_cols, gat_cols

def make_preprocessor(desc_cols, emb_cols, poly_cols, fp_cols, gat_cols):
    num_cols = list(desc_cols) + list(emb_cols) + list(poly_cols) + list(fp_cols) + list(gat_cols)
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols)
        ],
        remainder="drop"
    )

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "BalancedAcc": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(CSV_PATH).copy()

required_cols = [SEQUENCE_COL, TARGET_COL, POLYMER_SMILES_COL]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in CSV: {missing_cols}")

df = df.dropna(subset=[SEQUENCE_COL, TARGET_COL, POLYMER_SMILES_COL]).copy()
df[SEQUENCE_COL] = df[SEQUENCE_COL].astype(str)
df[POLYMER_SMILES_COL] = df[POLYMER_SMILES_COL].astype(str).str.strip()
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df = df.dropna(subset=[TARGET_COL]).copy()
df[TARGET_COL] = df[TARGET_COL].astype(int)

seq_embed_cols = [c for c in df.columns if c.startswith("seq_embed_")]
if len(seq_embed_cols) == 0:
    raise ValueError("No concatenated sequence embedding columns found. Expected columns starting with 'seq_embed_'.")

unique_targets = sorted(df[TARGET_COL].unique().tolist())
if set(unique_targets) != {0, 1}:
    if len(unique_targets) != 2:
        raise ValueError(f"TARGET_COL must be binary. Found values: {unique_targets}")
    mapping = {unique_targets[0]: 0, unique_targets[1]: 1}
    df[TARGET_COL] = df[TARGET_COL].map(mapping).astype(int)

y = df[TARGET_COL].values

print("Samples:", len(df))
print("Positives:", int((y == 1).sum()), "| Negatives:", int((y == 0).sum()))
print("Building MMseqs2 clusters...")

groups = build_mmseqs2_groups(
    df=df,
    seq_col=SEQUENCE_COL,
    work_dir=MMSEQS_TMP_DIR,
    min_seq_id=MMSEQS_MIN_SEQ_ID,
    coverage=MMSEQS_COVERAGE,
    cov_mode=MMSEQS_COV_MODE,
    cluster_mode=MMSEQS_CLUSTER_MODE,
    sensitivity=MMSEQS_SENSITIVITY
)

print("Number of clusters:", len(pd.Series(groups).unique()))

# =========================================================
# MODEL PIPES
# =========================================================
lgbm_pipe = Pipeline([
    ("prep", "passthrough"),
    ("clf", LGBMClassifier(
        objective="binary",
        random_state=RANDOM_STATE,
        verbose=-1
    ))
])

xgb_pipe = Pipeline([
    ("prep", "passthrough"),
    ("clf", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        tree_method="hist"
    ))
])

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
# OUTER CV
# =========================================================
outer_cv = StratifiedGroupKFold(
    n_splits=N_OUTER_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

all_results = []

for model_name, model_info in model_dict.items():
    print("\n" + "=" * 100)
    print(f"RUNNING MODEL: {model_name}")
    print("=" * 100)

    pipe = model_info["pipe"]
    param_dist = model_info["param_dist"]

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(df, y, groups=groups), start=1):
        print("\n" + "-" * 90)
        print(f"{model_name} | OUTER FOLD {fold}/{N_OUTER_SPLITS}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        train_groups = groups[train_idx]

        seq_train = train_df[SEQUENCE_COL].values
        seq_test = test_df[SEQUENCE_COL].values
        poly_train = train_df[POLYMER_SMILES_COL].values
        poly_test = test_df[POLYMER_SMILES_COL].values
        y_train = train_df[TARGET_COL].values
        y_test = test_df[TARGET_COL].values

        print("  Computing stronger handcrafted sequence descriptors...")
        X_desc_train = pd.DataFrame([extract_features(seq) for seq in seq_train])
        X_desc_test = pd.DataFrame([extract_features(seq) for seq in seq_test])

        print("  Getting ProtBERT + ESM-2 concatenated embeddings...")
        seq_emb_train_df = train_df[seq_embed_cols].copy().reset_index(drop=True)
        seq_emb_test_df = test_df[seq_embed_cols].copy().reset_index(drop=True)

        print("  Computing polymer chemistry descriptors...")
        X_poly_train = pd.DataFrame([compute_polymer_descriptor_dict(s) for s in poly_train])
        X_poly_test = pd.DataFrame([compute_polymer_descriptor_dict(s) for s in poly_test])

        print("  Computing fingerprints...")
        X_fp_train = compute_fingerprint_df(poly_train, radius=FINGERPRINT_RADIUS, n_bits=FINGERPRINT_NBITS)
        X_fp_test = compute_fingerprint_df(poly_test, radius=FINGERPRINT_RADIUS, n_bits=FINGERPRINT_NBITS)

        print("  Extracting fixed untrained GAT embeddings...")
        gat_emb_train, gat_emb_test = build_untrained_gat_embeddings(
            train_df=train_df,
            test_df=test_df,
            fold_seed=RANDOM_STATE + fold
        )

        X_train_df, desc_cols, emb_cols, poly_cols, fp_cols, gat_cols = build_feature_table(
            index=np.arange(len(train_df)),
            handcrafted_df=X_desc_train,
            seq_embeddings_df=seq_emb_train_df,
            polymer_df=X_poly_train,
            fingerprint_df=X_fp_train,
            gat_embeddings=gat_emb_train
        )

        X_test_df, _, _, _, _, _ = build_feature_table(
            index=np.arange(len(test_df)),
            handcrafted_df=X_desc_test,
            seq_embeddings_df=seq_emb_test_df,
            polymer_df=X_poly_test,
            fingerprint_df=X_fp_test,
            gat_embeddings=gat_emb_test
        )

        preprocessor = make_preprocessor(desc_cols, emb_cols, poly_cols, fp_cols, gat_cols)

        inner_cv = StratifiedGroupKFold(
            n_splits=N_INNER_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + fold
        )

        model_pipe = clone(pipe)
        model_pipe.set_params(prep=preprocessor)

        search = RandomizedSearchCV(
            estimator=model_pipe,
            param_distributions=param_dist,
            n_iter=N_ITER_SEARCH,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
            verbose=1,
            random_state=RANDOM_STATE + fold
        )

        print(f"  Running randomized search ({N_ITER_SEARCH} iterations)...")
        search.fit(X_train_df, y_train, groups=train_groups)

        best_model = search.best_estimator_
        y_prob = best_model.predict_proba(X_test_df)[:, 1]
        metrics = compute_metrics(y_test, y_prob)

        row = {
            "model": model_name,
            "fold": fold,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_clusters_train": len(pd.Series(train_groups).unique()),
            "n_clusters_test": len(pd.Series(groups[test_idx]).unique()),
            "n_features": X_train_df.shape[1],
            "best_cv_AUROC": search.best_score_,
            "best_params": str(search.best_params_),
            **metrics
        }
        all_results.append(row)

        print(
            f"{model_name} | Fold {fold} | "
            f"AUROC={metrics['AUROC']:.3f} | "
            f"AUPRC={metrics['AUPRC']:.3f} | "
            f"F1={metrics['F1']:.3f} | "
            f"BalAcc={metrics['BalancedAcc']:.3f} | "
            f"MCC={metrics['MCC']:.3f}"
        )

# =========================================================
# RESULTS SUMMARY
# =========================================================
results_df = pd.DataFrame(all_results)
metrics = ["AUROC", "AUPRC", "F1", "BalancedAcc", "MCC"]

summary_rows = []
for model_name in results_df["model"].unique():
    sub = results_df[results_df["model"] == model_name].copy()
    row = {"model": model_name}
    for metric in metrics:
        row[metric] = f"{sub[metric].mean():.3f} ± {sub[metric].std(ddof=1):.3f}"
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

print("\n" + "=" * 100)
print("PER-FOLD RESULTS")
print(results_df)

print("\n" + "=" * 100)
print("SUMMARY")
print(summary_df)

results_df.to_csv("/content/final_all_modalities_mmseqs2_codeA_fold_results.csv", index=False)
summary_df.to_csv("/content/final_all_modalities_mmseqs2_codeA_summary.csv", index=False)

print("\nSaved files:")
print(" - /content/final_all_modalities_mmseqs2_codeA_fold_results.csv")
print(" - /content/final_all_modalities_mmseqs2_codeA_summary.csv")
