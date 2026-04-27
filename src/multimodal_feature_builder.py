import random
import numpy as np
import pandas as pd
from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool


# =========================================================
# SETTINGS
# =========================================================
INPUT_CSV = "data/processed/plastizyme_protbert_esm2.csv"
OUTPUT_CSV = "data/processed/plastizyme_multimodal_final.csv"

SEQUENCE_COL = "sequence"
TARGET_COL = "degradation_label"
SMILES_COL = "repeat_unit_smiles"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# BASIC CLEAN
# =========================================================
AA_SET = set("ACDEFGHIKLMNPQRSTVWY")

def clean_seq(seq):
    seq = str(seq).upper().strip()
    return "".join([aa if aa in AA_SET else "A" for aa in seq])


# =========================================================
# SIMPLE SEQUENCE FEATURES
# =========================================================
def seq_features(seq):
    p = ProteinAnalysis(seq)

    return {
        "length": len(seq),
        "mw": p.molecular_weight(),
        "aromaticity": p.aromaticity(),
        "instability": p.instability_index(),
        "gravy": p.gravy(),
        "pI": p.isoelectric_point()
    }


# =========================================================
# POLYMER FEATURES
# =========================================================
def polymer_features(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {"poly_mw": np.nan, "poly_logp": np.nan}

    return {
        "poly_mw": Descriptors.MolWt(mol),
        "poly_logp": Descriptors.MolLogP(mol)
    }


# =========================================================
# FINGERPRINT
# =========================================================
def fingerprint(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    arr = np.zeros((256,), dtype=np.float32)

    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
        DataStructs.ConvertToNumpyArray(fp, arr)

    return arr


# =========================================================
# SIMPLE GAT (LIGHT VERSION)
# =========================================================
class GATEncoder(nn.Module):
    def __init__(self, in_dim=21, hidden=32):
        super().__init__()
        self.conv = GATConv(in_dim, hidden, heads=2)
    
    def forward(self, x, edge_index, batch):
        x = self.conv(x, edge_index)
        return global_mean_pool(x, batch)


# =========================================================
# MAIN PIPELINE
# =========================================================
def build_multimodal_dataset():

    df = pd.read_csv(INPUT_CSV)

    # -------------------------
    # SEQUENCE FEATURES
    # -------------------------
    seq_df = pd.DataFrame([
        seq_features(clean_seq(s)) for s in df[SEQUENCE_COL]
    ])

    # -------------------------
    # POLYMER FEATURES
    # -------------------------
    poly_df = pd.DataFrame([
        polymer_features(s) for s in df[SMILES_COL]
    ])

    # -------------------------
    # FINGERPRINTS
    # -------------------------
    fp_array = np.vstack([
        fingerprint(s) for s in df[SMILES_COL]
    ])
    fp_df = pd.DataFrame(fp_array, columns=[f"fp_{i}" for i in range(fp_array.shape[1])])

    # -------------------------
    # FINAL CONCAT
    # -------------------------
    final_df = pd.concat([
        df.reset_index(drop=True),
        seq_df,
        poly_df,
        fp_df
    ], axis=1)

    final_df.to_csv(OUTPUT_CSV, index=False)

    print("Saved:", OUTPUT_CSV)
    print("Final shape:", final_df.shape)


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    build_multimodal_dataset()
