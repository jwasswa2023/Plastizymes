import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from tqdm import tqdm
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem


# =========================================================
# SETTINGS
# =========================================================
INPUT_CSV = "data/processed/plastizyme_cleaned_dataset.csv"
OUTPUT_CSV = "data/processed/plastizyme_with_transpolymer2.csv"

POLYMER_COL = "repeat_unit_smiles"

HF_SPACE_ID = "transpolymer/Transpolymer2"
CHEMBERTA_ID = "seyonec/ChemBERTa-zinc-base-v1"

BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)


# =========================================================
# MODEL CLASS
# =========================================================
class TransformerRegressor(nn.Module):
    def __init__(self, feat_dim=2058, embedding_dim=768):
        super().__init__()

        self.feat_proj = nn.Linear(feat_dim, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def extract_embedding(self, chemberta_emb, feat):
        feat_emb = self.feat_proj(feat)
        stacked = torch.stack([chemberta_emb, feat_emb], dim=1)
        encoded = self.encoder(stacked)
        return encoded.mean(dim=1)


# =========================================================
# LOAD MODEL FILES
# =========================================================
model_path = hf_hub_download(
    repo_id=HF_SPACE_ID,
    filename="transformer_model.bin",
    repo_type="space"
)

tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_ID)
chemberta = AutoModel.from_pretrained(CHEMBERTA_ID).to(DEVICE).eval()

model = TransformerRegressor()
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model = model.to(DEVICE).eval()


# =========================================================
# FEATURE FUNCTIONS
# =========================================================
def compute_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    desc = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.RingCount(mol),
        Descriptors.MolMR(mol)
    ]

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros((2048,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return np.concatenate([desc, arr])


def get_chemberta_embedding(smiles_list):
    encoded = tokenizer(
        smiles_list,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = chemberta(**encoded)
        return outputs.last_hidden_state.mean(dim=1)


# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(INPUT_CSV)

if POLYMER_COL not in df.columns:
    raise ValueError(f"{POLYMER_COL} not found in dataset")

smiles_list = df[POLYMER_COL].astype(str).tolist()


# =========================================================
# GENERATE EMBEDDINGS
# =========================================================
embeddings = []

for i in tqdm(range(0, len(smiles_list), BATCH_SIZE)):
    batch = smiles_list[i:i+BATCH_SIZE]

    valid_batch = []
    features = []

    for smi in batch:
        feat = compute_features(smi)
        if feat is not None:
            valid_batch.append(smi)
            features.append(feat)

    if len(valid_batch) == 0:
        continue

    chemberta_emb = get_chemberta_embedding(valid_batch)
    feat_tensor = torch.tensor(np.vstack(features), dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        emb = model.extract_embedding(chemberta_emb, feat_tensor)

    embeddings.append(emb.cpu().numpy())


embeddings = np.vstack(embeddings)
print("Embedding shape:", embeddings.shape)


# =========================================================
# SAVE
# =========================================================
embed_cols = [f"transpolymer2_{i}" for i in range(embeddings.shape[1])]
emb_df = pd.DataFrame(embeddings, columns=embed_cols)

final_df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)

final_df.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV)
