import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


# =========================================================
# SETTINGS
# =========================================================
MODEL_NAME = "Rostlab/prot_bert"
INPUT_CSV = "data/processed/plastizyme_cleaned_dataset.csv"
OUTPUT_CSV = "data/processed/plastizyme_with_protbert.csv"

SEQUENCE_COLUMN = "sequence"
BATCH_SIZE = 4
MAX_LENGTH = 1024

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(INPUT_CSV)
print("Input shape:", df.shape)


# =========================================================
# LOAD MODEL
# =========================================================
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
model = BertModel.from_pretrained(MODEL_NAME)
model = model.to(DEVICE)
model.eval()


# =========================================================
# CLEAN SEQUENCES
# =========================================================
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

def clean_sequence(seq):
    if pd.isna(seq):
        return None

    seq = str(seq).strip().upper()

    # remove X from ends only
    seq = re.sub(r"^X+|X+$", "", seq)

    if len(seq) == 0:
        return None

    if any(aa not in STANDARD_AA for aa in seq):
        return None

    return seq


def format_for_protbert(seq):
    return " ".join(list(seq))


# =========================================================
# MEAN POOLING
# =========================================================
def mean_pooling(last_hidden_state, attention_mask):
    pooled_embeddings = []

    for i in range(last_hidden_state.shape[0]):
        mask = attention_mask[i].bool()
        token_embeddings = last_hidden_state[i][mask]

        if token_embeddings.shape[0] > 2:
            token_embeddings = token_embeddings[1:-1]
        else:
            pooled_embeddings.append(np.full(last_hidden_state.shape[-1], np.nan))
            continue

        pooled = token_embeddings.mean(dim=0).detach().cpu().numpy()
        pooled_embeddings.append(pooled)

    return np.vstack(pooled_embeddings)


# =========================================================
# PREPARE DATA
# =========================================================
df["clean_sequence"] = df[SEQUENCE_COLUMN].apply(clean_sequence)

valid_df = df[df["clean_sequence"].notna()].copy().reset_index(drop=True)
print("Valid sequences:", len(valid_df))

valid_df["protbert_input"] = valid_df["clean_sequence"].apply(format_for_protbert)


# =========================================================
# GENERATE EMBEDDINGS
# =========================================================
all_embeddings = []

for start_idx in tqdm(range(0, len(valid_df), BATCH_SIZE)):
    batch_text = valid_df["protbert_input"].iloc[start_idx:start_idx + BATCH_SIZE].tolist()

    encoded = tokenizer(
        batch_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )

    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        batch_embeddings = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])

    all_embeddings.append(batch_embeddings)

all_embeddings = np.vstack(all_embeddings)
print("Embedding shape:", all_embeddings.shape)


# =========================================================
# MERGE + SAVE
# =========================================================
embedding_cols = [f"protbert_{i}" for i in range(all_embeddings.shape[1])]
emb_df = pd.DataFrame(all_embeddings, columns=embedding_cols)

final_df = pd.concat(
    [valid_df.drop(columns=["protbert_input"]), emb_df],
    axis=1
)

final_df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)
