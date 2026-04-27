import pandas as pd
import numpy as np
import re


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_text(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x if x != "" else np.nan


def clean_sequence(seq):
    if pd.isna(seq):
        return np.nan
    seq = str(seq).upper()
    seq = re.sub(r"\s+", "", seq)
    seq = re.sub(r"[^A-Z]", "", seq)
    return seq if seq != "" else np.nan


def canonical_polymer_name(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip().upper()
    x = re.sub(r"\s+", " ", x)

    mapping = {
        "POLYETHYLENE TEREPHTHALATE": "PET",
        "PET": "PET",
        "POLYLACTIC ACID": "PLA",
        "PLA": "PLA",
        "PHA": "PHA",
        "PHB": "PHB",
        "PBAT": "PBAT",
        "PBS": "PBS",
        "NYLON": "PA",
        "PA": "PA",
        "NR": "NR",
        "PUR": "PUR",
        "POLYURETHANE": "PUR",
    }

    return mapping.get(x, x)


# ============================================================
# MAIN FUNCTION
# ============================================================

def process_plastizyme_data(file1, file2, output_file):
    """
    Combine, clean, and deduplicate plastizyme datasets
    """

    # LOAD DATA
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # STANDARDIZE COLUMN NAMES
    df1.columns = df1.columns.str.strip().str.lower()
    df2.columns = df2.columns.str.strip().str.lower()

    # ALIGN COLUMNS
    all_columns = sorted(set(df1.columns).union(set(df2.columns)))
    df1 = df1.reindex(columns=all_columns)
    df2 = df2.reindex(columns=all_columns)

    # COMBINE DATA
    combined = pd.concat([df1, df2], ignore_index=True)

    print("Rows before cleaning:", len(combined))

    # CLEAN KEY COLUMNS (only if they exist)
    if "plastic" in combined.columns:
        combined["plastic"] = combined["plastic"].apply(normalize_text).apply(canonical_polymer_name)

    if "sequence" in combined.columns:
        combined["sequence"] = combined["sequence"].apply(clean_sequence)

    # REMOVE INVALID ROWS
    if "plastic" in combined.columns and "sequence" in combined.columns:
        combined = combined.dropna(subset=["plastic", "sequence"])

    # REMOVE DUPLICATES
    if "plastic" in combined.columns and "sequence" in combined.columns:
        combined = combined.drop_duplicates(subset=["plastic", "sequence"])

    print("Rows after cleaning:", len(combined))

    # SAVE
    combined.to_csv(output_file, index=False)
    print(f"Saved cleaned dataset to: {output_file}")


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    process_plastizyme_data(
        "CLEAN_V4.csv",
        "Cleaned_Combined_filled.csv",
        "plastizyme_cleaned_dataset.csv"
    )
