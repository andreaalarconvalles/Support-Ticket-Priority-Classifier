"""Text cleaning + tokenisation pipeline for the Support Ticket Priority Classifier.

Phase 3 (Person 2 support / Juan Jose ML): clean -> tokenise -> pad -> encode labels.
"""

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

PLACEHOLDER_PATTERN = re.compile(r"\{[a-z_]+\}")

# 95th-percentile word count from eda.ipynb section 3 - covers the vast majority of
# tickets without padding everything out to the (rare) 63-word max.
MAX_SEQ_LEN = 57

# Fixed order matches the API contract's `confidence` object (Low/Medium/High/Critical),
# not alphabetical - makes the encoding human-readable as an ordinal priority scale.
LABEL_TO_ID = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

DATA_DIR = Path(__file__).parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def fill_product_placeholder(description: str, product: str) -> str:
    """Replace the literal 'product_purchased' token with the real product name.

    The raw dataset is template-generated and every row has at least one unfilled
    placeholder (see backend/eda.ipynb, section 4b) - the most common by far is
    '{product_purchased}', which has the real value sitting in the Product Purchased
    column of the same row. Matches on the bare word rather than requiring both braces,
    since ~2% of rows have a malformed version with one or both braces missing
    (e.g. 'product_purchased}' or bare 'product_purchased') - any leftover braces get
    stripped separately in strip_unfilled_placeholders.
    """
    return description.replace("product_purchased", product)


def strip_unfilled_placeholders(description: str) -> str:
    """Remove any remaining '{placeholder}' tokens that have no real value to fill in.

    Unlike {product_purchased}, things like {error_message} and {product_id} have no
    backing column in the dataset - there's no ground truth to substitute, so the only
    honest option is to drop the token. Covers the full long tail of ~150 placeholder
    variants found in eda.ipynb, not just these two.
    """
    description = PLACEHOLDER_PATTERN.sub("", description)
    # ~4.7% of rows have a malformed/truncated placeholder with no matching brace
    # (e.g. "{product_purch" cut off mid-word, or a stray lone "}"). Drop the leftover
    # brace character itself; the truncated word fragment stays as harmless plain text.
    description = description.replace("{", "").replace("}", "")
    return re.sub(r"\s{2,}", " ", description).strip()


def clean_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning to the 'Ticket Description' column, returning a copy."""
    df = df.copy()
    df["Ticket Description"] = df.apply(
        lambda row: fill_product_placeholder(row["Ticket Description"], row["Product Purchased"]),
        axis=1,
    )
    df["Ticket Description"] = df["Ticket Description"].apply(strip_unfilled_placeholders)
    return df


def drop_duplicate_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact-duplicate ticket descriptions (~4.6% of rows, see eda.ipynb 4a).

    Must happen before the train/val/test split, otherwise the same exact string
    could land in both train and test - trivial leakage that inflates eval metrics.
    """
    return df.drop_duplicates(subset="Ticket Description", keep="first").reset_index(drop=True)


def split_data(df: pd.DataFrame, val_size=0.15, test_size=0.15, random_state=42):
    """Stratified 70/15/15 train/val/test split on Ticket Priority."""
    train_df, rest_df = train_test_split(
        df,
        test_size=val_size + test_size,
        stratify=df["Ticket Priority"],
        random_state=random_state,
    )
    val_df, test_df = train_test_split(
        rest_df,
        test_size=test_size / (val_size + test_size),
        stratify=rest_df["Ticket Priority"],
        random_state=random_state,
    )
    return train_df, val_df, test_df


def encode_labels(priorities: pd.Series) -> np.ndarray:
    return priorities.map(LABEL_TO_ID).to_numpy()


def run_pipeline():
    """Clean -> dedupe -> split -> tokenize -> pad -> encode labels -> save artifacts."""
    df = pd.read_csv(DATA_DIR / "raw" / "customer_support_tickets.csv")
    df = clean_descriptions(df)
    df = drop_duplicate_descriptions(df)

    train_df, val_df, test_df = split_data(df)

    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["Ticket Description"])
    vocab_size = len(tokenizer.word_index) + 1  # +1 since index 0 is reserved for padding

    def to_padded(texts):
        sequences = tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")

    X_train = to_padded(train_df["Ticket Description"])
    X_val = to_padded(val_df["Ticket Description"])
    X_test = to_padded(test_df["Ticket Description"])

    y_train = encode_labels(train_df["Ticket Priority"])
    y_val = encode_labels(val_df["Ticket Priority"])
    y_test = encode_labels(test_df["Ticket Priority"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "X_val.npy", X_val)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "y_val.npy", y_val)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)

    with open(PROCESSED_DIR / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    notes = (
        "Handoff notes for Juan Jose (Phase 4-6)\n"
        "========================================\n"
        f"VOCAB_SIZE  = {vocab_size}\n"
        f"MAX_SEQ_LEN = {MAX_SEQ_LEN}\n"
        f"NUM_CLASSES = {len(LABEL_TO_ID)}\n"
        f"LABEL_TO_ID = {LABEL_TO_ID}\n"
        f"\nSplit sizes (train/val/test): {len(train_df)} / {len(val_df)} / {len(test_df)}\n"
        "Stratified by Ticket Priority, random_state=42, 70/15/15.\n"
        "Exact-duplicate descriptions dropped before splitting (see eda.ipynb 4a).\n"
        "Tokenizer fit on train split only - val/test use the same tokenizer,\n"
        "unseen words map to <OOV> (index 1).\n"
    )
    (PROCESSED_DIR / "HANDOFF_NOTES.txt").write_text(notes)
    print(notes)


if __name__ == "__main__":
    run_pipeline()
