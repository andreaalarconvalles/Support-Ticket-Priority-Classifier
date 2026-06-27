"""Text cleaning + tokenisation pipeline for the Support Ticket Priority Classifier.

Phase 3 (Person 2 support / Juan Jose ML): clean -> tokenise -> pad -> encode labels.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Fix the import path so we can import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# 95th-percentile word count. We'll stick with 57 for now, or dynamically adjust.
MAX_SEQ_LEN = 57

# The new dataset uses 3 priorities.
LABEL_TO_ID = {"low": 0, "medium": 1, "high": 2}

def split_data(df: pd.DataFrame, val_size=0.15, test_size=0.15, random_state=42):
    """Stratified 70/15/15 train/val/test split on priority."""
    train_df, rest_df = train_test_split(
        df,
        test_size=val_size + test_size,
        stratify=df["priority"],
        random_state=random_state,
    )
    val_df, test_df = train_test_split(
        rest_df,
        test_size=test_size / (val_size + test_size),
        stratify=rest_df["priority"],
        random_state=random_state,
    )
    return train_df, val_df, test_df

def encode_labels(priorities: pd.Series) -> np.ndarray:
    return priorities.map(LABEL_TO_ID).to_numpy()

def run_pipeline():
    """Clean -> dedupe -> split -> tokenize -> pad -> encode labels -> save artifacts."""
    print("Loading raw data...")
    df = pd.read_csv(RAW_DATA_DIR / "aa_dataset-tickets-multi-lang-5-2-50-version.csv")
    
    # 1. Filter English only
    df = df[df["language"] == "en"].copy()
    
    # 2. Combine Subject and Body for full text context
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df["full_text"] = df["subject"] + " - " + df["body"]
    
    # 3. Deduplicate exact texts
    df = df.drop_duplicates(subset=["full_text"], keep="first").reset_index(drop=True)
    
    print(f"Total English tickets after deduplication: {len(df)}")
    
    # 4. Split
    train_df, val_df, test_df = split_data(df)

    # 5. Tokenize & Vectorize
    print("Vectorizing text...")
    
    # TextVectorization automatically standardizes (lowercase + strip punctuation)
    # and splits by whitespace.
    vectorizer = TextVectorization(
        output_mode='int'
        # We REMOVED output_sequence_length so it doesn't auto-pad (which forces 'post')
    )
    
    vectorizer.adapt(train_df["full_text"].values)
    vocab_size = vectorizer.vocabulary_size()

    def to_pre_padded(texts):
        # 1. Convert to ragged tensor of tokens
        sequences = vectorizer(texts)
        # 2. Pad using Keras utility to enforce 'pre' padding
        return pad_sequences(sequences.numpy(), maxlen=MAX_SEQ_LEN, padding="pre", truncating="post")

    X_train_text = to_pre_padded(train_df["full_text"].values)
    X_val_text = to_pre_padded(val_df["full_text"].values)
    X_test_text = to_pre_padded(test_df["full_text"].values)

    # 6. Encode Target
    y_train = encode_labels(train_df["priority"])
    y_val = encode_labels(val_df["priority"])
    y_test = encode_labels(test_df["priority"])

    # 7. Save outputs
    print("Saving processed data...")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    np.save(PROCESSED_DATA_DIR / "X_train_text.npy", X_train_text)
    np.save(PROCESSED_DATA_DIR / "X_val_text.npy", X_val_text)
    np.save(PROCESSED_DATA_DIR / "X_test_text.npy", X_test_text)
    
    np.save(PROCESSED_DATA_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DATA_DIR / "y_val.npy", y_val)
    np.save(PROCESSED_DATA_DIR / "y_test.npy", y_test)

    # TextVectorization can't be easily pickled, but we can save its vocabulary
    vocab = vectorizer.get_vocabulary()
    with open(PROCESSED_DATA_DIR / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    notes = (
        "Handoff notes for Juan Jose (Phase 4-6)\n"
        "========================================\n"
        f"VOCAB_SIZE  = {vocab_size}\n"
        f"MAX_SEQ_LEN = {MAX_SEQ_LEN}\n"
        f"NUM_CLASSES = {len(LABEL_TO_ID)}\n"
        f"LABEL_TO_ID = {LABEL_TO_ID}\n"
        f"\nSplit sizes (train/val/test): {len(train_df)} / {len(val_df)} / {len(test_df)}\n"
        "Stratified by priority, random_state=42, 70/15/15.\n"
        "Filtered to English language only. Tabular features dropped to prevent data leakage.\n"
        "Tokenizer fit on train split only - val/test use the same tokenizer,\n"
        "unseen words map to <OOV>. Padding is PRE (using pad_sequences).\n"
    )
    (PROCESSED_DATA_DIR / "HANDOFF_NOTES.txt").write_text(notes)
    print("Pipeline complete!")
    print(notes)


if __name__ == "__main__":
    run_pipeline()
