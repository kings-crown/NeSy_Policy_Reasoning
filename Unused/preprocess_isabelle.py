import json
import os
import re
import sys

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

#sys.set_int_max_str_digits(100000)


def clean_text(text):
    """Remove unwanted characters and clean text."""
    if not isinstance(text, str):
        return text
    text = text.replace("\n", " ").strip()  # Normalize new lines
    text = re.sub(r"\s+", " ", text)  # Remove excessive spaces
    return text


def filter_valid_entries(df):
    """Filter dataset entries to remove invalid data."""
    # Remove rows where any important column is missing
    df = df.dropna(subset=["natural_language_statement", "isabelle_translation"])

    # Remove problems containing "<image>", "<span", or weird HTML-like patterns
    df = df[
        ~df["natural_language_statement"].str.contains("<image>|<span ", regex=True, na=False)
    ]

    # Filter based on proof length (keeping it reasonable for training)
    df["proof_length"] = df["formal_proof"].apply(
        lambda x: len(tokenizer.tokenize(x)) if isinstance(x, str) else 0
    )
    df = df[df["proof_length"] < 2048]
    df = df[df["proof_length"] > 64]

    return df


if __name__ == "__main__":
    os.makedirs("/data/isabelle", exist_ok=True)
    
    # Load dataset
    ds = load_dataset("kings-crown/Isabelle_SFT", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    for split in ds.keys():
        df = ds[split].to_pandas()
        df["task_id"] = np.arange(len(df))

        # Clean text fields
        for col in ["natural_language_statement", "isabelle_translation", "formal_proof", "isabelle_body"]:
            df[col] = df[col].apply(clean_text)

        # Filter dataset to remove problematic entries
        df = filter_valid_entries(df)

        # Select only relevant fields
        df = df[["task_id", "natural_language_statement", "isabelle_translation", "formal_proof", "isabelle_body"]]

        # Deduplicate by `natural_language_statement`
        df = df.drop_duplicates(subset=["natural_language_statement"])

        # Save to JSONL format
        df.to_json(
            f"scripts/data/isabelle/{split}.jsonl",
            lines=True,
            orient="records",
            force_ascii=False,
        )
        del df
