"""
Step 1: Load the Alpaca Eval dataset from HuggingFace.

Loads 805 instruction-following examples with reference answers from a strong
baseline model (text-davinci-003). Each example contains:
  - instruction: the user prompt
  - output: the reference/baseline answer
  - generator: which model produced the reference
  - dataset: source dataset name
"""

import json
import logging
from pathlib import Path

import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, DATASET_NAME, DATASET_SPLIT, NUM_SAMPLES_GENERATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Direct URL to the raw JSON file on HuggingFace
ALPACA_EVAL_JSON_URL = (
    "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json"
)


def load_alpaca_eval() -> pd.DataFrame:
    """
    Load the Alpaca Eval dataset from HuggingFace and return as DataFrame.
    
    Downloads the raw JSON directly since the datasets library no longer
    supports the legacy loading script used by this dataset.
    
    Returns:
        pd.DataFrame with columns: [instruction, reference_output, generator, dataset_source]
    """
    cache_path = DATA_DIR / "alpaca_eval_cache.json"
    
    # Try loading from cache first
    if cache_path.exists():
        logger.info(f"Loading dataset from local cache: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        # Download directly from HuggingFace
        logger.info(f"Downloading Alpaca Eval dataset from HuggingFace...")
        try:
            response = requests.get(ALPACA_EVAL_JSON_URL, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            # Cache locally
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Cached dataset to {cache_path}")
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
        
        df = pd.DataFrame(data)
    
    # Standardize column names
    column_mapping = {
        "instruction": "instruction",
        "output": "reference_output",
        "generator": "generator",
        "dataset": "dataset_source"
    }
    
    rename_map = {}
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            rename_map[old_name] = new_name
    
    df = df.rename(columns=rename_map)
    
    logger.info(f"Loaded {len(df)} examples from Alpaca Eval")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Generators: {df['generator'].unique().tolist() if 'generator' in df.columns else 'N/A'}")
    
    return df


def sample_instructions(df: pd.DataFrame, n: int = None) -> pd.DataFrame:
    """
    Sample N instructions from the dataset for evaluation.
    
    Args:
        df: Full Alpaca Eval DataFrame
        n: Number of samples (defaults to NUM_SAMPLES_GENERATE from config)
    
    Returns:
        Sampled DataFrame
    """
    if n is None:
        n = NUM_SAMPLES_GENERATE
    
    if n >= len(df):
        logger.info(f"Requested {n} samples but dataset has {len(df)} — using all.")
        return df.copy()
    
    sampled = df.sample(n=n, random_state=42).reset_index(drop=True)
    logger.info(f"Sampled {len(sampled)} instructions for evaluation")
    return sampled


def save_dataset(df: pd.DataFrame, filename: str = "alpaca_eval_dataset.csv"):
    """Save the processed dataset to disk."""
    path = DATA_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"Saved dataset to {path}")
    
    # Also save as JSON for easier inspection
    json_path = DATA_DIR / filename.replace(".csv", ".json")
    df.to_json(json_path, orient="records", indent=2, force_ascii=False)
    logger.info(f"Saved JSON to {json_path}")
    
    return path


def main():
    """Load, sample, and save the Alpaca Eval dataset."""
    logger.info("=" * 70)
    logger.info("STEP 1: Loading Alpaca Eval Dataset")
    logger.info("=" * 70)
    
    # Load full dataset
    df = load_alpaca_eval()
    
    # Display stats
    logger.info(f"\n{'-' * 50}")
    logger.info(f"Dataset Statistics:")
    logger.info(f"  Total examples:     {len(df)}")
    logger.info(f"  Columns:            {list(df.columns)}")
    if "reference_output" in df.columns:
        avg_len = df["reference_output"].str.len().mean()
        logger.info(f"  Avg reference len:  {avg_len:.0f} chars")
    logger.info(f"{'-' * 50}\n")
    
    # Sample for evaluation
    sampled = sample_instructions(df)
    
    # Save both full and sampled
    save_dataset(df, "alpaca_eval_full.csv")
    save_dataset(sampled, "alpaca_eval_sampled.csv")
    
    # Show first few examples
    logger.info("\nFirst 3 instructions:")
    for i, row in sampled.head(3).iterrows():
        logger.info(f"\n  [{i+1}] Instruction: {row['instruction'][:100]}...")
        if "reference_output" in row:
            logger.info(f"      Reference:   {str(row['reference_output'])[:100]}...")
    
    logger.info("\n[SUCCESS] Dataset loading complete!")
    return sampled


if __name__ == "__main__":
    main()
