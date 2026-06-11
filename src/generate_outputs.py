"""
Step 2: Generate outputs from the weaker model (GPT-3.5-Turbo).

Runs the sampled Alpaca Eval instructions through the weaker model and
collects its outputs for subsequent evaluation by the GPT-4 judge.
"""

import json
import time
import logging
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OPENAI_API_KEY, WEAK_MODEL, WEAK_MODEL_TEMPERATURE,
    WEAK_MODEL_MAX_TOKENS, DATA_DIR, OUTPUTS_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def initialize_client() -> OpenAI:
    """Initialize the OpenAI client."""
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not set! Please set it in .env file or environment variable.\n"
            "Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


def generate_single_output(client: OpenAI, instruction: str, retries: int = 3) -> dict:
    """
    Generate a single output from the weak model for a given instruction.
    
    Args:
        client: OpenAI client
        instruction: The user instruction/prompt
        retries: Number of retry attempts on failure
    
    Returns:
        dict with keys: output, model, tokens_used, latency_ms
    """
    for attempt in range(retries):
        try:
            start_time = time.time()
            
            response = client.chat.completions.create(
                model=WEAK_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful, harmless, and honest AI assistant. "
                            "Follow the user's instructions carefully and provide "
                            "a thorough, accurate response."
                        )
                    },
                    {
                        "role": "user",
                        "content": instruction
                    }
                ],
                temperature=WEAK_MODEL_TEMPERATURE,
                max_tokens=WEAK_MODEL_MAX_TOKENS,
            )
            
            latency = (time.time() - start_time) * 1000  # ms
            
            output_text = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            return {
                "output": output_text,
                "model": WEAK_MODEL,
                "tokens_used": tokens_used,
                "latency_ms": round(latency, 2)
            }
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                wait = 2 ** attempt
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"All retries exhausted for instruction: {instruction[:80]}...")
                return {
                    "output": f"[ERROR] Generation failed: {str(e)}",
                    "model": WEAK_MODEL,
                    "tokens_used": 0,
                    "latency_ms": 0
                }


def generate_all_outputs(df: pd.DataFrame, checkpoint_every: int = 25) -> pd.DataFrame:
    """
    Generate weak model outputs for all instructions in the DataFrame.
    
    Includes checkpointing to resume from interruptions.
    
    Args:
        df: DataFrame with 'instruction' column
        checkpoint_every: Save checkpoint every N generations
    
    Returns:
        DataFrame with added columns: weak_output, weak_model, tokens_used, latency_ms
    """
    client = initialize_client()
    
    # Check for existing checkpoint
    checkpoint_path = OUTPUTS_DIR / "generation_checkpoint.json"
    results = []
    start_idx = 0
    
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        results = checkpoint_data.get("results", [])
        start_idx = len(results)
        logger.info(f"Resuming from checkpoint: {start_idx}/{len(df)} completed")
    
    logger.info(f"Generating outputs with {WEAK_MODEL} for {len(df) - start_idx} remaining instructions...")
    
    total_tokens = sum(r.get("tokens_used", 0) for r in results)
    total_latency = sum(r.get("latency_ms", 0) for r in results)
    
    for idx in tqdm(range(start_idx, len(df)), desc="Generating outputs"):
        instruction = df.iloc[idx]["instruction"]
        
        result = generate_single_output(client, instruction)
        results.append(result)
        
        total_tokens += result["tokens_used"]
        total_latency += result["latency_ms"]
        
        # Checkpoint
        if (idx + 1) % checkpoint_every == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"results": results}, f, indent=2)
            logger.info(f"  Checkpoint saved at {idx + 1}/{len(df)}")
    
    # Build output DataFrame
    df_out = df.copy()
    df_out["weak_output"] = [r["output"] for r in results]
    df_out["weak_model"] = [r["model"] for r in results]
    df_out["tokens_used"] = [r["tokens_used"] for r in results]
    df_out["latency_ms"] = [r["latency_ms"] for r in results]
    
    # Summary stats
    logger.info(f"\n{'-' * 50}")
    logger.info(f"Generation Complete:")
    logger.info(f"  Model:              {WEAK_MODEL}")
    logger.info(f"  Total outputs:      {len(results)}")
    logger.info(f"  Total tokens:       {total_tokens:,}")
    logger.info(f"  Avg latency:        {total_latency / len(results):.0f}ms")
    logger.info(f"  Errors:             {sum(1 for r in results if r['output'].startswith('[ERROR]'))}")
    avg_len = df_out["weak_output"].str.len().mean()
    logger.info(f"  Avg output length:  {avg_len:.0f} chars")
    logger.info(f"{'-' * 50}\n")
    
    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    
    return df_out


def save_outputs(df: pd.DataFrame, filename: str = "weak_model_outputs.csv"):
    """Save generated outputs to disk."""
    csv_path = OUTPUTS_DIR / filename
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"Saved outputs to {csv_path}")
    
    json_path = OUTPUTS_DIR / filename.replace(".csv", ".json")
    df.to_json(json_path, orient="records", indent=2, force_ascii=False)
    logger.info(f"Saved JSON to {json_path}")
    
    return csv_path


def main():
    """Load sampled dataset and generate weak model outputs."""
    logger.info("=" * 70)
    logger.info("STEP 2: Generating Weak Model Outputs")
    logger.info("=" * 70)
    
    # Load sampled dataset
    sampled_path = DATA_DIR / "alpaca_eval_sampled.csv"
    if not sampled_path.exists():
        logger.error("Sampled dataset not found! Run data_loader.py first (Step 1).")
        raise FileNotFoundError(f"Missing: {sampled_path}")
    
    df = pd.read_csv(sampled_path)
    logger.info(f"Loaded {len(df)} sampled instructions")
    
    # Generate outputs
    df_with_outputs = generate_all_outputs(df)
    
    # Save
    save_outputs(df_with_outputs)
    
    # Show examples
    logger.info("\nExample outputs:")
    for i, row in df_with_outputs.head(3).iterrows():
        logger.info(f"\n  [{i+1}] Instruction: {row['instruction'][:80]}...")
        logger.info(f"      Weak Output: {row['weak_output'][:80]}...")
    
    logger.info("\n[SUCCESS] Output generation complete!")
    return df_with_outputs


if __name__ == "__main__":
    main()
