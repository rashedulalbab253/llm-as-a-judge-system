"""
Step 3: Evaluate weak model outputs using DeepEval G-Eval.

Defines 4 evaluation rubrics (coherence, factuality, tone, safety) and
runs GPT-4 as the judge on each output. Collects scores and chain-of-thought
reasoning for every evaluation.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    JUDGE_MODEL, EVALUATION_CRITERIA, GEVAL_THRESHOLD,
    OUTPUTS_DIR, RESULTS_DIR, NUM_SAMPLES_EVALUATE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def create_geval_metrics() -> Dict[str, GEval]:
    """
    Create G-Eval metric instances for each evaluation criterion.
    
    Returns:
        Dict mapping criterion name to GEval metric instance
    """
    metrics = {}
    
    for key, config in EVALUATION_CRITERIA.items():
        # Determine which params the metric needs
        if key == "factuality":
            # Factuality needs reference output for comparison
            eval_params = [
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT
            ]
        else:
            # Coherence, tone, safety are referenceless
            eval_params = [
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT
            ]
        
        metric = GEval(
            name=config["name"],
            criteria=config["criteria"],
            evaluation_steps=config["evaluation_steps"],
            evaluation_params=eval_params,
            model=JUDGE_MODEL,
            threshold=GEVAL_THRESHOLD,
            async_mode=False,  # Sequential for reliability
            verbose_mode=False,
        )
        
        metrics[key] = metric
        logger.info(f"Created metric: {config['name']} (params: {[p.value for p in eval_params]})")
    
    return metrics


def evaluate_single_output(
    metrics: Dict[str, GEval],
    instruction: str,
    actual_output: str,
    reference_output: str = None
) -> Dict:
    """
    Evaluate a single output against all rubrics.
    
    Args:
        metrics: Dict of GEval metric instances
        instruction: The original instruction
        actual_output: The weak model's output
        reference_output: The reference/baseline output (for factuality)
    
    Returns:
        Dict with scores and reasoning for each criterion
    """
    results = {}
    
    for key, metric in metrics.items():
        try:
            # Build test case
            test_case_kwargs = {
                "input": instruction,
                "actual_output": actual_output,
            }
            
            if key == "factuality" and reference_output:
                test_case_kwargs["expected_output"] = reference_output
            
            test_case = LLMTestCase(**test_case_kwargs)
            
            # Run evaluation
            metric.measure(test_case)
            
            results[key] = {
                "score": metric.score,
                "reason": metric.reason,
                "passed": metric.score >= GEVAL_THRESHOLD,
            }
            
        except Exception as e:
            logger.warning(f"  Evaluation failed for {key}: {e}")
            results[key] = {
                "score": None,
                "reason": f"[ERROR] {str(e)}",
                "passed": False,
            }
    
    return results


def evaluate_all_outputs(df: pd.DataFrame, checkpoint_every: int = 10) -> pd.DataFrame:
    """
    Evaluate all weak model outputs using G-Eval across all rubrics.
    
    Args:
        df: DataFrame with columns [instruction, weak_output, reference_output]
        checkpoint_every: Save checkpoint every N evaluations
    
    Returns:
        DataFrame with evaluation scores added
    """
    # Limit to configured sample size
    if len(df) > NUM_SAMPLES_EVALUATE:
        df = df.head(NUM_SAMPLES_EVALUATE).copy()
        logger.info(f"Limiting evaluation to {NUM_SAMPLES_EVALUATE} samples")
    
    # Create metrics
    metrics = create_geval_metrics()
    
    # Check for checkpoint
    checkpoint_path = RESULTS_DIR / "evaluation_checkpoint.json"
    all_results = []
    start_idx = 0
    
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        all_results = checkpoint.get("results", [])
        start_idx = len(all_results)
        logger.info(f"Resuming from checkpoint: {start_idx}/{len(df)} evaluated")
    
    logger.info(f"\nEvaluating {len(df) - start_idx} remaining outputs with {JUDGE_MODEL}...")
    logger.info(f"Criteria: {list(EVALUATION_CRITERIA.keys())}")
    
    for idx in tqdm(range(start_idx, len(df)), desc="Evaluating with G-Eval"):
        row = df.iloc[idx]
        
        instruction = row["instruction"]
        actual_output = row.get("weak_output", "")
        reference_output = row.get("reference_output", "")
        
        # Run all evaluations
        result = evaluate_single_output(
            metrics, instruction, actual_output, reference_output
        )
        result["index"] = idx
        all_results.append(result)
        
        # Checkpoint
        if (idx + 1) % checkpoint_every == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"results": all_results}, f, indent=2)
            logger.info(f"  Checkpoint saved at {idx + 1}/{len(df)}")
    
    # Build results DataFrame
    df_results = df.head(len(all_results)).copy()
    
    for key in EVALUATION_CRITERIA.keys():
        df_results[f"{key}_score"] = [
            r[key]["score"] if r.get(key) else None for r in all_results
        ]
        df_results[f"{key}_reason"] = [
            r[key]["reason"] if r.get(key) else "" for r in all_results
        ]
        df_results[f"{key}_passed"] = [
            r[key]["passed"] if r.get(key) else False for r in all_results
        ]
    
    # Compute aggregate score
    score_cols = [f"{k}_score" for k in EVALUATION_CRITERIA.keys()]
    df_results["avg_score"] = df_results[score_cols].mean(axis=1)
    
    # Summary
    logger.info(f"\n{'-' * 60}")
    logger.info(f"Evaluation Results Summary:")
    logger.info(f"  Judge Model:       {JUDGE_MODEL}")
    logger.info(f"  Total Evaluated:   {len(all_results)}")
    for key in EVALUATION_CRITERIA.keys():
        col = f"{key}_score"
        scores = df_results[col].dropna()
        if len(scores) > 0:
            logger.info(f"  {key:20s} Mean: {scores.mean():.3f}  Std: {scores.std():.3f}  "
                       f"Pass Rate: {(scores >= GEVAL_THRESHOLD).mean():.1%}")
    avg = df_results["avg_score"].dropna()
    if len(avg) > 0:
        logger.info(f"  {'OVERALL':20s} Mean: {avg.mean():.3f}  Std: {avg.std():.3f}")
    logger.info(f"{'-' * 60}\n")
    
    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    
    return df_results


def save_results(df: pd.DataFrame, filename: str = "geval_results.csv"):
    """Save evaluation results to disk."""
    csv_path = RESULTS_DIR / filename
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"Saved results to {csv_path}")
    
    json_path = RESULTS_DIR / filename.replace(".csv", ".json")
    df.to_json(json_path, orient="records", indent=2, force_ascii=False)
    logger.info(f"Saved JSON to {json_path}")
    
    # Save just the scores and reasoning for detailed analysis
    reasoning_data = []
    for _, row in df.iterrows():
        entry = {"instruction": row["instruction"][:200]}
        for key in EVALUATION_CRITERIA.keys():
            entry[f"{key}_score"] = row.get(f"{key}_score")
            entry[f"{key}_reason"] = row.get(f"{key}_reason", "")
        entry["avg_score"] = row.get("avg_score")
        reasoning_data.append(entry)
    
    reasoning_path = RESULTS_DIR / "geval_reasoning.json"
    with open(reasoning_path, "w", encoding="utf-8") as f:
        json.dump(reasoning_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved reasoning to {reasoning_path}")
    
    return csv_path


def main():
    """Load outputs and run G-Eval evaluation."""
    logger.info("=" * 70)
    logger.info("STEP 3: Running G-Eval Evaluation")
    logger.info("=" * 70)
    
    # Load weak model outputs
    outputs_path = OUTPUTS_DIR / "weak_model_outputs.csv"
    if not outputs_path.exists():
        logger.error("Weak model outputs not found! Run generate_outputs.py first (Step 2).")
        raise FileNotFoundError(f"Missing: {outputs_path}")
    
    df = pd.read_csv(outputs_path)
    logger.info(f"Loaded {len(df)} outputs for evaluation")
    
    # Run evaluation
    df_results = evaluate_all_outputs(df)
    
    # Save results
    save_results(df_results)
    
    logger.info("\n[SUCCESS] G-Eval evaluation complete!")
    return df_results


if __name__ == "__main__":
    main()
