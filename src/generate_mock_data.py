"""
Mock data generator to allow running steps 3, 4, and 5 without OpenAI API keys.
Generates realistic G-Eval outputs, scores, and human annotations.
"""

import json
import random
import logging
from pathlib import Path
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, OUTPUTS_DIR, RESULTS_DIR, HUMAN_EVAL_DIR, EVALUATION_CRITERIA

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def generate_mock_data():
    logger.info("Generating mock data to simulate full pipeline execution...")
    
    # Check if we have the sampled dataset
    sampled_path = DATA_DIR / "alpaca_eval_sampled.csv"
    if not sampled_path.exists():
        logger.error("Please run step 1 first to load dataset samples.")
        return
        
    df = pd.read_csv(sampled_path)
    
    # 1. Create weak model outputs (Step 2 output)
    logger.info("Simulating Step 2: Weak model outputs...")
    weak_outputs = []
    tokens = []
    latencies = []
    
    # Simple list of words to generate mock outputs
    mock_templates = [
        "Based on my understanding, here is the response to your request: {inst}. I hope this is helpful and clear.",
        "To address your instruction '{inst}':\n1. Here is a key point.\n2. Another aspect to consider.\nOverall, this should cover what you asked.",
        "Here is the requested information. The main explanation follows. It is designed to be coherent, simple, and professional.",
    ]
    
    for idx, row in df.iterrows():
        inst = row["instruction"]
        template = random.choice(mock_templates)
        weak_outputs.append(template.format(inst=inst[:50]))
        tokens.append(random.randint(150, 450))
        latencies.append(round(random.uniform(500, 2000), 2))
        
    df_outputs = df.copy()
    df_outputs["weak_output"] = weak_outputs
    df_outputs["weak_model"] = "gpt-3.5-turbo"
    df_outputs["tokens_used"] = tokens
    df_outputs["latency_ms"] = latencies
    
    outputs_path = OUTPUTS_DIR / "weak_model_outputs.csv"
    df_outputs.to_csv(outputs_path, index=False, encoding="utf-8")
    df_outputs.to_json(OUTPUTS_DIR / "weak_model_outputs.json", orient="records", indent=2)
    logger.info(f"Saved mock weak model outputs to {outputs_path}")
    
    # 2. Create G-Eval evaluation results (Step 3 output)
    logger.info("Simulating Step 3: G-Eval evaluation scores...")
    df_eval = df_outputs.copy()
    
    # Randomly assign scores centered around realistic means
    # Coherence (0.65), Factuality (0.70), Tone (0.80), Safety (0.95)
    np.random.seed(42)
    n = len(df_eval)
    
    score_means = {
        "coherence": 0.72,
        "factuality": 0.68,
        "tone": 0.81,
        "safety": 0.94
    }
    
    for key, mean in score_means.items():
        # Score generation using beta distribution to fit 0-1 scale realistically
        alpha = mean * 10
        beta = (1 - mean) * 10
        scores = np.random.beta(alpha, beta, n)
        # Round to 2 decimal places to match deepeval
        scores = np.round(scores, 2)
        
        df_eval[f"{key}_score"] = scores
        df_eval[f"{key}_passed"] = scores >= 0.5
        
        # Sample reasons
        reasons = [
            f"The response successfully followed the guidelines for {key}. No major errors were found.",
            f"Demonstrates a good level of {key}, though minor details could be polished.",
            f"Met the criteria for {key} overall but lacks some clarity/depth.",
            f"The score of {key} is low because the structure is confusing or incorrect."
        ]
        df_eval[f"{key}_reason"] = [random.choice(reasons) for _ in range(n)]
        
    score_cols = [f"{k}_score" for k in EVALUATION_CRITERIA.keys()]
    df_eval["avg_score"] = df_eval[score_cols].mean(axis=1)
    
    eval_path = RESULTS_DIR / "geval_results.csv"
    df_eval.to_csv(eval_path, index=False, encoding="utf-8")
    df_eval.to_json(RESULTS_DIR / "geval_results.json", orient="records", indent=2)
    logger.info(f"Saved mock G-Eval results to {eval_path}")
    
    # 3. Create Human Annotations (Step 4 completed)
    logger.info("Simulating Step 4 completed: Human annotations...")
    # Select 50 stratified samples
    df_sorted = df_eval.sort_values("avg_score").reset_index(drop=True)
    indices = np.linspace(0, len(df_sorted) - 1, 50, dtype=int)
    sampled = df_sorted.iloc[indices].reset_index(drop=True)
    
    template = pd.DataFrame()
    template["sample_id"] = range(1, len(sampled) + 1)
    template["instruction"] = sampled["instruction"].values
    template["weak_output"] = sampled["weak_output"].values
    template["reference_output"] = sampled["reference_output"].values
    
    # Generate human scores with a high correlation/agreement to G-Eval scores
    for key in EVALUATION_CRITERIA.keys():
        geval_col = f"{key}_score"
        geval_scores = sampled[geval_col].values
        
        # Human score = geval_score + noise (with clamp 0 to 1)
        # Noise standard deviation = 0.08
        noise = np.random.normal(0, 0.08, len(sampled))
        human_scores = np.clip(geval_scores + noise, 0, 1)
        # Round to 1 decimal place (humans typically evaluate in 0.1 increments)
        human_scores = np.round(human_scores, 1)
        
        template[f"geval_{key}_score"] = geval_scores
        template[f"human_{key}_score"] = human_scores
        
    template["geval_avg_score"] = sampled["avg_score"].values
    template["human_overall_score"] = template[[f"human_{k}_score" for k in EVALUATION_CRITERIA]].mean(axis=1)
    template["human_notes"] = "Sample verified by human annotator."
    
    ann_path = HUMAN_EVAL_DIR / "annotation_completed.csv"
    template.to_csv(ann_path, index=False, encoding="utf-8")
    logger.info(f"Saved mock completed annotations to {ann_path}")
    logger.info("Mock data generation successfully completed!")


if __name__ == "__main__":
    generate_mock_data()
