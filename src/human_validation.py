"""
Step 4: Human Annotation Tool for Judge Validation.

Generates annotation templates, collects human scores,
and computes agreement rate between G-Eval judge and human annotators.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    EVALUATION_CRITERIA, RESULTS_DIR, HUMAN_EVAL_DIR,
    NUM_SAMPLES_HUMAN, GEVAL_THRESHOLD
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def generate_annotation_template(df, n=None):
    if n is None:
        n = NUM_SAMPLES_HUMAN
    n = min(n, len(df))
    if "avg_score" in df.columns:
        df_sorted = df.sort_values("avg_score").reset_index(drop=True)
        indices = np.linspace(0, len(df_sorted) - 1, n, dtype=int)
        sampled = df_sorted.iloc[indices].reset_index(drop=True)
    else:
        sampled = df.sample(n=n, random_state=42).reset_index(drop=True)

    template = pd.DataFrame()
    template["sample_id"] = range(1, len(sampled) + 1)
    template["instruction"] = sampled["instruction"].values
    template["weak_output"] = sampled.get("weak_output", sampled.get("actual_output", "")).values
    template["reference_output"] = sampled.get("reference_output", "").values
    for key in EVALUATION_CRITERIA:
        template[f"human_{key}_score"] = ""
    template["human_overall_score"] = ""
    template["human_notes"] = ""
    for key in EVALUATION_CRITERIA:
        col = f"{key}_score"
        if col in sampled.columns:
            template[f"geval_{key}_score"] = sampled[col].values
    if "avg_score" in sampled.columns:
        template["geval_avg_score"] = sampled["avg_score"].values
    return template


def save_annotation_template(template):
    csv_path = HUMAN_EVAL_DIR / "annotation_template.csv"
    template.to_csv(csv_path, index=False, encoding="utf-8")
    json_path = HUMAN_EVAL_DIR / "annotation_template.json"
    template.to_json(json_path, orient="records", indent=2, force_ascii=False)
    instr_path = HUMAN_EVAL_DIR / "ANNOTATION_INSTRUCTIONS.md"
    with open(instr_path, "w", encoding="utf-8") as f:
        f.write(_build_instructions())
    try:
        xlsx_path = HUMAN_EVAL_DIR / "annotation_template.xlsx"
        template.to_excel(xlsx_path, index=False, sheet_name="Annotations")
    except Exception:
        pass
    logger.info(f"Saved annotation template ({len(template)} samples) to {HUMAN_EVAL_DIR}")
    return csv_path


def _build_instructions():
    criteria_text = ""
    for key, cfg in EVALUATION_CRITERIA.items():
        steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(cfg["evaluation_steps"]))
        criteria_text += f"\n### {cfg['name']} (`human_{key}_score`)\n{cfg['criteria']}\n{steps}\n"
    return f"""# Human Annotation Instructions
Score each dimension 0.0-1.0. Open annotation_template.csv, fill human_*_score columns, save as annotation_completed.csv.
{criteria_text}
"""


def load_completed_annotations():
    for name in ["annotation_completed.csv", "annotation_template.csv"]:
        path = HUMAN_EVAL_DIR / name
        if path.exists():
            df = pd.read_csv(path)
            human_cols = [c for c in df.columns if c.startswith("human_") and c.endswith("_score")]
            has_scores = any(df[col].notna().any() and (df[col] != "").any() for col in human_cols)
            if has_scores:
                for col in human_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                return df
    return None


def compute_agreement(df):
    from scipy import stats
    from sklearn.metrics import cohen_kappa_score
    results = {}
    all_g, all_h = [], []
    for key in EVALUATION_CRITERIA:
        gc, hc = f"geval_{key}_score", f"human_{key}_score"
        if gc not in df.columns or hc not in df.columns:
            continue
        mask = df[gc].notna() & df[hc].notna()
        gs = df.loc[mask, gc].astype(float)
        hs = df.loc[mask, hc].astype(float)
        if len(gs) < 5:
            continue
        all_g.extend(gs.tolist())
        all_h.extend(hs.tolist())
        exact = (abs(gs - hs) <= 0.15).mean()
        pr, pp = stats.pearsonr(gs, hs)
        sr, sp = stats.spearmanr(gs, hs)
        try:
            kappa = cohen_kappa_score((hs >= GEVAL_THRESHOLD).astype(int), (gs >= GEVAL_THRESHOLD).astype(int))
        except Exception:
            kappa = float("nan")
        results[key] = {
            "n_pairs": len(gs), "exact_agreement": round(exact, 4),
            "pearson_r": round(pr, 4), "spearman_r": round(sr, 4),
            "cohens_kappa": round(kappa, 4), "mae": round(abs(gs - hs).mean(), 4),
        }
    if all_g:
        ag, ah = np.array(all_g), np.array(all_h)
        results["overall"] = {
            "n_pairs": len(ag),
            "exact_agreement": round((abs(ag - ah) <= 0.15).mean(), 4),
            "pearson_r": round(stats.pearsonr(ag, ah)[0], 4),
            "spearman_r": round(stats.spearmanr(ag, ah)[0], 4),
            "mae": round(abs(ag - ah).mean(), 4),
        }
    return results


def print_agreement_report(agreement):
    logger.info(f"\n{'=' * 60}")
    logger.info("  JUDGE VALIDATION: G-Eval vs Human Agreement")
    logger.info(f"{'=' * 60}")
    for key, m in agreement.items():
        label = EVALUATION_CRITERIA.get(key, {}).get("name", key.upper())
        logger.info(f"\n  {label}: Agreement={m['exact_agreement']:.1%}  Pearson={m['pearson_r']:.3f}  MAE={m['mae']:.3f}")
    o = agreement.get("overall", {})
    logger.info(f"\n  RESUME METRIC: {o.get('exact_agreement',0):.1%} agreement rate (r={o.get('pearson_r',0):.3f})")


def main():
    logger.info("STEP 4: Human Validation")
    results_path = RESULTS_DIR / "geval_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Run evaluate_outputs.py first. Missing: {results_path}")
    df = pd.read_csv(results_path)
    completed = load_completed_annotations()
    if completed is not None:
        agreement = compute_agreement(completed)
        print_agreement_report(agreement)
        with open(RESULTS_DIR / "agreement_results.json", "w") as f:
            json.dump(agreement, f, indent=2)
        return agreement
    else:
        template = generate_annotation_template(df)
        save_annotation_template(template)
        logger.info("Template generated. Fill in scores, save as annotation_completed.csv, re-run.")
        return template


if __name__ == "__main__":
    main()
