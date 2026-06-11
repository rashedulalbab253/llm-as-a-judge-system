"""
Step 5: Visualization and Analysis of Evaluation Results.

Generates comprehensive plots and analysis reports:
  - Score distributions per criterion
  - Correlation heatmaps between criteria
  - Pass/fail rates
  - Human vs Judge agreement scatter plots
"""

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EVALUATION_CRITERIA, RESULTS_DIR, PLOTS_DIR, GEVAL_THRESHOLD

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Style configuration
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
sns.set_palette(COLORS)


def plot_score_distributions(df):
    """Plot score distributions for each evaluation criterion."""
    criteria_keys = list(EVALUATION_CRITERIA.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, key in enumerate(criteria_keys):
        col = f"{key}_score"
        if col not in df.columns:
            continue
        scores = df[col].dropna()
        ax = axes[i]
        ax.hist(scores, bins=20, color=COLORS[i], alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.axvline(scores.mean(), color="black", linestyle="--", linewidth=1.5, label=f"Mean: {scores.mean():.3f}")
        ax.axvline(GEVAL_THRESHOLD, color="red", linestyle=":", linewidth=1.5, label=f"Threshold: {GEVAL_THRESHOLD}")
        ax.set_title(EVALUATION_CRITERIA[key]["name"], fontsize=13, fontweight="bold")
        ax.set_xlabel("Score", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)

    plt.suptitle("G-Eval Score Distributions by Criterion", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = PLOTS_DIR / "score_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_correlation_heatmap(df):
    """Plot correlation heatmap between evaluation criteria."""
    score_cols = [f"{k}_score" for k in EVALUATION_CRITERIA if f"{k}_score" in df.columns]
    if len(score_cols) < 2:
        return
    labels = [EVALUATION_CRITERIA[c.replace("_score", "")]["name"] for c in score_cols]
    corr = df[score_cols].corr()
    corr.index = labels
    corr.columns = labels

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdYlBu_r", center=0,
                square=True, linewidths=1, ax=ax, vmin=-1, vmax=1,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Inter-Criteria Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    path = PLOTS_DIR / "correlation_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_pass_fail_rates(df):
    """Plot pass/fail rates for each criterion."""
    criteria_keys = list(EVALUATION_CRITERIA.keys())
    pass_rates = []
    labels = []
    for key in criteria_keys:
        col = f"{key}_score"
        if col in df.columns:
            rate = (df[col].dropna() >= GEVAL_THRESHOLD).mean()
            pass_rates.append(rate)
            labels.append(EVALUATION_CRITERIA[key]["name"])

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, pass_rates, color=COLORS[:len(labels)], edgecolor="white", height=0.6)
    for bar, rate in zip(bars, pass_rates):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1%}", va="center", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Pass Rate", fontsize=12)
    ax.set_title(f"Pass Rate by Criterion (threshold ≥ {GEVAL_THRESHOLD})", fontsize=14, fontweight="bold")
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
    plt.tight_layout()
    path = PLOTS_DIR / "pass_fail_rates.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_score_boxplots(df):
    """Box plots comparing scores across criteria."""
    data = []
    for key in EVALUATION_CRITERIA:
        col = f"{key}_score"
        if col in df.columns:
            for score in df[col].dropna():
                data.append({"Criterion": EVALUATION_CRITERIA[key]["name"], "Score": score})
    if not data:
        return
    plot_df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=plot_df, x="Score", y="Criterion", palette=COLORS, ax=ax, width=0.5)
    ax.axvline(GEVAL_THRESHOLD, color="red", linestyle=":", linewidth=1.5, label=f"Threshold: {GEVAL_THRESHOLD}")
    ax.set_title("Score Distribution by Criterion", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = PLOTS_DIR / "score_boxplots.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_agreement_scatter(agreement_path=None):
    """Scatter plot of human vs G-Eval scores (if annotations available)."""
    from config import HUMAN_EVAL_DIR
    ann_path = HUMAN_EVAL_DIR / "annotation_completed.csv"
    if not ann_path.exists():
        logger.info("No completed annotations found, skipping agreement scatter plot")
        return
    df = pd.read_csv(ann_path)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    criteria_keys = list(EVALUATION_CRITERIA.keys())
    for i, key in enumerate(criteria_keys):
        gc, hc = f"geval_{key}_score", f"human_{key}_score"
        if gc not in df.columns or hc not in df.columns:
            continue
        gs = pd.to_numeric(df[gc], errors="coerce")
        hs = pd.to_numeric(df[hc], errors="coerce")
        mask = gs.notna() & hs.notna()
        ax = axes[i]
        ax.scatter(hs[mask], gs[mask], alpha=0.6, color=COLORS[i], s=50, edgecolors="white")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect agreement")
        ax.set_xlabel("Human Score", fontsize=11)
        ax.set_ylabel("G-Eval Score", fontsize=11)
        ax.set_title(EVALUATION_CRITERIA[key]["name"], fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    plt.suptitle("Human vs G-Eval Judge Agreement", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = PLOTS_DIR / "agreement_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def generate_summary_report(df):
    """Generate a text summary report of all results."""
    report = ["=" * 60, "  LLM-as-a-Judge: Evaluation Summary Report", "=" * 60, ""]
    report.append(f"Total samples evaluated: {len(df)}")
    report.append(f"Passing threshold: {GEVAL_THRESHOLD}\n")
    for key in EVALUATION_CRITERIA:
        col = f"{key}_score"
        if col not in df.columns:
            continue
        scores = df[col].dropna()
        name = EVALUATION_CRITERIA[key]["name"]
        report.append(f"{name}:")
        report.append(f"  Mean:      {scores.mean():.4f}")
        report.append(f"  Std Dev:   {scores.std():.4f}")
        report.append(f"  Median:    {scores.median():.4f}")
        report.append(f"  Min/Max:   {scores.min():.4f} / {scores.max():.4f}")
        report.append(f"  Pass Rate: {(scores >= GEVAL_THRESHOLD).mean():.1%}")
        report.append("")
    if "avg_score" in df.columns:
        avg = df["avg_score"].dropna()
        report.append(f"Overall Average Score: {avg.mean():.4f} (std: {avg.std():.4f})")

    # Load agreement if available
    agr_path = RESULTS_DIR / "agreement_results.json"
    if agr_path.exists():
        with open(agr_path) as f:
            agr = json.load(f)
        overall = agr.get("overall", {})
        report.append(f"\nHuman-Judge Agreement: {overall.get('exact_agreement', 0):.1%}")
        report.append(f"Pearson r: {overall.get('pearson_r', 0):.4f}")

    report_text = "\n".join(report)
    path = RESULTS_DIR / "evaluation_report.txt"
    with open(path, "w") as f:
        f.write(report_text)
    logger.info(f"Saved report to {path}")
    print(report_text)


def main():
    logger.info("STEP 5: Generating Visualizations")
    results_path = RESULTS_DIR / "geval_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Run evaluate_outputs.py first. Missing: {results_path}")
    df = pd.read_csv(results_path)

    plot_score_distributions(df)
    plot_correlation_heatmap(df)
    plot_pass_fail_rates(df)
    plot_score_boxplots(df)
    plot_agreement_scatter()
    generate_summary_report(df)
    logger.info("\n[SUCCESS] Visualization complete! Check plots/ directory.")


if __name__ == "__main__":
    main()
