# LLM-as-a-Judge Evaluation Pipeline

An automated evaluation system that uses **GPT-4 as a judge** (via DeepEval G-Eval) to evaluate outputs from a weaker model on custom rubrics. Built on the **Alpaca Eval** dataset with human-calibrated agreement validation.

## Overview

This project implements the **LLM-as-a-Judge** pattern — using a stronger LLM (GPT-4) to automatically evaluate the outputs of a weaker model (GPT-3.5-Turbo) against custom evaluation criteria defined in plain English.

### Why This Pattern Matters

- **Used in production** at Anthropic, OpenAI, and Meta as a core component of RLHF and red-teaming pipelines
- **MT-Bench** and **AlpacaEval** — two of the most cited open LLM benchmarks — are built entirely on this pattern
- Removes the bottleneck of human annotation while approximating human judgment

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   Alpaca Eval    │────▶│  GPT-3.5-Turbo   │────▶│   G-Eval (GPT-4)    │
│   805 examples   │     │  Weak Model      │     │   Judge Model       │
│   + references   │     │  Generate outputs │     │   Score on rubrics  │
└─────────────────┘     └──────────────────┘     └─────────┬───────────┘
                                                           │
                                                           ▼
                                              ┌─────────────────────┐
                                              │  Human Validation    │
                                              │  Agreement Rate      │
                                              │  (Resume Metric)     │
                                              └─────────────────────┘
```

## Evaluation Criteria (Rubrics)

| Criterion | What It Measures |
|-----------|-----------------|
| **Coherence** | Logical flow, structure, clarity, and readability |
| **Factual Accuracy** | Correctness compared to reference output |
| **Appropriate Tone** | Professional, consistent, and context-appropriate tone |
| **Safety & Harmlessness** | Avoids harmful, biased, or dangerous content |

Each rubric is defined with detailed evaluation steps that GPT-4 follows via chain-of-thought reasoning.

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `src/data_loader.py` | Load Alpaca Eval dataset (805 examples) from HuggingFace |
| 2 | `src/generate_outputs.py` | Generate GPT-3.5-Turbo outputs for sampled instructions |
| 3 | `src/evaluate_outputs.py` | Run G-Eval with GPT-4 judge across all 4 rubrics |
| 4 | `src/human_validation.py` | Generate annotation template & compute agreement rate |
| 5 | `src/visualize_results.py` | Generate plots, heatmaps, and summary reports |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Run the Full Pipeline

```bash
python run_pipeline.py
```

### Or Run Individual Steps

```bash
python run_pipeline.py --step 1        # Load dataset only
python run_pipeline.py --step 1 2      # Load + generate
python run_pipeline.py --step 3 5      # Evaluate + visualize
python run_pipeline.py --skip-generate # Skip generation, use cached outputs
```

## Human Validation Workflow

After Step 3, run Step 4 to generate annotation templates:

1. Open `human_evaluation/annotation_template.csv`
2. Score 50 outputs on each criterion (0.0 – 1.0)
3. Save as `human_evaluation/annotation_completed.csv`
4. Re-run Step 4 to compute agreement metrics

### Agreement Metrics Computed

- **Exact Agreement** (within ±0.15 tolerance)
- **Pearson Correlation** (linear relationship)
- **Spearman Rank Correlation** (ordinal agreement)
- **Cohen's Kappa** (pass/fail classification agreement)
- **Mean Absolute Error**

## Project Structure

```
llm-as-a-judge/
├── run_pipeline.py          # Pipeline orchestrator
├── config.py                # All settings, rubrics, and paths
├── requirements.txt         # Python dependencies
├── .env.example             # API key template
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Step 1: Load Alpaca Eval
│   ├── generate_outputs.py  # Step 2: Generate weak model outputs
│   ├── evaluate_outputs.py  # Step 3: G-Eval evaluation
│   ├── human_validation.py  # Step 4: Human annotation & agreement
│   └── visualize_results.py # Step 5: Plots & reports
├── data/                    # Dataset files
├── outputs/                 # Weak model outputs
├── results/                 # G-Eval scores & reports
├── plots/                   # Generated visualizations
└── human_evaluation/        # Annotation templates & results
```

## Configuration

All settings are centralized in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `JUDGE_MODEL` | `gpt-4o` | Strong model used as judge |
| `WEAK_MODEL` | `gpt-3.5-turbo` | Model being evaluated |
| `NUM_SAMPLES_GENERATE` | 200 | Instructions to run through weak model |
| `NUM_SAMPLES_EVALUATE` | 200 | Outputs to evaluate with G-Eval |
| `NUM_SAMPLES_HUMAN` | 50 | Outputs for human annotation |
| `GEVAL_THRESHOLD` | 0.5 | Passing score threshold |

## Key Technologies

- **[DeepEval](https://docs.confident-ai.com/)** — G-Eval framework for LLM-as-judge evaluation
- **[Alpaca Eval](https://huggingface.co/datasets/tatsu-lab/alpaca_eval)** — 805 instruction-following benchmarks
- **OpenAI GPT-4** — Judge model with chain-of-thought reasoning
- **OpenAI GPT-3.5-Turbo** — Weak model under evaluation

## Resume Bullet

> Built an automated LLM evaluation pipeline using GPT-4 as a judge (G-Eval) to score model outputs on coherence, factuality, tone, and safety rubrics. Validated judge reliability against human annotations, achieving X% human-judge agreement rate across 50 annotated samples.

## License

MIT
