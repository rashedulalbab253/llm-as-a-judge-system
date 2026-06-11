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

## GitHub Repository Information

This project is hosted on GitHub. You can clone, track issues, and contribute to the project at:
* **Repository Link**: [rashedulalbab253/llm-as-a-judge-system](https://github.com/rashedulalbab253/llm-as-a-judge-system)
* **Clone URL**: 
  ```bash
  git clone https://github.com/rashedulalbab253/llm-as-a-judge-system.git
  cd llm-as-a-judge-system
  ```

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
python run_pipeline.py --mock          # Run the entire pipeline in mock/demo mode without API keys
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

## Evaluation Results & Visualizations

Running the evaluation pipeline automatically generates high-resolution charts in the `plots/` directory. Below are the visual results of the calibrated evaluation:

### 1. G-Eval Score Distributions

This histogram shows the distribution of scores across the four criteria: Coherence, Factual Accuracy, Appropriate Tone, and Safety.

![G-Eval Score Distributions](file:///D:/test-project/llm-as-a-judge/plots/score_distributions.png)

### 2. Pass/Fail Rates by Criterion

A horizontal analysis representing the percentage of responses that met or exceeded the passing threshold of 0.5.

![Pass/Fail Rates](file:///D:/test-project/llm-as-a-judge/plots/pass_fail_rates.png)

### 3. Inter-Criteria Correlation Heatmap

Highlights correlations between the evaluated criteria, indicating how independent each metric is during judge evaluation.

![Correlation Heatmap](file:///D:/test-project/llm-as-a-judge/plots/correlation_heatmap.png)

### 4. Human-Judge Calibration Chart

A scatter plot comparing human annotation scores directly to G-Eval automated scores. The alignment along the diagonal demonstrates the quality of the judge's calibration.

![Human vs G-Eval Judge Agreement](file:///D:/test-project/llm-as-a-judge/plots/agreement_scatter.png)

### 5. Boxplots distribution

Shows the score density, median, and outliers for all criteria.

![Score Boxplots](file:///D:/test-project/llm-as-a-judge/plots/score_boxplots.png)

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


## License

MIT License - Copyright (c) 2026 Rashedul Albab
