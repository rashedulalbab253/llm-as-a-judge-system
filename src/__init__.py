"""
LLM-as-a-Judge Evaluation Pipeline

A production-grade automated evaluation system that uses GPT-4 (via DeepEval G-Eval)
to evaluate outputs from a weaker model on custom rubrics:
  - Coherence
  - Factual Accuracy
  - Appropriate Tone
  - Safety & Harmlessness

Uses the Alpaca Eval dataset (805 instruction-following examples) and validates
judge reliability by computing human-judge agreement rates.
"""

__version__ = "1.0.0"
