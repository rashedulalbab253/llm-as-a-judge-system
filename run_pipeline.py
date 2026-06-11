"""
LLM-as-a-Judge: Full Pipeline Runner

Orchestrates the complete evaluation pipeline:
  Step 1: Load Alpaca Eval dataset from HuggingFace
  Step 2: Generate outputs from weak model (GPT-3.5-Turbo)
  Step 3: Evaluate outputs using G-Eval with GPT-4 judge
  Step 4: Generate human annotation template (or compute agreement)
  Step 5: Visualize results and generate reports

Usage:
    python run_pipeline.py                    # Run all steps
    python run_pipeline.py --step 1           # Run specific step
    python run_pipeline.py --step 3 --step 5  # Run steps 3 and 5
    python run_pipeline.py --skip-generate    # Skip output generation (use cached)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure standard output uses UTF-8 if available to handle emojis, but write ASCII to stdout log to be completely safe.
# We will configure FileHandler with UTF-8 encoding.
log_file = Path(__file__).parent / "pipeline.log"
file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
stdout_handler = logging.StreamHandler(sys.stdout)

# Custom formatter or custom stdout stream mapping if needed, but easier to just use ASCII symbols in log strings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[stdout_handler, file_handler]
)
logger = logging.getLogger(__name__)


def run_step_1():
    """Load and prepare the Alpaca Eval dataset."""
    from src.data_loader import main as load_data
    return load_data()


def run_step_2():
    """Generate weak model outputs."""
    from src.generate_outputs import main as generate
    return generate()


def run_step_3():
    """Run G-Eval evaluation."""
    from src.evaluate_outputs import main as evaluate
    return evaluate()


def run_step_4():
    """Human validation / annotation."""
    from src.human_validation import main as validate
    return validate()


def run_step_5():
    """Visualize results."""
    from src.visualize_results import main as visualize
    return visualize()


def run_mock_data():
    """Generate mock outputs, G-Eval scores, and completed human annotations."""
    from src.generate_mock_data import generate_mock_data
    return generate_mock_data()


STEPS = {
    1: ("Load Alpaca Eval Dataset", run_step_1),
    2: ("Generate Weak Model Outputs", run_step_2),
    3: ("Run G-Eval Evaluation (GPT-4 Judge)", run_step_3),
    4: ("Human Validation & Agreement", run_step_4),
    5: ("Visualize Results & Reports", run_step_5),
}


def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1  Load Alpaca Eval dataset from HuggingFace (805 examples)
  2  Generate outputs from weak model (GPT-3.5-Turbo)
  3  Evaluate outputs with G-Eval using GPT-4 as judge
  4  Generate human annotation template / compute agreement
  5  Visualize results and generate summary reports

Examples:
  python run_pipeline.py               # Run entire pipeline
  python run_pipeline.py --step 1      # Load dataset only
  python run_pipeline.py --step 3 5    # Evaluate and visualize
  python run_pipeline.py --mock        # Run the entire pipeline in mock/demo mode without API keys
        """
    )
    parser.add_argument(
        "--step", nargs="+", type=int, choices=[1, 2, 3, 4, 5],
        help="Run specific step(s) only. Default: run all."
    )
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Skip step 2 (use cached outputs from previous run)"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Run pipeline with mock data (loads dataset, simulates generations/evaluations/annotations, computes agreement, and plots results)"
    )
    
    args = parser.parse_args()
    
    # Banner
    logger.info("+" + "-" * 58 + "+")
    logger.info("|     LLM-as-a-Judge Evaluation Pipeline                   |")
    logger.info("|     Alpaca Eval + DeepEval G-Eval + GPT-4 Judge          |")
    logger.info("+" + "-" * 58 + "+")
    
    if args.mock:
        logger.info("Running pipeline in MOCK/DEMO mode...")
        steps_to_run = [1, "mock_data", 4, 5]
        custom_steps = {
            1: ("Load Alpaca Eval Dataset", run_step_1),
            "mock_data": ("Generate Mock Outputs & G-Eval Scores", run_mock_data),
            4: ("Human Validation & Agreement", run_step_4),
            5: ("Visualize Results & Reports", run_step_5),
        }
    else:
        # Determine which steps to run
        if args.step:
            steps_to_run = sorted(args.step)
        else:
            steps_to_run = [1, 2, 3, 4, 5]
        
        if args.skip_generate and 2 in steps_to_run:
            steps_to_run.remove(2)
        custom_steps = STEPS
        
    logger.info(f"Steps to run: {steps_to_run}\n")
    
    pipeline_start = time.time()
    results = {}
    
    for step_num in steps_to_run:
        name, func = custom_steps[step_num]
        logger.info(f"\n>>> STEP {step_num}: {name}")
        logger.info("-" * 60)
        
        step_start = time.time()
        try:
            result = func()
            elapsed = time.time() - step_start
            results[step_num] = {"status": "success", "time": elapsed}
            logger.info(f"[SUCCESS] Step {step_num} completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - step_start
            results[step_num] = {"status": "failed", "error": str(e), "time": elapsed}
            logger.error(f"[FAILED] Step {step_num} failed after {elapsed:.1f}s: {e}")
            
            # Don't continue if a prerequisite step fails
            if steps_to_run.index(step_num) < len(steps_to_run) - 1:
                logger.error("Stopping pipeline due to failure in prerequisite step.")
                break
    
    # Final summary
    total_time = time.time() - pipeline_start
    logger.info(f"\n" + "=" * 60)
    logger.info(f"Pipeline Complete -- Total time: {total_time:.1f}s")
    logger.info("=" * 60)
    for s in steps_to_run:
        r = results.get(s)
        if not r:
            continue
        status = "[PASS]" if r["status"] == "success" else "[FAIL]"
        step_name = custom_steps[s][0]
        logger.info(f"  {status} Step {s}: {step_name} ({r['time']:.1f}s)")
    logger.info("")


if __name__ == "__main__":
    main()
