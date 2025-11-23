import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils.config_dataclasses import (
    ExperimentConfig,
    ModelConfig,
    Strategy,
    TrainingConfig,
)
from utils.experiment import Experiment
from utils.utils import EnumEncoder

# ==================== CONFIGURATION ====================

MODELS = {
    "phi2": "microsoft/phi-2",
    "smollm2": "HuggingFaceTB/SmolLM2-135M",
}

DEFAULT_TRAINING_CONFIG = {
    "num_epochs": 2,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "gradient_accumulation_steps": 4,
}

# CURRICULUM_METHODS = [Strategy.CURRICULUM_SIMPLE, Strategy.CURRICULUM_GOOD]
CURRICULUM_METHODS = [Strategy.CURRICULUM_GOOD]


# Default experiment parameters
DEFAULT_MAX_TEST_SAMPLES = 512
DEFAULT_MAX_TRAIN_SAMPLES = None
DEFAULT_EXPERIMENT_BASE_DIR = Path("./experimental_results")
DEFAULT_EXPERIMENT_BASE_DIR.mkdir(exist_ok=True)

TESTING_MODE = False  # Default to production mode, override with --test flag


# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


# ==================== HELPERS ====================


def create_exp_config(
    strategy: Strategy,
    base_dir: Path = DEFAULT_EXPERIMENT_BASE_DIR,
    max_train_samples: Optional[int] = DEFAULT_MAX_TRAIN_SAMPLES,
    max_test_samples: Optional[int] = DEFAULT_MAX_TEST_SAMPLES,
    testing: bool = TESTING_MODE,
) -> ExperimentConfig:
    """Build ExperimentConfig with strategy-specific settings."""
    if testing:
        max_train_samples = 100
        max_test_samples = 30
        base_dir = Path("./tmp_results")
        base_dir.mkdir(exist_ok=True)

    wb_project = "math-curriculum-learning-additional"
    if testing:
        wb_project += "-test"

    return ExperimentConfig(
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        experiment_base_dir=base_dir,
        strategy=strategy,
        wb_project_name=wb_project,
    )


def run_experiment(
    model_key: str,
    model_name: str,
    strategy: Strategy,
    testing: bool = False,
) -> Dict:
    """Run single experiment with specified model and strategy."""
    logging.info("=" * 80)
    logging.info(f"Running {str(strategy).upper()} experiment for model: {model_key}")
    if testing:
        logging.warning("TEST MODE ENABLED - Using limited samples")
    logging.info("=" * 80)

    model_cfg = ModelConfig(short_name=model_key, huggingface_path=model_name)
    train_cfg = TrainingConfig(**DEFAULT_TRAINING_CONFIG)
    exp_cfg = create_exp_config(strategy, testing=testing)

    experiment = Experiment(model_cfg, train_cfg, exp_cfg)
    results = experiment.run()
    return results


def run_all_experiments(
    selected_models: Optional[List[str]] = None,
    selected_strategies: Optional[List[str]] = None,
    testing: bool = False,
) -> Dict[str, Dict[str, dict]]:
    """Run experiments for selected models and strategies."""
    selected_models = selected_models or list(MODELS.keys())
    selected_strategies = selected_strategies or ["baseline", "curriculum"]

    all_results: Dict[str, Dict[str, dict]] = {}

    for model_key in selected_models:
        model_name = MODELS[model_key.lower()]
        all_results[model_key] = {}

        if "baseline" in selected_strategies or "all" in selected_strategies:
            all_results[model_key]["baseline"] = run_experiment(
                model_key,
                model_name,
                strategy=Strategy.BASELINE_FINETUNING,
                testing=testing,
            )

        if "curriculum" in selected_strategies or "all" in selected_strategies:
            for method in CURRICULUM_METHODS:
                key = f"curriculum_{method.name.lower()}"
                all_results[model_key][key] = run_experiment(
                    model_key, model_name, strategy=method, testing=testing
                )

    # Save summary
    summary_file = (
        DEFAULT_EXPERIMENT_BASE_DIR
        / f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, cls=EnumEncoder)

    logging.info("=" * 80)
    logging.info("ALL EXPERIMENTS COMPLETED!")
    logging.info(f"Summary saved to: {summary_file}")
    logging.info("=" * 80)

    return all_results


# ==================== ENTRYPOINT ====================


def main() -> None:
    """Entry point for GSM8K fine-tuning experiments."""
    parser = argparse.ArgumentParser(
        description="Run GSM8K fine-tuning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline fine-tuning with phi2 model in test mode
  python run_experiments.py --model phi2 --strategy baseline --test

  # Run all strategies with smollm2 model (production mode)
  python run_experiments.py --model smollm2 --strategy all

  # Run curriculum learning only with all models
  python run_experiments.py --model all --strategy curriculum
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        choices=list(MODELS.keys()) + ["all"],
        default="all",
        help="Which model to use (default: all)",
    )

    parser.add_argument(
        "--strategy",
        "-s",
        choices=["baseline", "curriculum", "all"],
        default="all",
        help="Training strategy to run (default: all)",
    )

    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Enable test mode (uses limited samples: 100 train, 30 test)",
    )

    args = parser.parse_args()

    # Log configuration
    logging.info("=" * 80)
    logging.info("EXPERIMENT CONFIGURATION")
    logging.info(f"Model: {args.model}")
    logging.info(f"Strategy: {args.strategy}")
    logging.info(f"Test Mode: {args.test}")
    logging.info("=" * 80)

    # Normalize arguments
    selected_models = list(MODELS.keys()) if args.model == "all" else [args.model]
    selected_strategies = (
        ["baseline", "curriculum"] if args.strategy == "all" else [args.strategy]
    )

    # Run experiments
    run_all_experiments(selected_models, selected_strategies, testing=args.test)


if __name__ == "__main__":
    main()
