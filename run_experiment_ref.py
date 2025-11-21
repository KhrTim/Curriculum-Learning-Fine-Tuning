import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

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

CURRICULUM_METHODS = [Strategy.CURRICULUM_SIMPLE, Strategy.CURRICULUM_GOOD]

# Default experiment parameters
DEFAULT_MAX_TEST_SAMPLES = 512
DEFAULT_MAX_TRAIN_SAMPLES = None
DEFAULT_EXPERIMENT_BASE_DIR = Path("./experiment_results")
DEFAULT_EXPERIMENT_BASE_DIR.mkdir(exist_ok=True)

# TESTING_MODE = False
TESTING_MODE = True


# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

if TESTING_MODE:
    logging.warning(
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!  <TEST MODE>  !!!!!!!!!!!!!!!!!!!!!!!!!!!"
    )

# ==================== HELPERS ====================


def create_exp_config(
    strategy: Strategy,
    base_dir: Path = DEFAULT_EXPERIMENT_BASE_DIR,
    max_train_samples: Optional[int] = DEFAULT_MAX_TRAIN_SAMPLES,
    max_test_samples: Optional[int] = DEFAULT_MAX_TEST_SAMPLES,
    testing: bool = TESTING_MODE,
) -> ExperimentConfig:
    """
    Build an ExperimentConfig object.
    """
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
) -> Dict:
    """
    Create and run a single experiment instance.
    """
    logging.info("=" * 80)
    logging.info(f"Running {str(strategy).upper()} experiment for model: {model_key}")
    logging.info("=" * 80)

    model_cfg = ModelConfig(short_name=model_key, huggingface_path=model_name)
    train_cfg = TrainingConfig(**DEFAULT_TRAINING_CONFIG)
    exp_cfg = create_exp_config(strategy)

    experiment = Experiment(model_cfg, train_cfg, exp_cfg)
    results = experiment.run()
    return results


def run_all_experiments(
    selected_models: Optional[List[str]] = None,
    selected_strategies: Optional[List[str]] = None,
) -> Dict[str, Dict[str, dict]]:
    """
    Run a complete experimental suite for the selected models and strategies.
    """
    selected_models = selected_models or list(MODELS.keys())
    selected_strategies = selected_strategies or ["baseline", "curriculum"]

    all_results: Dict[str, Dict[str, dict]] = {}

    for model_key in selected_models:
        model_name = MODELS[model_key.lower()]
        all_results[model_key] = {}

        # # Baseline
        if "baseline" in selected_strategies or "all" in selected_strategies:
            all_results[model_key]["baseline"] = run_experiment(
                model_key, model_name, strategy=Strategy.BASELINE_FINETUNING
            )

        # Curriculum
        # if "curriculum" in selected_strategies or "all" in selected_strategies:
        #     for method in CURRICULUM_METHODS:
        #         key = f"curriculum_{method.value}"
        #         all_results[model_key][key] = run_experiment(model_key, model_name, strategy=method)

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


def main():
    parser = argparse.ArgumentParser(description="Run GSM8K fine-tuning experiments")

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="Which models to run",
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["baseline", "curriculum", "all"],
        default=["all"],
        help="Training strategies to run",
    )

    parser.add_argument(
        "--single-experiment",
        choices=["baseline", "curriculum"],
        help="Run only one experiment (for quick testing)",
    )

    args = parser.parse_args()

    # Normalize arguments
    selected_models = list(MODELS.keys()) if "all" in args.models else args.models
    selected_strategies = (
        ["baseline", "curriculum"] if "all" in args.strategies else args.strategies
    )

    # Quick experiment mode
    if args.single_experiment:
        logging.info("Running single-experiment mode...")
        model_key = "phi2"
        model_name = MODELS[model_key]

        if args.single_experiment == "baseline":
            run_experiment(model_key, model_name, strategy=Strategy.BASELINE_FINETUNING)

        elif args.single_experiment == "curriculum":
            for method in CURRICULUM_METHODS:
                run_experiment(model_key, model_name, strategy=method)
        return

    # Full suite
    run_all_experiments(selected_models, selected_strategies)


if __name__ == "__main__":
    main()
