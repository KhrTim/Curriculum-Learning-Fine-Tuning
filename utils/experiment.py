import json
import logging
from datetime import datetime
from pathlib import Path

import wandb
from utils.config_dataclasses import (
    ExperimentConfig,
    ModelConfig,
    Strategy,
    TrainingConfig,
)
from utils.dataloader import DatasetLoader
from utils.evaluation import Evaluation
from utils.utils import EnumEncoder, fine_tune

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


class Experiment:
    """Manages experiment lifecycle including training, evaluation, and logging."""

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        experiment_config: ExperimentConfig,
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.experiment_config = experiment_config

        experiment_config.experiment_base_dir.mkdir(exist_ok=True)
        self.evaluator = Evaluation()

    # -------------------------------------------------------------------------
    #                              DIRECTORY MGMT
    # -------------------------------------------------------------------------

    def get_experiment_dir(self) -> Path:
        """Generate timestamped output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy = self.experiment_config.strategy

        dir_name = f"{strategy}_{self.model_config.short_name}_gsm8k_{timestamp}"

        return self.experiment_config.experiment_base_dir / dir_name

    def save_experiment_metadata(self, output_dir: Path, metadata: dict) -> None:
        """Save experiment metadata to JSON file."""
        (output_dir / "experiment_metadata.json").write_text(
            json.dumps(metadata, indent=2, cls=EnumEncoder)
        )

    def create_metadata(self, dataset_info: dict) -> dict:
        """Merge dataset info and configs into metadata."""
        return {
            "strategy": self.experiment_config.strategy,
            "model": self.model_config.short_name,
            "dataset": "gsm8k",
            "config": self.training_config.to_dict(),
            "timestamp": datetime.now().isoformat(),
            **dataset_info,
        }

    # -------------------------------------------------------------------------
    #                               SETUP
    # -------------------------------------------------------------------------

    def setup_experiment(self) -> Path:
        """Prepare output directory."""
        out_dir = self.get_experiment_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

        logging.info("Loading GSM8K dataset...")
        if self.experiment_config.max_test_samples:
            logging.info(
                f"Using {self.experiment_config.max_test_samples} test samples."
            )

        return out_dir

    def initialize_wandb_run(self, metadata: dict, mode: str = None):
        """Initialize W&B run with metadata."""
        run_name = f"{self.experiment_config.strategy}_{self.model_config.short_name}{'_' + mode if mode else ''}"

        return wandb.init(
            project=self.experiment_config.wb_project_name,
            name=run_name,
            reinit=True,
            config={**metadata, **self.training_config.to_dict()},
        )

    def finalize_experiment(self, output_dir: Path, results: dict) -> None:
        """Save results and finish W&B logging."""
        filename = (
            f"{self.experiment_config.strategy}_results.json"
            if self.experiment_config.strategy != "baseline"
            else "evaluation_results.json"
        )

        with open(output_dir / filename, "w") as f:
            json.dump(results, f, indent=2, cls=EnumEncoder)

        if "exact_match_pct" in results:
            wandb.log({"final_exact_match": results["exact_match_pct"]})

        wandb.finish()

    # -------------------------------------------------------------------------
    #                          PUBLIC MAIN FUNCTION
    # -------------------------------------------------------------------------

    def run(self) -> dict:
        """Run experiment with dataset loading, training, and evaluation."""
        output_dir = self.setup_experiment()

        dataset_splits, dataset_info = DatasetLoader(
            self.experiment_config
        ).get_formatted()

        metadata = self.create_metadata(dataset_info)
        self.save_experiment_metadata(output_dir, metadata)

        wandb_run = self.initialize_wandb_run(metadata)

        strategy = self.experiment_config.strategy
        if strategy == Strategy.BASELINE_FINETUNING:
            results = self.run_baseline_experiment(
                dataset_splits, output_dir, wandb_run.name
            )
        else:
            results = self.run_curriculum_experiment(
                dataset_splits, output_dir, wandb_run
            )

        self.finalize_experiment(output_dir, results)
        return results

    # -------------------------------------------------------------------------
    #                           BASELINE TRAINING
    # -------------------------------------------------------------------------

    def run_baseline_experiment(
        self, dataset_splits: dict, output_dir: Path, run_name: str
    ) -> dict:
        """Train model on full dataset."""
        logging.info(f"Running BASELINE: {self.model_config.short_name}")

        train_config = {
            k: v for k, v in self.training_config.to_dict().items() if k != "use_lora"
        }

        model_path = fine_tune(
            model_name=self.model_config.huggingface_path,
            model_config=self.model_config,
            train_dataset=dataset_splits["train"],
            output_dir=str(output_dir),
            wandb_run_name=run_name,
            from_scratch=True,
            **train_config,
        )

        logging.info("Evaluating baseline model...")

        results = self.evaluator.evaluate_model(
            model_path, dataset_splits["test"], batch_size=16, max_new_tokens=512,
            save_samples_dir=Path("tmp_results")
        )

        logging.info("Baseline completed.")
        logging.info(f"Exact Match: {results['exact_match_pct']:.2f}%")

        return results

    # -------------------------------------------------------------------------
    #                           CURRICULUM TRAINING
    # -------------------------------------------------------------------------

    def run_curriculum_experiment(
        self, dataset_splits: dict, output_dir: Path, wandb_run
    ) -> dict:
        """Train progressively from easy to hard."""
        logging.info(f"Running CURRICULUM: {self.model_config.short_name}")

        checkpoint = None
        stage_results = {}

        for idx, (stage_name, train_dataset) in enumerate(
            dataset_splits["train_splits"].items()
        ):
            logging.info(
                f"[Stage {idx + 1}] {stage_name} ({len(train_dataset)} samples)"
            )

            stage_dir = output_dir / f"stage_{idx + 1}_{stage_name}"
            stage_dir.mkdir(exist_ok=True)

            train_config = {
                k: v
                for k, v in self.training_config.to_dict().items()
                if k != "use_lora"
            }

            model_path = fine_tune(
                model_name=(
                    self.model_config.huggingface_path
                    if checkpoint is None
                    else checkpoint
                ),
                model_config=self.model_config,
                train_dataset=train_dataset,
                output_dir=str(stage_dir),
                wandb_run_name=wandb_run.name,
                from_scratch=checkpoint is None,
                **train_config,
            )

            # Merge LoRA before next stage
            logging.info(f"Merging LoRA for stage: {stage_name}")

            checkpoint = model_path

            # Evaluate
            logging.info(f"Evaluating after stage: {stage_name}")
            results = self.evaluator.evaluate_model(
                model_path, dataset_splits["test"], batch_size=16, max_new_tokens=512,
                save_samples_dir=Path("tmp_results")
            )
            stage_results[stage_name] = results

            wandb.log({"eval/exact_match": results["exact_match_pct"]})

        logging.info("Curriculum completed.")
        return stage_results
