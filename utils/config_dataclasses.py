from dataclasses import asdict, dataclass, field
from enum import Flag, auto
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """Holds model name mappings."""

    huggingface_path: str
    short_name: str

    def to_dict(self) -> dict:
        """Return a dictionary representation of the model config."""
        return asdict(self)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 3e-4
    gradient_accumulation_steps: int = 4

    def to_dict(self) -> dict:
        """Return a dictionary representation of the training config."""
        return asdict(self)


class Strategy(Flag):
    BASELINE_FINETUNING = auto()
    CURRICULUM_SIMPLE = auto()
    CURRICULUM_GOOD = auto()


@dataclass
class ExperimentConfig:
    """Experiment configuration."""

    wb_project_name: str = "math-curriculum-learning"
    strategy: Strategy = Strategy.BASELINE_FINETUNING
    max_train_samples: Optional[int] = None
    max_test_samples: Optional[int] = 100
    experiment_base_dir: Path = field(
        default_factory=lambda: Path("./experiment_results")
    )

    def to_dict(self) -> dict:
        """Return a dictionary representation of the experiment config."""
        return asdict(self)
