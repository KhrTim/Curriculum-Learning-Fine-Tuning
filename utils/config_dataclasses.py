from dataclasses import asdict, dataclass, field
from enum import Flag, auto
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration with HuggingFace path and short name."""

    huggingface_path: str
    short_name: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""

    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 3e-4
    gradient_accumulation_steps: int = 4

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class Strategy(Flag):
    BASELINE_FINETUNING = auto()
    CURRICULUM_SIMPLE = auto()
    CURRICULUM_GOOD = auto()


@dataclass
class ExperimentConfig:
    """Experiment configuration with strategy and dataset limits."""

    wb_project_name: str = "math-curriculum-learning"
    strategy: Strategy = Strategy.BASELINE_FINETUNING
    max_train_samples: Optional[int] = None
    max_test_samples: Optional[int] = 100
    experiment_base_dir: Path = field(
        default_factory=lambda: Path("./experiment_results")
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
