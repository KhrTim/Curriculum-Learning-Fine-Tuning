"""Tests for config_dataclasses module."""

import unittest
from pathlib import Path

from utils.config_dataclasses import (
    ExperimentConfig,
    ModelConfig,
    Strategy,
    TrainingConfig,
)


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig dataclass."""

    def test_model_config_creation(self):
        """Test ModelConfig initialization."""
        config = ModelConfig(huggingface_path="microsoft/phi-2", short_name="phi2")
        self.assertEqual(config.huggingface_path, "microsoft/phi-2")
        self.assertEqual(config.short_name, "phi2")

    def test_model_config_to_dict(self):
        """Test ModelConfig conversion to dictionary."""
        config = ModelConfig(huggingface_path="microsoft/phi-2", short_name="phi2")
        result = config.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual(result["huggingface_path"], "microsoft/phi-2")
        self.assertEqual(result["short_name"], "phi2")


class TestTrainingConfig(unittest.TestCase):
    """Test TrainingConfig dataclass."""

    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        self.assertEqual(config.num_epochs, 3)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.learning_rate, 3e-4)
        self.assertEqual(config.gradient_accumulation_steps, 4)

    def test_training_config_custom_values(self):
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(
            num_epochs=5,
            batch_size=8,
            learning_rate=1e-4,
            gradient_accumulation_steps=2,
        )
        self.assertEqual(config.num_epochs, 5)
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.gradient_accumulation_steps, 2)

    def test_training_config_to_dict(self):
        """Test TrainingConfig conversion to dictionary."""
        config = TrainingConfig()
        result = config.to_dict()
        self.assertIsInstance(result, dict)
        self.assertIn("num_epochs", result)
        self.assertIn("batch_size", result)


class TestStrategy(unittest.TestCase):
    """Test Strategy enum."""

    def test_strategy_values(self):
        """Test Strategy enum has expected values."""
        self.assertIsNotNone(Strategy.BASELINE_FINETUNING)
        self.assertIsNotNone(Strategy.CURRICULUM_SIMPLE)
        self.assertIsNotNone(Strategy.CURRICULUM_GOOD)


class TestExperimentConfig(unittest.TestCase):
    """Test ExperimentConfig dataclass."""

    def test_experiment_config_defaults(self):
        """Test ExperimentConfig default values."""
        config = ExperimentConfig()
        self.assertEqual(config.wb_project_name, "math-curriculum-learning")
        self.assertEqual(config.strategy, Strategy.BASELINE_FINETUNING)
        self.assertIsNone(config.max_train_samples)
        self.assertEqual(config.max_test_samples, 100)
        self.assertIsInstance(config.experiment_base_dir, Path)

    def test_experiment_config_custom_values(self):
        """Test ExperimentConfig with custom values."""
        config = ExperimentConfig(
            wb_project_name="test-project",
            strategy=Strategy.CURRICULUM_SIMPLE,
            max_train_samples=500,
            max_test_samples=200,
        )
        self.assertEqual(config.wb_project_name, "test-project")
        self.assertEqual(config.strategy, Strategy.CURRICULUM_SIMPLE)
        self.assertEqual(config.max_train_samples, 500)
        self.assertEqual(config.max_test_samples, 200)

    def test_experiment_config_to_dict(self):
        """Test ExperimentConfig conversion to dictionary."""
        config = ExperimentConfig()
        result = config.to_dict()
        self.assertIsInstance(result, dict)
        self.assertIn("wb_project_name", result)
        self.assertIn("strategy", result)


if __name__ == "__main__":
    unittest.main()
