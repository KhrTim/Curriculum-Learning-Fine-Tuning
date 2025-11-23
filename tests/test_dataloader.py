"""Tests for dataloader module."""

import unittest

from utils.config_dataclasses import ExperimentConfig, Strategy
from utils.dataloader import DatasetLoader


class TestDatasetLoader(unittest.TestCase):
    """Test DatasetLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.baseline_config = ExperimentConfig(
            strategy=Strategy.BASELINE_FINETUNING,
            max_train_samples=10,
            max_test_samples=5,
        )
        self.curriculum_config = ExperimentConfig(
            strategy=Strategy.CURRICULUM_SIMPLE,
            max_train_samples=15,
            max_test_samples=5,
        )

    def test_problem_complexity(self):
        """Test problem complexity calculation."""
        question = "Add 5 plus 3 and multiply by 2"
        complexity = DatasetLoader.problem_complexity(question)
        self.assertIsInstance(complexity, float)
        self.assertGreater(complexity, 0)

    def test_problem_complexity_simple(self):
        """Test problem complexity for simple addition."""
        question = "What is 2 plus 2?"
        complexity = DatasetLoader.problem_complexity(question)
        self.assertGreater(complexity, 0)

    def test_difficulty_score(self):
        """Test difficulty score calculation."""
        loader = DatasetLoader(self.baseline_config)
        example = {
            "question": "John has 5 apples. He buys 3 more. How many does he have?",
            "answer": "John starts with 5 apples.\nHe buys 3 more.\n5 + 3 = 8\n#### 8",
        }
        score = loader.difficulty_score(example)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0)

    def test_format_example_gsm8k(self):
        """Test GSM8K example formatting."""
        loader = DatasetLoader(self.baseline_config)
        example = {"question": "What is 2 + 2?", "answer": "4"}
        formatted = loader.format_example_gsm8k(example)
        self.assertIn("prompt", formatted)
        self.assertIn("completion", formatted)
        self.assertIn("Question:", formatted["prompt"])
        self.assertEqual(formatted["completion"], "4")

    def test_as_dataset(self):
        """Test dataset creation from lists."""
        easy = [{"question": "1+1?", "answer": "2"}]
        normal = [{"question": "2+2?", "answer": "4"}]
        hard = [{"question": "3+3?", "answer": "6"}]

        result = DatasetLoader._as_dataset(easy, normal, hard)
        self.assertIn("easy", result)
        self.assertIn("normal", result)
        self.assertIn("difficult", result)


if __name__ == "__main__":
    unittest.main()
