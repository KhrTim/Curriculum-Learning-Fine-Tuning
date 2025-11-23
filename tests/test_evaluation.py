"""Tests for evaluation module."""

import unittest

from utils.evaluation import Evaluation


class TestEvaluation(unittest.TestCase):
    """Test Evaluation class."""

    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = Evaluation()

    def test_extract_answer_gsm8k_format(self):
        """Test answer extraction from GSM8K format."""
        text = "Let's solve this step by step.\n5 + 3 = 8\n#### 8"
        answer = Evaluation.extract_answer(text)
        self.assertEqual(answer, "8")

    def test_extract_answer_negative_number(self):
        """Test answer extraction with negative numbers."""
        text = "The result is negative.\n#### -15"
        answer = Evaluation.extract_answer(text)
        self.assertEqual(answer, "-15")

    def test_extract_answer_decimal(self):
        """Test answer extraction with decimals."""
        text = "The answer is 3.5\n#### 3.5"
        answer = Evaluation.extract_answer(text)
        self.assertEqual(answer, "3.5")

    def test_extract_answer_fallback(self):
        """Test fallback to last number when no #### marker."""
        text = "The calculation gives us 42 as the final answer."
        answer = Evaluation.extract_answer(text)
        self.assertEqual(answer, "42")

    def test_extract_answer_none_input(self):
        """Test extraction with None input."""
        answer = Evaluation.extract_answer(None)
        self.assertIsNone(answer)

    def test_extract_answer_empty_string(self):
        """Test extraction with empty string."""
        answer = Evaluation.extract_answer("")
        self.assertIsNone(answer)

    def test_extract_answer_no_numbers(self):
        """Test extraction when no numbers present."""
        text = "This text has no numbers at all."
        answer = Evaluation.extract_answer(text)
        self.assertIsNone(answer)

    def test_extract_answer_trailing_period(self):
        """Test answer extraction removes trailing period."""
        text = "#### 100."
        answer = Evaluation.extract_answer(text)
        self.assertEqual(answer, "100")


if __name__ == "__main__":
    unittest.main()
