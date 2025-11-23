import re
from collections import OrderedDict

from datasets import Dataset, load_dataset

from utils.config_dataclasses import ExperimentConfig, Strategy


class DatasetLoader:
    """Load, preprocess, and optionally curriculum-split GSM8K dataset."""

    def __init__(self, config: ExperimentConfig):
        self.max_test_samples = config.max_test_samples
        self.max_train_samples = config.max_train_samples
        self.strategy = config.strategy
        self.dataset_info = None

        # Assign split method dynamically

        if self.strategy != Strategy.BASELINE_FINETUNING:
            self.split_gsm8k_by_difficulty = {
                Strategy.CURRICULUM_SIMPLE: self._split_by_answer_length,
                Strategy.CURRICULUM_GOOD: self._split_by_weighted_score,
            }[self.strategy]

    # -------------------------------------------------------------------------
    # ------------------------------ UTILITIES --------------------------------
    # -------------------------------------------------------------------------

    @staticmethod
    def problem_complexity(question: str) -> float:
        """Estimate complexity by counting weighted math operations."""
        operations = {
            "add": r"\b(?:add|plus|sum|more|increase|total|combined|\+)\b",
            "sub": r"\b(?:subtract|minus|less|fewer|decrease|difference|remaining|\-)\b",
            "mul": r"\b(?:multiply|times|product|each|per|\*|\×)\b",
            "div": r"\b(?:divide|split|share|distribute|half|quarter|\/)\b",
        }
        q = question.lower()

        add = len(re.findall(operations["add"], q))
        sub = len(re.findall(operations["sub"], q))
        mul = len(re.findall(operations["mul"], q))
        div = len(re.findall(operations["div"], q))

        return max(add + sub + 1.5 * mul + 1.5 * div, 1)

    def difficulty_score(self, ex: dict) -> float:
        """Calculate difficulty score from steps, operations, and numbers."""
        steps = sum(1 for line in ex["answer"].split("\n") if line.strip())
        nums = len(re.findall(r"\d+", ex["question"]))
        ops = self.problem_complexity(ex["question"])
        return steps + ops * nums

    @staticmethod
    def _as_dataset(easy: list, normal: list, hard: list) -> OrderedDict:
        """Convert difficulty lists to ordered dataset dictionary."""
        return OrderedDict(
            [
                ("easy", Dataset.from_list(easy)),
                ("normal", Dataset.from_list(normal)),
                ("difficult", Dataset.from_list(hard)),
            ]
        )

    # -------------------------------------------------------------------------
    # ------------------------------ FORMATTING -------------------------------
    # -------------------------------------------------------------------------

    def format_example_gsm8k(self, example: dict) -> dict:
        """Format GSM8K example into prompt-completion pairs."""
        return {
            "prompt": f"Question: {example['question']}\nAnswer:",
            "completion": example["answer"],
        }

    def _format_splits(self, splits: dict) -> dict:
        """Apply GSM8K formatting to training splits."""
        if "train" in splits:
            splits["train"] = splits["train"].map(
                self.format_example_gsm8k,
                remove_columns=["answer", "question"],
                num_proc=16,
            )
            return splits

        formatted_splits = {
            diff: ds.map(
                self.format_example_gsm8k,
                remove_columns=["answer", "question"],
                num_proc=16,
            )
            for diff, ds in splits["train_splits"].items()
        }
        return {
            "name": splits["name"],
            "train_splits": formatted_splits,
            "test": splits["test"],
        }

    # -------------------------------------------------------------------------
    # ------------------------------ LOADING ----------------------------------
    # -------------------------------------------------------------------------

    def _load_split(self, split_name: str, max_samples: int | None) -> Dataset:
        """Load GSM8K split with optional sample limit."""
        ds = load_dataset("gsm8k", "main", split=split_name)
        return (
            ds.select(range(max_samples))
            if max_samples and len(ds) > max_samples
            else ds
        )

    def get_train_test_base(self) -> dict:
        """Load train and test splits."""
        train = self._load_split("train", self.max_train_samples)
        test = self._load_split("test", self.max_test_samples)

        self.dataset_info = {
            "train_samples": len(train),
            "test_samples": len(test),
        }

        return {"name": "gsm8k", "train": train, "test": test}

    # -------------------------------------------------------------------------
    # --------------------------- SPLITTING METHODS ---------------------------
    # -------------------------------------------------------------------------

    def _split_by_answer_length(self, dataset: Dataset) -> OrderedDict:
        """Split dataset by answer length into easy, normal, hard."""
        easy, normal, hard = [], [], []
        for ex in dataset:
            L = len(ex["answer"])
            (easy if L < 168 else normal if L <= 280 else hard).append(ex)
        return self._as_dataset(easy, normal, hard)

    def _split_by_weighted_score(self, dataset: Dataset) -> OrderedDict:
        """Split dataset by weighted difficulty score."""
        scored = sorted(
            ((ex, self.difficulty_score(ex)) for ex in dataset), key=lambda x: x[1]
        )
        n = len(scored)
        easy = [ex for ex, _ in scored[: n // 3]]
        normal = [ex for ex, _ in scored[n // 3 : 2 * n // 3]]
        hard = [ex for ex, _ in scored[2 * n // 3 :]]
        return self._as_dataset(easy, normal, hard)

    def get_curriculum_splits(self) -> dict:
        """Get curriculum splits ordered by difficulty."""
        splits = self.get_train_test_base()
        train = splits["train"]

        diff_splits = self.split_gsm8k_by_difficulty(train)

        self.dataset_info = {
            "curriculum_stages": list(diff_splits.keys()),
            "stage_sizes": {k: len(v) for k, v in diff_splits.items()},
            "test_samples": len(splits["test"]),
        }

        return {"name": "gsm8k", "train_splits": diff_splits, "test": splits["test"]}

    # -------------------------------------------------------------------------
    # ------------------------------ PUBLIC API --------------------------------
    # -------------------------------------------------------------------------

    def get_original_format(self) -> tuple:
        """Get dataset in original format based on strategy."""
        if self.strategy == Strategy.BASELINE_FINETUNING:
            return self.get_train_test_base(), self.dataset_info
        return self.get_curriculum_splits(), self.dataset_info

    def get_formatted(self) -> tuple:
        """Get formatted dataset based on strategy."""
        if self.strategy == Strategy.BASELINE_FINETUNING:
            return self._format_splits(self.get_train_test_base()), self.dataset_info
        return self._format_splits(self.get_curriculum_splits()), self.dataset_info
