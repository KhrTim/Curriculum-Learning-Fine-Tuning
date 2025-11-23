# Tests

This directory contains unit tests for the math_llm_v3 project.

## Running Tests

Run all tests:
```bash
python -m pytest tests/
```

Run specific test file:
```bash
python -m pytest tests/test_config_dataclasses.py
```

Run with coverage:
```bash
python -m pytest tests/ --cov=utils --cov-report=html
```

## Running Linting

Check code style with flake8:
```bash
flake8 utils/ run_experiment_ref.py tests/
```

Format code with black:
```bash
black utils/ run_experiment_ref.py tests/
```

## Running Type Checking

Check types with mypy:
```bash
mypy utils/ run_experiment_ref.py
```

## Test Structure

- `test_config_dataclasses.py` - Tests for configuration dataclasses
- `test_dataloader.py` - Tests for dataset loading and processing
- `test_evaluation.py` - Tests for model evaluation functions
