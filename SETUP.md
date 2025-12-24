# Development Setup Guide

Quick guide to set up the development environment for MDB Engine.

## Prerequisites

- Python 3.8+
- pip
- git

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd mdb-runtime

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

## Verify Setup

```bash
# Run pre-commit hooks manually to verify
pre-commit run --all-files

# Run tests
pytest

# Check exception handling
python scripts/check_exception_handling.py mdb_engine/
```

## Pre-commit Hooks

Pre-commit hooks automatically run before each commit to ensure code quality:

- **Code formatting** (Black)
- **Import sorting** (isort)
- **Linting** (flake8)
- **Exception handling checks** (custom)

### Running Hooks Manually

```bash
# Run all hooks on staged files
pre-commit run

# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run flake8 --all-files
pre-commit run check-exception-handling --all-files
```

### Bypassing Hooks (Not Recommended)

Only bypass hooks in emergencies:

```bash
git commit --no-verify -m "Emergency fix"
```

## Code Formatting

```bash
# Format code
black --line-length=100 .

# Sort imports
isort --profile black --line-length=100 .

# Check formatting without changing files
black --check --line-length=100 .
```

## Linting

```bash
# Run flake8
flake8 mdb_engine tests

# Run with specific error codes
flake8 --select=E,W,F mdb_engine
```

## Exception Handling Checker

The custom exception handling checker enforces best practices:

```bash
# Check specific files
python scripts/check_exception_handling.py file1.py file2.py

# Check entire module
python scripts/check_exception_handling.py mdb_engine/database/

# Check all Python files (excluding tests/examples)
find mdb_engine -name "*.py" -not -path "*/test*" | xargs python scripts/check_exception_handling.py
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mdb_engine --cov-report=html

# Run specific test file
pytest tests/unit/test_engine.py

# Run with verbose output
pytest -v -s
```

## Common Issues

### Pre-commit hooks not running

```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install
```

### Import errors in scripts

Make sure you're in the project root directory and the virtual environment is activated.

### Flake8 errors

Check `.flake8` configuration. Some errors may be acceptable (e.g., in `__init__.py` files).

## Next Steps

- Read [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines
- Review exception handling best practices in CONTRIBUTING.md
- Check existing issues and PRs before starting work

