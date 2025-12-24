# Scripts

This directory contains utility scripts for development and maintenance.

## check_exception_handling.py

Custom pre-commit hook that enforces exception handling best practices.

### What it checks:

1. **Bare except clauses** - Catches `except:` without exception types
2. **Overly broad Exception catches** - Warns about `except Exception:` without proper handling
3. **Missing exception context** - Checks for `raise ... from e` pattern
4. **Missing logger.exception()** - Ensures error handlers use proper logging

### Usage:

```bash
# Check specific files
python scripts/check_exception_handling.py file1.py file2.py

# Run as pre-commit hook (automatic)
pre-commit run check-exception-handling --all-files
```

### Exclusions:

- Test files (`test_*.py`) - May have different exception handling patterns
- Example files (`examples/`) - May have simpler patterns for demonstration

### Integration:

This script is automatically run as part of the pre-commit hooks. See `.pre-commit-config.yaml` for configuration.
