# Contributing to MDB Engine

Thank you for your interest in contributing to MDB Engine! This document provides guidelines and best practices for contributing code.

## Table of Contents

- [Code Style](#code-style)
- [Exception Handling Best Practices](#exception-handling-best-practices)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Code Review Guidelines](#code-review-guidelines)

## Code Style

We use automated tools to maintain consistent code style:

- **Black**: Code formatting (line length: 100)
- **isort**: Import sorting (Black profile)
- **flake8**: Linting with plugins for bug detection
  - **E722**: Catches bare `except:` clauses (enforced for exception handling best practices)

### Setup

```bash
# Install development dependencies
pip install -e ".[test]"

# Install pre-commit hooks (recommended)
pip install pre-commit
pre-commit install
```

### Manual Formatting

```bash
# Format code
black --line-length=100 .

# Sort imports
isort --profile black --line-length=100 .

# Run linting (recommended - uses Makefile)
make lint-check

# Or run flake8 directly
flake8 mdb_engine tests
```

## Exception Handling Best Practices

**MDB Engine follows Staff Engineer-level exception handling practices.** All exception handling must adhere to these rules:

### 1. Never Use Bare Except Clauses ❌

```python
# BAD - Never do this
try:
    result = await db.collection.find_one({})
except:
    return None

# GOOD - Always specify exception types
try:
    result = await db.collection.find_one({})
except (OperationFailure, ConnectionFailure) as e:
    logger.exception("Database operation failed")
    return None
```

### 2. Catch Specific Exceptions ✅

Always catch the most specific exception types you can handle:

```python
# BAD - Too broad
try:
    token = jwt.decode(token, secret)
except Exception as e:
    logger.error(f"Error: {e}")
    return None

# GOOD - Specific exceptions
try:
    token = jwt.decode(token, secret)
except jwt.ExpiredSignatureError:
    logger.debug("Token expired")
    return None
except jwt.InvalidTokenError as e:
    logger.warning(f"Invalid token: {e}")
    return None
except Exception as e:
    logger.exception("Unexpected error decoding token")
    raise  # Re-raise unexpected errors
```

### 3. Preserve Exception Context 🔗

Always use `raise ... from e` when re-raising exceptions:

```python
# BAD - Loses exception context
try:
    result = await collection.insert_one(doc)
except OperationFailure as e:
    logger.error(f"Error: {e}")
    raise MongoDBEngineError("Insert failed")

# GOOD - Preserves exception chain
try:
    result = await collection.insert_one(doc)
except OperationFailure as e:
    logger.exception("Database operation failed")
    raise MongoDBEngineError(
        "Failed to insert document",
        context={"operation": "insert_one"}
    ) from e
```

### 4. Use logger.exception() for Error Logging 📝

Always use `logger.exception()` in exception handlers for full tracebacks:

```python
# BAD - Missing traceback
except Exception as e:
    logger.error(f"Error: {e}")

# GOOD - Full traceback automatically included
except Exception as e:
    logger.exception("Operation failed")
```

### 5. MongoDB-Specific Exception Handling 🗄️

For MongoDB operations, catch specific MongoDB exceptions:

```python
from pymongo.errors import OperationFailure, AutoReconnect, ConnectionFailure, ServerSelectionTimeoutError
from mdb_engine.exceptions import MongoDBEngineError

try:
    result = await collection.insert_one(document)
except (OperationFailure, AutoReconnect) as e:
    logger.exception("Database operation failed")
    raise MongoDBEngineError(
        "Failed to insert document",
        context={"operation": "insert_one"}
    ) from e
except (ConnectionFailure, ServerSelectionTimeoutError) as e:
    logger.exception("Database connection failed")
    raise MongoDBEngineError(
        "Connection failed",
        context={"operation": "insert_one"}
    ) from e
except Exception as e:
    logger.exception("Unexpected error")
    raise MongoDBEngineError(
        "Unexpected error inserting document",
        context={"operation": "insert_one"}
    ) from e
```

### 6. Authentication Exception Handling 🔐

For JWT and authentication operations:

```python
import jwt

try:
    payload = jwt.decode(token, secret, algorithms=["HS256"])
except jwt.ExpiredSignatureError:
    logger.debug("Token expired")
    return None
except jwt.InvalidTokenError as e:
    logger.warning(f"Invalid token: {e}")
    return None
except (ValueError, TypeError) as e:
    logger.exception("Validation error decoding token")
    return None
except Exception as e:
    logger.exception("Unexpected error decoding token")
    raise  # Re-raise unexpected errors
```

### 7. Don't Swallow Exceptions Silently 🚫

Always log or re-raise exceptions:

```python
# BAD - Silent failure
try:
    result = await operation()
except Exception:
    pass

# GOOD - Log or re-raise
try:
    result = await operation()
except SpecificException as e:
    logger.exception("Operation failed")
    return None  # Or handle appropriately
except Exception as e:
    logger.exception("Unexpected error")
    raise  # Re-raise unexpected errors
```

### 8. Use Custom Exceptions for Domain Errors 🎯

Use `MongoDBEngineError` and its subclasses for domain-specific errors:

```python
from mdb_engine.exceptions import MongoDBEngineError, InitializationError, ConfigurationError

# For initialization errors
raise InitializationError(
    "Failed to connect to MongoDB",
    mongo_uri=self.mongo_uri,
    db_name=self.db_name
) from e

# For configuration errors
raise ConfigurationError(
    "Invalid configuration",
    config_key="database_url",
    config_value=value
) from e

# For general engine errors
raise MongoDBEngineError(
    "Operation failed",
    context={"operation": "insert_one", "collection": "users"}
) from e
```

### Exception Handling Patterns

#### Pattern 1: Operations That Should Fail Fast

```python
async def insert_one(self, document: Dict[str, Any]) -> InsertOneResult:
    try:
        return await self._collection.insert_one(document)
    except (OperationFailure, AutoReconnect) as e:
        logger.exception("Database operation failed")
        raise MongoDBEngineError(
            "Failed to insert document",
            context={"operation": "insert_one"}
        ) from e
    except Exception as e:
        logger.exception("Unexpected error")
        raise MongoDBEngineError(
            "Unexpected error inserting document",
            context={"operation": "insert_one"}
        ) from e
```

#### Pattern 2: Operations That Can Return None/False

```python
async def find_one(self, filter: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        return await self._collection.find_one(filter)
    except (OperationFailure, ConnectionFailure) as e:
        logger.exception("Database operation failed")
        return None
    except Exception as e:
        logger.exception("Unexpected error")
        # Re-raise unexpected errors
        raise MongoDBEngineError(
            "Unexpected error retrieving document",
            context={"operation": "find_one"}
        ) from e
```

#### Pattern 3: Optional Dependencies

```python
try:
    from some_optional_library import Feature
    FEATURE_AVAILABLE = True
except ImportError:
    FEATURE_AVAILABLE = False
    logger.debug("Optional dependency not available")
except Exception as e:
    logger.warning("Failed to import optional feature", exc_info=True)
    FEATURE_AVAILABLE = False
```

## Pre-commit Hooks

We use pre-commit hooks to automatically check code quality before commits. The hooks check for:

- Code formatting (Black)
- Import sorting (isort)
- Linting (flake8)
- Exception handling best practices (custom checker)

### Running Pre-commit Hooks Manually

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run

# Run specific hook
pre-commit run flake8 --all-files
pre-commit run check-exception-handling --all-files
```

### Bypassing Pre-commit Hooks

**Only bypass hooks in emergencies** and document why:

```bash
git commit --no-verify -m "Emergency fix (bypassing hooks)"
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mdb_engine --cov-report=html

# Run specific test file
pytest tests/unit/test_engine.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use descriptive test names: `test_function_name_scenario_expected_result`
- Mock external dependencies (MongoDB, APIs, etc.)
- Test both success and failure paths

### Test Example

```python
import pytest
from mdb_engine.database.abstraction import Collection
from pymongo.errors import OperationFailure

@pytest.mark.asyncio
async def test_insert_one_operation_failure_raises_mongodb_engine_error():
    """Test that OperationFailure is caught and re-raised as MongoDBEngineError."""
    collection = Collection(mock_scoped_collection)
    mock_scoped_collection.insert_one.side_effect = OperationFailure("Error", 1)
    
    with pytest.raises(MongoDBEngineError) as exc_info:
        await collection.insert_one({"test": "data"})
    
    assert "Failed to insert document" in str(exc_info.value)
    assert exc_info.value.__cause__ is not None  # Context preserved
```

## Pull Request Process

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the guidelines above

3. **Run pre-commit hooks**:
   ```bash
   pre-commit run --all-files
   ```

4. **Write/update tests** for your changes

5. **Update documentation** if needed

6. **Commit your changes** with descriptive messages:
   ```bash
   git commit -m "feat: add new feature X"
   git commit -m "fix: handle OperationFailure in insert_one"
   git commit -m "docs: update exception handling guidelines"
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request** with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots (if UI changes)
   - Test results

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

## Code Review Guidelines

### For Authors

- Keep PRs focused and small (< 500 lines when possible)
- Respond to review comments promptly
- Be open to feedback and suggestions
- Update PR based on feedback

### For Reviewers

- Be constructive and respectful
- Focus on code quality and maintainability
- Check exception handling patterns
- Verify tests are adequate
- Approve when satisfied

## Questions?

If you have questions about contributing, please:

1. Check existing issues and PRs
2. Read the documentation in `docs/`
3. Open an issue for discussion
4. Reach out to maintainers

Thank you for contributing to MDB Engine! 🚀

