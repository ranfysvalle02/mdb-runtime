# MDB_RUNTIME

**MongoDB Multi-Tenant Experiment Runtime Engine**

Enterprise-grade runtime engine for building multi-tenant applications with automatic database scoping, authentication, authorization, and resource management.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🎯 **Multi-tenant Database Scoping** - Automatic experiment isolation with transparent query filtering
- 🔐 **Authentication & Authorization** - Built-in auth with Casbin/OSO support
- ✅ **Manifest Validation** - JSON schema validation with versioning and migration
- 📊 **Index Management** - Automatic Atlas Search and Vector index management
- 🚀 **Runtime Engine** - Centralized orchestration for all components
- 📈 **Observability** - Built-in metrics, structured logging, and health checks
- 🧪 **Test Infrastructure** - Comprehensive test suite with fixtures
- 🔧 **Type Safety** - Extensive type hints for better IDE support

## Installation

```bash
pip install mdb-runtime
```

Or install with optional dependencies:

```bash
# With test dependencies
pip install -e ".[test]"

# With all optional dependencies
pip install -e ".[all]"
```

## Quick Start

```python
from mdb_runtime import RuntimeEngine

# Initialize engine
engine = RuntimeEngine(
    mongo_uri="mongodb://localhost:27017",
    db_name="my_database"
)

# Initialize (async)
await engine.initialize()

# Get scoped database (automatic experiment isolation)
db = engine.get_scoped_db("my_experiment")

# Use MongoDB-style API - all queries automatically scoped
doc = await db.my_collection.find_one({"name": "test"})
await db.my_collection.insert_one({"name": "New Document"})

# Register experiment from manifest
manifest = await engine.load_manifest("path/to/manifest.json")
await engine.register_experiment(manifest)

# Health checks and metrics
health = await engine.get_health_status()
metrics = engine.get_metrics()

# Cleanup
await engine.shutdown()
```

### Using Context Manager

```python
async with RuntimeEngine(mongo_uri, db_name) as engine:
    await engine.reload_experiments()
    db = engine.get_scoped_db("my_experiment")
    # ... use engine
    # Automatic cleanup on exit
```

## Core Components

### RuntimeEngine

The central orchestration engine that manages:
- Database connections and scoping
- Experiment registration and lifecycle
- Manifest validation and parsing
- Index management
- Resource lifecycle

### Database Scoping

Automatic experiment isolation - all queries are automatically filtered by `experiment_id`:

```python
db = engine.get_scoped_db("my_experiment")

# This query automatically includes: {"experiment_id": {"$in": ["my_experiment"]}}
docs = await db.users.find({"status": "active"}).to_list(length=10)

# Inserts automatically get experiment_id added
await db.users.insert_one({"name": "John", "status": "active"})
# Document stored as: {"name": "John", "status": "active", "experiment_id": "my_experiment"}
```

### Manifest System

JSON schema validation with versioning:

```python
from mdb_runtime.core import ManifestValidator, ManifestParser

# Validate manifest
validator = ManifestValidator()
is_valid, error, paths = validator.validate(manifest)

# Load and parse
parser = ManifestParser()
manifest = await parser.load_from_file("manifest.json")
```

### Index Management

Automatic index creation and management:

```python
from mdb_runtime.indexes import AsyncAtlasIndexManager

# Create vector search indexes
index_manager = AsyncAtlasIndexManager(collection)
await index_manager.create_search_index(
    name="vector_index",
    definition={...},
    index_type="vectorSearch"
)
```

### Observability

Built-in metrics, logging, and health checks:

```python
from mdb_runtime.observability import (
    get_metrics_collector,
    get_logger,
    set_correlation_id,
)

# Get metrics
collector = get_metrics_collector()
metrics = collector.get_summary()

# Structured logging with correlation IDs
correlation_id = set_correlation_id()
logger = get_logger(__name__)
logger.info("Operation completed")  # Automatically includes correlation_id

# Health checks
health = await engine.get_health_status()
```

## Project Structure

```
mdb-runtime/
├── mdb_runtime/          # Main package
│   ├── core/             # RuntimeEngine, manifest validation
│   ├── database/         # Scoped wrappers, connection pooling
│   ├── auth/             # Authentication & authorization
│   ├── indexes/           # Index management
│   ├── observability/     # Metrics, logging, health checks
│   ├── utils/            # Utility functions
│   └── constants.py       # Shared constants
├── tests/                # Test suite
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── docs/                 # Documentation
└── scripts/              # Utility scripts
```

## Documentation

- **[Quick Start Guide](docs/QUICK_START.md)** - Get started quickly
- **[Project Structure](PROJECT_STRUCTURE.md)** - Detailed project organization
- **[Test Documentation](tests/README.md)** - Testing guide

## Testing

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest

# Run with coverage
pytest --cov=mdb_runtime --cov-report=html

# Run specific test
pytest tests/unit/test_engine.py
```

## Code Quality

The codebase follows best practices:

- ✅ **Comprehensive test suite** - 40+ unit tests
- ✅ **Type hints** - 85%+ coverage
- ✅ **Error handling** - Context-rich exceptions
- ✅ **Constants** - No magic numbers
- ✅ **Observability** - Metrics, logging, health checks
- ✅ **Documentation** - Comprehensive docstrings

## Requirements

- Python 3.8+
- MongoDB 4.4+
- Motor 3.0+
- PyMongo 4.0+

### Optional Dependencies

- `pydantic>=2.0.0` - Enhanced configuration validation
- `casbin>=1.0.0` - Casbin authorization provider
- `oso>=0.27.0` - OSO authorization provider
- `ray>=2.0.0` - Ray actor support

## License

MIT License

## Contributing

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for project organization guidelines.

## Support

For issues and questions, please open an issue on the repository.
