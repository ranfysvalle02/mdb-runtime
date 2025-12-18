# MDB_RUNTIME Project Structure

This document describes the organization of the MDB_RUNTIME project.

## Directory Structure

```
mdb-runtime/
├── docs/                      # Documentation
│   ├── README.md              # Documentation index
│   ├── QUICK_START.md          # Quick start guide
│   ├── EXTRACTION_SUMMARY.md   # Code extraction notes
│   └── WEEK1_SUMMARY.md        # Week 1 improvements summary
│
├── scripts/                    # Utility scripts
│   ├── README.md               # Scripts documentation
│   └── run_tests.sh            # Test runner script
│
├── tests/                      # Test suite
│   ├── README.md               # Test documentation
│   ├── conftest.py             # Shared pytest fixtures
│   ├── unit/                   # Unit tests
│   │   ├── test_engine.py
│   │   ├── test_manifest.py
│   │   └── test_exceptions.py
│   └── integration/            # Integration tests
│       └── (to be added)
│
├── mdb_runtime/                # Main package
│   ├── __init__.py             # Package exports
│   ├── README.md               # Package documentation
│   ├── exceptions.py           # Custom exceptions
│   ├── config.py               # Configuration management
│   │
│   ├── core/                   # Core runtime engine
│   │   ├── engine.py           # RuntimeEngine class
│   │   └── manifest.py         # Manifest validation & parsing
│   │
│   ├── database/               # Database layer
│   │   ├── connection.py       # Connection management
│   │   ├── scoped_wrapper.py   # Scoped database wrapper
│   │   └── abstraction.py      # Database abstractions
│   │
│   ├── auth/                   # Authentication & authorization
│   │   ├── provider.py         # Authorization providers
│   │   ├── jwt.py              # JWT handling
│   │   ├── dependencies.py     # FastAPI dependencies
│   │   ├── restrictions.py     # Access restrictions
│   │   └── sub_auth.py         # Sub-authentication
│   │
│   ├── indexes/                # Index management
│   │   └── manager.py          # Index creation & management
│   │
│   ├── utils/                  # Utility functions
│   │   ├── validation.py       # Input validation
│   │   └── decorators.py       # Decorators
│   │
│   └── [other modules]/        # Additional modules
│       ├── actors/
│       ├── cache/
│       ├── multi_tenancy/
│       ├── observability/
│       ├── resilience/
│       ├── routing/
│       ├── security/
│       └── testing/
│
├── .gitignore                  # Git ignore rules
├── LICENSE                     # License file
├── MANIFEST.in                 # Package manifest
├── pyproject.toml              # Project configuration
├── pytest.ini                  # Pytest configuration
├── README.md                   # Main project README
├── setup.py                    # Setup script
└── IMPROVEMENTS.md             # Improvement tracking
```

## Organization Principles

### 1. **Separation of Concerns**
- **docs/**: All documentation files
- **scripts/**: Utility and helper scripts
- **tests/**: Test suite organized by type (unit/integration)
- **mdb_runtime/**: Source code organized by feature domain

### 2. **Documentation Location**
- Package-level docs: `mdb_runtime/README.md`
- Project-level docs: `docs/`
- Test docs: `tests/README.md`
- Script docs: `scripts/README.md`

### 3. **Test Organization**
- **unit/**: Fast, isolated unit tests with mocks
- **integration/**: Tests requiring external services (MongoDB)
- **conftest.py**: Shared fixtures and configuration

### 4. **Code Organization**
- Feature-based modules (auth, database, core, etc.)
- Each module has its own `__init__.py`
- Related functionality grouped together

## File Naming Conventions

- **Python files**: `snake_case.py`
- **Test files**: `test_*.py`
- **Documentation**: `README.md` or descriptive names
- **Scripts**: `snake_case.sh` or descriptive names

## Adding New Files

### Adding a New Module
1. Create directory in `mdb_runtime/`
2. Add `__init__.py` with appropriate exports
3. Update `mdb_runtime/__init__.py` if needed
4. Add tests in `tests/unit/`

### Adding Documentation
1. Place in `docs/` directory
2. Update `docs/README.md` index
3. Link from main `README.md` if relevant

### Adding Tests
1. Unit tests → `tests/unit/test_*.py`
2. Integration tests → `tests/integration/test_*.py`
3. Shared fixtures → `tests/conftest.py`

### Adding Scripts
1. Place in `scripts/` directory
2. Make executable: `chmod +x scripts/script_name.sh`
3. Update `scripts/README.md`

## Maintenance

- Keep documentation in sync with code changes
- Update this structure document when adding new major directories
- Follow existing patterns for consistency

