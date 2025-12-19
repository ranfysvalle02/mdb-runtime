# MDB_RUNTIME

**The MongoDB Runtime Engine That Makes Multi-App Development Actually Bearable**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## The Problem: MongoDB Development is a Nightmare (Even for One App)

You're building an application with MongoDB. Maybe it's a single app, maybe you're planning for multiple apps later. Either way, you hit the same problems:

**You find yourself writing the same boilerplate over and over:**

```python
# Every. Single. Query.
docs = await collection.find({
    "$and": [
        {"app_id": current_app_id},  # Don't forget this!
        {"status": "active"}
    ]
}).to_list(length=10)

# Every. Single. Insert.
await collection.insert_one({
    **document,
    "app_id": current_app_id  # Hope you didn't forget...
})
```

**You're constantly worried about data leaks** - one missing `app_id` filter and you're exposing data across apps. You write defensive code everywhere, add validation layers, and still wake up at 3 AM wondering if you missed a spot.

**Index management becomes a chore** - you need indexes on `app_id` for every collection, plus compound indexes for your actual queries. You're manually creating and managing dozens of indexes, and every schema change means updating index definitions.

**Authentication and authorization get messy** - each app might need different access rules, but you're building this from scratch every time. You're juggling JWT tokens, session cookies, role-based access, and permission checks across multiple apps.

**Observability is an afterthought** - you're adding logging and metrics manually, trying to trace requests across services, and debugging issues without proper context.

**You're spending 70% of your time on infrastructure, 30% on features.**

And if you're building a single app? You still need all of this - indexes, auth, observability, error handling. The only difference is you're not worried about data leaks (yet). But what happens when you need to add a second app? You're back to rewriting everything.

## The Solution: MDB_RUNTIME

MDB_RUNTIME is a runtime engine that handles all the hard parts of MongoDB development - whether you're building one app or a hundred. It gives you production-ready infrastructure from day one, so you can focus on building features instead of boilerplate.

**Single app?** You get automatic index management, built-in observability, and manifest-driven configuration.  
**Multiple apps?** You get all that plus automatic data isolation and multi-app auth.  
**Starting with one, planning for more?** You're already set up for scale.

### What It Does

**Automatic Data Isolation** - Every query is automatically scoped to the current app. No more manual `app_id` filters. No more data leaks. (Even with one app, this gives you a clean data model and makes it trivial to add more apps later.)

```python
# That's it. Seriously.
db = engine.get_scoped_db("my_app")
docs = await db.users.find({"status": "active"}).to_list(length=10)
# Automatically filtered by app_id - you never even see it
```

**Automatic Index Management** - Indexes are created automatically based on your query patterns. The `app_id` index? Handled. Compound indexes for your queries? Created in the background.

**Built-in Authentication & Authorization** - Support for multiple auth strategies, role-based access control, and app-specific user management. All configured through simple manifests.

**Observability Built-In** - Structured logging with correlation IDs, metrics collection, and health checks. Everything you need to debug and monitor in production.

**Manifest-Driven Configuration** - Define your app's configuration, indexes, and auth rules in a JSON manifest. Version it, validate it, and deploy it.

## Quick Start

```bash
pip install mdb-runtime
```

### Single App? Just as Easy

```python
from mdb_runtime import RuntimeEngine

# Initialize once
engine = RuntimeEngine(
    mongo_uri="mongodb://localhost:27017",
    db_name="my_database"
)
await engine.initialize()

# Get a scoped database - works perfectly for a single app
db = engine.get_scoped_db("my_app")

# Use it like normal MongoDB - all the infrastructure is handled
doc = await db.users.find_one({"email": "user@example.com"})
await db.users.insert_one({"name": "John", "email": "john@example.com"})

# Register your app from a manifest (optional but recommended)
manifest = await engine.load_manifest("path/to/manifest.json")
await engine.register_app(manifest)

# Health checks and metrics are built-in
health = await engine.get_health_status()
```

**That's it.** Even with one app, you get automatic index management, structured logging, metrics, and health checks. When you're ready to add more apps, you're already set up.

## Building Your First App: A Complete Guide

Let's build a simple task management app from scratch. This guide assumes you're building a single-app application (the most common starting point).

### Step 1: Install and Setup

```bash
pip install mdb-runtime
```

Create a new directory for your project:

```bash
mkdir my-task-app
cd my-task-app
```

### Step 2: Create Your App Manifest

Create a `manifest.json` file that defines your app configuration:

```json
{
  "version": "2.0",
  "slug": "task_manager",
  "name": "Task Manager",
  "description": "A simple task management application",
  "status": "active",
  "developer_id": "you@example.com",
  "auth_required": true,
  "auth_policy": {
    "roles": ["admin", "user"],
    "owner_can_access": true
  },
  "managed_indexes": {
    "tasks": [
      {
        "keys": {"status": 1, "created_at": -1},
        "name": "status_created_idx",
        "background": true
      }
    ]
  }
}
```

This manifest tells MDB_RUNTIME:
- Your app slug is `task_manager`
- It requires authentication
- Users with "admin" or "user" roles can access it
- You want an index on the `tasks` collection for status and created_at

### Step 3: Initialize the Runtime Engine

Create `main.py`:

```python
import asyncio
from mdb_runtime import RuntimeEngine
from pathlib import Path

async def main():
    # Initialize the engine
    engine = RuntimeEngine(
        mongo_uri="mongodb://localhost:27017",
        db_name="task_app_db"
    )
    
    # Connect to MongoDB
    await engine.initialize()
    
    # Load and register your app manifest
    manifest_path = Path("manifest.json")
    manifest = await engine.load_manifest(manifest_path)
    await engine.register_app(manifest)
    
    print("✅ App registered successfully!")
    
    # Get a scoped database for your app
    db = engine.get_scoped_db("task_manager")
    
    # Now you can use it like normal MongoDB
    # All queries are automatically scoped to "task_manager"
    
    # Create a task
    result = await db.tasks.insert_one({
        "title": "Learn MDB_RUNTIME",
        "description": "Read the docs and build something",
        "status": "todo",
        "created_at": "2024-01-15T10:00:00Z"
    })
    print(f"✅ Created task with ID: {result.inserted_id}")
    
    # Find tasks
    tasks = await db.tasks.find({"status": "todo"}).to_list(length=10)
    print(f"✅ Found {len(tasks)} todo tasks")
    
    # Update a task
    await db.tasks.update_one(
        {"_id": result.inserted_id},
        {"$set": {"status": "in_progress"}}
    )
    print("✅ Updated task status")
    
    # Check health
    health = await engine.get_health_status()
    print(f"✅ Engine health: {health['status']}")
    
    # Cleanup
    await engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 4: Run Your App

Make sure MongoDB is running, then:

```bash
python main.py
```

You should see:
```
✅ App registered successfully!
✅ Created task with ID: 507f1f77bcf86cd799439011
✅ Found 1 todo tasks
✅ Updated task status
✅ Engine health: connected
```

### Step 5: Understanding What Happened

**Automatic App Scoping:**
- When you called `db.tasks.insert_one()`, MDB_RUNTIME automatically added `"app_id": "task_manager"` to your document
- When you called `db.tasks.find()`, it automatically filtered by `{"app_id": {"$in": ["task_manager"]}}`
- You never had to think about it - it just works

**Automatic Index Management:**
- The index defined in your manifest (`status_created_idx`) was created automatically
- The `app_id` index was also created automatically (required for all queries)
- All indexes are created in the background, so your queries are fast

**What's in the Database:**
Your task document actually looks like this:
```json
{
  "_id": "507f1f77bcf86cd799439011",
  "title": "Learn MDB_RUNTIME",
  "description": "Read the docs and build something",
  "status": "todo",
  "created_at": "2024-01-15T10:00:00Z",
  "app_id": "task_manager"  // ← Added automatically
}
```

### Step 6: Building a Real API

Let's create a FastAPI application:

```python
from fastapi import FastAPI, HTTPException
from mdb_runtime import RuntimeEngine
from mdb_runtime.database import AppDB, get_app_db
from pathlib import Path
import asyncio

app = FastAPI()
engine = None

@app.on_event("startup")
async def startup():
    global engine
    engine = RuntimeEngine(
        mongo_uri="mongodb://localhost:27017",
        db_name="task_app_db"
    )
    await engine.initialize()
    
    # Register app from manifest
    manifest = await engine.load_manifest(Path("manifest.json"))
    await engine.register_app(manifest)

@app.on_event("shutdown")
async def shutdown():
    if engine:
        await engine.shutdown()

def get_scoped_db():
    """Helper to get scoped database for FastAPI dependency"""
    return engine.get_scoped_db("task_manager")

@app.get("/tasks")
async def list_tasks(db: AppDB = None):
    """List all tasks"""
    if db is None:
        db = AppDB(engine.get_scoped_db("task_manager"))
    
    tasks = await db.tasks.find({}).to_list(length=100)
    return {"tasks": tasks}

@app.post("/tasks")
async def create_task(task: dict, db: AppDB = None):
    """Create a new task"""
    if db is None:
        db = AppDB(engine.get_scoped_db("task_manager"))
    
    result = await db.tasks.insert_one(task)
    return {"id": str(result.inserted_id), "task": task}

@app.get("/tasks/{task_id}")
async def get_task(task_id: str, db: AppDB = None):
    """Get a specific task"""
    if db is None:
        db = AppDB(engine.get_scoped_db("task_manager"))
    
    from bson import ObjectId
    task = await db.tasks.find_one({"_id": ObjectId(task_id)})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.put("/tasks/{task_id}")
async def update_task(task_id: str, updates: dict, db: AppDB = None):
    """Update a task"""
    if db is None:
        db = AppDB(engine.get_scoped_db("task_manager"))
    
    from bson import ObjectId
    result = await db.tasks.update_one(
        {"_id": ObjectId(task_id)},
        {"$set": updates}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"updated": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Now you have a fully functional API where:
- Every query is automatically scoped to your app
- Indexes are managed automatically
- You can add authentication later without changing your code
- All operations are logged and monitored

### Step 7: Adding Authentication (Optional)

If you want to add authentication, update your manifest:

```json
{
  "slug": "task_manager",
  "sub_auth": {
    "enabled": true,
    "strategy": "app_users",
    "users_collection": "users",
    "session_cookie_name": "task_manager_session",
    "session_ttl_seconds": 86400
  }
}
```

Then in your FastAPI routes:

```python
from mdb_runtime.auth import get_app_sub_user

@app.get("/tasks")
async def list_tasks(request: Request):
    # Get authenticated user
    db = engine.get_scoped_db("task_manager")
    user = await get_app_sub_user(request, "task_manager", db)
    
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # User is authenticated, proceed with query
    tasks = await db.tasks.find({}).to_list(length=100)
    return {"tasks": tasks}
```

### What You've Built

You now have:
- ✅ A MongoDB-backed application with automatic data scoping
- ✅ Automatic index management
- ✅ A clean data model ready for scaling
- ✅ Production-ready error handling and observability
- ✅ A foundation that makes adding more apps trivial

**Next Steps:**
- Add more collections (users, projects, etc.) - they all get automatic scoping
- Add more indexes in your manifest - they're created automatically
- Add authentication using the built-in auth system
- When you're ready for multiple apps, just use different app slugs - no code changes needed

### Using Context Manager

```python
async with RuntimeEngine(mongo_uri, db_name) as engine:
    await engine.reload_apps()
    db = engine.get_scoped_db("my_app")
    # ... use engine
    # Automatic cleanup on exit
```

## Core Features

### 🎯 Automatic App Isolation

Every query is automatically scoped. Every insert gets `app_id` added. You can't accidentally leak data between apps.

**Single app scenario?** The `app_id` is still there, giving you a clean data model. When you need to add a second app, you don't need to change any code - just use a different app slug.

```python
db = engine.get_scoped_db("my_app")

# This query automatically includes: {"app_id": {"$in": ["my_app"]}}
docs = await db.users.find({"status": "active"}).to_list(length=10)

# Inserts automatically get app_id added
await db.users.insert_one({"name": "John", "status": "active"})
# Document stored as: {"name": "John", "status": "active", "app_id": "my_app"}
```

### 🔐 Authentication & Authorization

Built-in support for multiple auth strategies, role-based access control, and app-specific user management. Configure it all through manifests.

```python
# App-specific authentication
from mdb_runtime.auth import get_app_sub_user

user = await get_app_sub_user(request, app_slug, db, config)
if user:
    # User authenticated for this app
    pass
```

### ✅ Manifest Validation

Define your app configuration in JSON. Validate it, version it, and deploy it. Schema validation with automatic migration support.

```python
from mdb_runtime.core import ManifestValidator

validator = ManifestValidator()
is_valid, error, paths = await validator.validate(manifest)
```

### 📊 Automatic Index Management

Indexes are created automatically based on query patterns. The `app_id` index is always there. Compound indexes are created in the background as you query.

```python
# Just query - indexes are created automatically
docs = await db.products.find({"category": "electronics"}).sort("price").to_list(10)
# Index on (app_id, category, price) created automatically
```

### 📈 Built-in Observability

Structured logging with correlation IDs, metrics collection, and health checks. Everything you need to debug and monitor.

```python
from mdb_runtime.observability import (
    set_app_context,
    record_operation,
    get_logger
)

# Set context for logging
set_app_context(app_slug="my_app")
logger = get_logger(__name__)
logger.info("Operation completed")  # Automatically includes app context

# Record metrics
record_operation("db.query", duration_ms=45.2, success=True)
```

## Project Structure

```
mdb-runtime/
├── mdb_runtime/          # Main package
│   ├── core/             # RuntimeEngine, manifest validation
│   ├── database/         # Scoped wrappers, connection pooling
│   ├── auth/             # Authentication & authorization
│   ├── indexes/          # Index management
│   ├── observability/    # Metrics, logging, health checks
│   ├── utils/            # Utility functions
│   └── constants.py      # Shared constants
├── tests/                # Test suite
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── docs/                 # Documentation
└── scripts/              # Utility scripts
```

## Documentation

- **[Quick Start Guide](docs/QUICK_START.md)** - Get started in 5 minutes
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

This isn't a side project. It's production-ready:

- ✅ **Comprehensive test suite** - 40+ unit tests with fixtures
- ✅ **Type hints** - 85%+ coverage for better IDE support
- ✅ **Error handling** - Context-rich exceptions with debugging info
- ✅ **Constants** - No magic numbers, everything is configurable
- ✅ **Observability** - Metrics, structured logging, health checks built-in
- ✅ **Documentation** - Comprehensive docstrings and examples

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

## Installation

```bash
# Basic installation
pip install mdb-runtime

# With test dependencies
pip install -e ".[test]"

# With all optional dependencies
pip install -e ".[all]"
```

## License

MIT License - Use it however you want.

## Contributing

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for project organization guidelines.

## Support

For issues and questions, please open an issue on the repository.

---

**Stop writing boilerplate. Start building features.**

---

## FAQ

### Can I use this for a single app?

**Absolutely.** MDB_RUNTIME works great for single-app scenarios. You get all the benefits:
- Automatic index management
- Built-in observability (metrics, logging, health checks)
- Manifest-driven configuration
- Production-ready error handling

The `app_id` field is still added to your documents, which gives you a clean data model and makes it trivial to add more apps later if needed. You're not paying any performance penalty - it's just a field in your documents.

### Do I need multiple apps to benefit?

**No.** Even with one app, you're getting:
- **Automatic index management** - No more manually creating and maintaining indexes
- **Observability** - Structured logging, metrics, and health checks out of the box
- **Manifest validation** - Version-controlled, validated configuration
- **Type safety** - Comprehensive type hints for better IDE support
- **Error handling** - Context-rich exceptions with debugging information

The multi-app isolation is a bonus feature that doesn't cost anything if you're only using one app.
