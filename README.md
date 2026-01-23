# mdb-engine

**The MongoDB Engine for Python Apps** — Auto-sandboxing, index management, and auth in one package.

[![PyPI](https://img.shields.io/pypi/v/mdb-engine)](https://pypi.org/project/mdb-engine/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://opensource.org/licenses/AGPL-3.0)

---

## 🎯 manifest.json: The Key to Everything

**`manifest.json` is the foundation of your application.** It's a single configuration file that defines your app's identity, data structure, authentication, indexes, and services. Everything flows from this file.

### Your First manifest.json

Create a `manifest.json` file with just 3 fields:

```json
{
  "schema_version": "2.0",
  "slug": "my_app",
  "name": "My App"
}
```

That's it! This minimal manifest gives you:
- ✅ Automatic data scoping (all queries filtered by `app_id`)
- ✅ Collection name prefixing (`db.tasks` → `my_app_tasks`)
- ✅ App registration and lifecycle management

**Learn more**: [Quick Start Guide](docs/QUICK_START.md) | [Manifest Deep Dive](docs/MANIFEST_DEEP_DIVE.md)

---

## Installation

```bash
pip install mdb-engine
```

---

## 30-Second Quick Start

**Step 1**: Create your `manifest.json`:

```json
{
  "schema_version": "2.0",
  "slug": "my_app",
  "name": "My App",
  "managed_indexes": {
    "tasks": [
      {
        "type": "regular",
        "keys": {"status": 1, "created_at": -1},
        "name": "status_sort"
      }
    ]
  }
}
```

**Step 2**: Create your FastAPI app:

```python
from pathlib import Path
from fastapi import Depends
from mdb_engine import MongoDBEngine
from mdb_engine.dependencies import get_scoped_db

# Initialize the engine
engine = MongoDBEngine(
    mongo_uri="mongodb://localhost:27017",
    db_name="my_database"
)

# Create app - manifest.json is loaded automatically!
app = engine.create_app(slug="my_app", manifest=Path("manifest.json"))

# Use request-scoped dependencies - all queries automatically isolated
@app.post("/tasks")
async def create_task(task: dict, db=Depends(get_scoped_db)):
    result = await db.tasks.insert_one(task)
    return {"id": str(result.inserted_id)}
```

**What just happened?**
- ✅ Engine loaded your `manifest.json`
- ✅ Indexes created automatically from `managed_indexes`
- ✅ Database queries automatically scoped to your app
- ✅ Lifecycle management handled (startup/shutdown)

That's it. Your data is automatically sandboxed, indexes are created, and cleanup is handled.

---

## Basic Examples

### 1. Index Management

Define indexes in your `manifest.json` — they're auto-created on startup:

```json
{
  "schema_version": "2.0",
  "slug": "my_app",
  "name": "My App",
  "status": "active",
  "managed_indexes": {
    "tasks": [
      {
        "type": "regular",
        "keys": {"status": 1, "created_at": -1},
        "name": "status_sort"
      },
      {
        "type": "regular",
        "keys": {"priority": -1},
        "name": "priority_idx"
      }
    ],
    "users": [
      {
        "type": "regular",
        "keys": {"email": 1},
        "name": "email_unique",
        "unique": true
      }
    ]
  }
}
```

Supported index types: `regular`, `text`, `vector`, `ttl`, `compound`.

### 2. CRUD Operations (Auto-Scoped)

All database operations are automatically scoped to your app. Use `Depends(get_scoped_db)` in route handlers:

```python
from mdb_engine.dependencies import get_scoped_db

@app.post("/tasks")
async def create_task(task: dict, db=Depends(get_scoped_db)):
    result = await db.tasks.insert_one(task)
    return {"id": str(result.inserted_id)}

@app.get("/tasks")
async def list_tasks(db=Depends(get_scoped_db)):
    return await db.tasks.find({"status": "pending"}).to_list(length=10)

@app.put("/tasks/{task_id}")
async def update_task(task_id: str, db=Depends(get_scoped_db)):
    await db.tasks.update_one({"_id": task_id}, {"$set": {"status": "done"}})
    return {"updated": True}

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str, db=Depends(get_scoped_db)):
    await db.tasks.delete_one({"_id": task_id})
    return {"deleted": True}
```

**What happens under the hood:**
```python
# You write:
await db.tasks.find({}).to_list(length=10)

# Engine executes:
# Collection: my_app_tasks
# Query: {"app_id": "my_app"}
```

### 3. Health Checks

Built-in observability:

```python
@app.get("/health")
async def health():
    status = await engine.get_health_status()
    return {"status": status.get("status", "unknown")}
```

---

## Why mdb-engine?

- **manifest.json is everything** — Single source of truth for your entire app configuration
- **Zero boilerplate** — No more connection setup, index creation scripts, or auth handlers
- **Data isolation** — Multi-tenant ready with automatic app sandboxing
- **Manifest-driven** — Define your app's "DNA" in JSON, not scattered code
- **Incremental adoption** — Start minimal, add features as needed
- **No lock-in** — Standard Motor/PyMongo underneath; export anytime with `mongodump --query='{"app_id":"my_app"}'`

---

## Advanced Features

| Feature | Description | Learn More |
|---------|-------------|------------|
| **Authentication** | JWT + Casbin/OSO RBAC | [Auth Guide](https://github.com/ranfysvalle02/mdb-engine/blob/main/docs/AUTHZ.md) |
| **Vector Search** | Atlas Vector Search + embeddings | [RAG Example](https://github.com/ranfysvalle02/mdb-engine/tree/main/examples/basic/interactive_rag) |
| **Memory Service** | Persistent AI memory with Mem0 | [Chat Example](https://github.com/ranfysvalle02/mdb-engine/tree/main/examples/basic/chit_chat) |
| **WebSockets** | Real-time updates from manifest | [Docs](https://github.com/ranfysvalle02/mdb-engine/blob/main/docs/ARCHITECTURE.md) |
| **Multi-App** | Secure cross-app data access | [Multi-App Example](https://github.com/ranfysvalle02/mdb-engine/tree/main/examples/advanced/multi_app) |
| **SSO** | Shared auth across apps | [Shared Auth Example](https://github.com/ranfysvalle02/mdb-engine/tree/main/examples/advanced/multi_app_shared) |

### AppContext — All Services in One Place ✨

```python
from fastapi import Depends
from mdb_engine.dependencies import AppContext

@app.post("/ai-chat")
async def chat(query: str, ctx: AppContext = Depends()):
    user = ctx.require_user()  # 401 if not logged in
    ctx.require_role("user")   # 403 if missing role
    
    # Everything available: ctx.db, ctx.embedding_service, ctx.memory, ctx.llm
    if ctx.llm:
        response = ctx.llm.chat.completions.create(
            model=ctx.llm_model,
            messages=[{"role": "user", "content": query}]
        )
        return {"response": response.choices[0].message.content}
```

---

## Full Examples

Clone and run:

```bash
git clone https://github.com/ranfysvalle02/mdb-engine.git
cd mdb-engine/examples/basic/chit_chat
docker-compose up --build
```

### Basic Examples

| Example | Description |
|---------|-------------|
| [chit_chat](https://github.com/ranfysvalle02/mdb-engine/tree/main/examples/basic/chit_chat) | AI chat with persistent memory |
| [interactive_rag](https://github.com/ranfysvalle02/mdb-engine/tree/main/examples/basic/interactive_rag) | RAG with vector search |
| [oso_hello_world](https://github.com/ranfysvalle02/mdb-engine/tree/main/examples/basic/oso_hello_world) | OSO Cloud authorization |
| [parallax](https://github.com/ranfysvalle02/mdb-engine/tree/main/examples/basic/parallax) | Dynamic schema generation |
| [vector_hacking](https://github.com/ranfysvalle02/mdb-engine/tree/main/examples/basic/vector_hacking) | Vector embeddings & attacks |

### Advanced Examples

| Example | Description |
|---------|-------------|
| [simple_app](https://github.com/ranfysvalle02/mdb-engine/tree/main/examples/advanced/simple_app) | Task management with `create_app()` pattern |
| [multi_app](https://github.com/ranfysvalle02/mdb-engine/tree/main/examples/advanced/multi_app) | Multi-tenant with cross-app access |
| [multi_app_shared](https://github.com/ranfysvalle02/mdb-engine/tree/main/examples/advanced/multi_app_shared) | SSO with shared user pool |

---

## Manual Setup (Alternative)

If you need more control over the FastAPI lifecycle:

```python
from pathlib import Path
from fastapi import FastAPI
from mdb_engine import MongoDBEngine

app = FastAPI()
engine = MongoDBEngine(mongo_uri="mongodb://localhost:27017", db_name="my_database")

@app.on_event("startup")
async def startup():
    await engine.initialize()
    manifest = await engine.load_manifest(Path("manifest.json"))
    await engine.register_app(manifest, create_indexes=True)

@app.on_event("shutdown")
async def shutdown():
    await engine.shutdown()

@app.get("/items")
async def get_items():
    db = engine.get_scoped_db("my_app")
    return await db.items.find({}).to_list(length=10)
```

---

## Understanding manifest.json

Your `manifest.json` is the heart of your application. It defines:

- **App Identity**: `slug`, `name`, `description`
- **Data Access**: `data_access.read_scopes`, `data_access.write_scope`
- **Indexes**: `managed_indexes` (regular, vector, text, TTL, compound)
- **Authentication**: `auth.policy`, `auth.users` (Casbin/OSO, demo users)
- **AI Services**: `embedding_config`, `memory_config`
- **Real-time**: `websockets` endpoints
- **CORS**: `cors` settings

**Start minimal, grow as needed.** You can begin with just `slug`, `name`, and `schema_version`, then add features incrementally.

**📖 Learn More:**
- [Quick Start Guide](docs/QUICK_START.md) - Get started with manifest.json
- [Manifest Deep Dive](docs/MANIFEST_DEEP_DIVE.md) - Comprehensive manifest.json guide
- [Examples](examples/) - Real-world manifest.json files

---

## Links

- [GitHub Repository](https://github.com/ranfysvalle02/mdb-engine)
- [Documentation](https://github.com/ranfysvalle02/mdb-engine/tree/main/docs)
- [All Examples](https://github.com/ranfysvalle02/mdb-engine/tree/main/examples)
- [Quick Start Guide](docs/QUICK_START.md) - **Start here!**
- [Contributing](CONTRIBUTING.md)

---

**Stop building scaffolding. Start building features.**
