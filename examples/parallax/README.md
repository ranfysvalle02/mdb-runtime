# Parallax - Tech News Intelligence Tool

**🔭 A focused tool for analyzing tech news from two perspectives.**

Parallax monitors Hacker News for stories matching your watchlist keywords and analyzes them from Relevance and Technical perspectives.

## The Parallax Concept

In astronomy, **Parallax** is the apparent displacement of an object when viewed from two different lines of sight. In this tool, it represents one story viewed from two angles:

1. **Relevance Lens** - Why this story matters given your watchlist, key insights, urgency
2. **Technical Lens** - Performance characteristics, complexity, readiness, use cases

## Key Features

- **Watchlist Filtering** - Automatically finds stories matching your keywords
- **Dual Analysis** - Concurrent Relevance and Technical perspectives
- **Structured Output** - Pydantic schemas ensure consistent analysis
- **Scannable Feed** - Compact 2-column layout for quick insights
- **Real-time Dashboard** - Clean interface for monitoring tech news

## Architecture

1. **The Source** - Ingests live data from Hacker News
2. **The Filter** - Detects "buzzword" keywords (MongoDB, Vector, RAG, Python, etc.)
3. **The Parallax View (Fan-Out)** - Orchestrator splits each story into two concurrent agent streams
4. **The Form Extractor** - Each agent enforces a strict Pydantic schema to structure unstructured text

## Prerequisites

**⚠️ IMPORTANT: This demo REQUIRES orchestration dependencies!**

- Python 3.8+
- Docker and Docker Compose (for containerized setup)
- OR MongoDB running locally (for local setup)
- MDB_RUNTIME installed
- **LLM API Key** (OpenAI, Azure OpenAI, or compatible provider) - **REQUIRED**
- **LangChain Dependencies** - **REQUIRED**:
  ```bash
  pip install langchain langchain-community langchain-core langchain-litellm litellm pydantic pydantic-settings httpx requests
  ```

**Note:** Without these dependencies, the application will fail to start. This demo is focused on multi-agent orchestration capabilities.

## Setup

### Using Docker Compose (Recommended)

**1. Set up your LLM API key:**

Create a `.env` file in this directory:

```bash
# OpenAI (default)
OPENAI_API_KEY=your-openai-api-key-here

# OR Azure OpenAI
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# OR Voyage AI (for embeddings)
VOYAGE_API_KEY=your-voyage-key
```

**2. Run the application:**

```bash
docker-compose up
```

That's it! The Docker Compose setup will:
1. Build the application Docker image
2. Start MongoDB with authentication
3. Start the **Parallax dashboard** on http://localhost:8000
4. Show you all the output

**Access the Dashboard:**
- 🌐 **Parallax Dashboard**: http://localhost:8000
- Click "Initialize Scan" to start analyzing Hacker News stories

**With optional MongoDB Express UI:**
```bash
docker-compose --profile ui up
```
Then visit http://localhost:8081 to browse the database.

**Stop everything:**
```bash
docker-compose down
```

**Rebuild after code changes:**
```bash
docker-compose up --build
```

## How It Works

### The Parallax Engine

The `ParallaxEngine` orchestrates the analysis:

1. **Fetch & Filter** - Retrieves top Hacker News stories and filters by watchlist keywords
2. **Fan-Out** - Launches two concurrent agents:
   - **Relevance Agent** - Analyzes why this matters given your watchlist, key insights, urgency
   - **Technical Agent** - Analyzes performance, complexity, readiness, use cases
3. **Aggregate** - Combines both viewpoints into a `ParallaxReport`
4. **Store** - Saves to MongoDB for dashboard visualization

### The Schemas

Each agent uses a strict Pydantic schema:

- **RelevanceView** - Relevance score, why it matters, key insight, urgency
- **TechnicalView** - Performance, complexity, readiness, use case

### The Dashboard

The dashboard displays each story in a 2-column layout, showing Relevance and Technical perspectives side-by-side for quick scanning.

## API Endpoints

- `GET /` - Parallax dashboard (HTML)
- `POST /api/refresh` - Trigger analysis of Hacker News feed
- `GET /api/reports` - Get recent Parallax reports (JSON)

## Understanding the Code

### Manifest (`manifest.json`)

The manifest defines:
- App configuration (slug: `parallax`, name, description)
- LLM configuration (enabled, default models, temperature: 0.0 for factual analysis)
- Indexes for `parallax_reports` collection

### App Scoping

All data is automatically scoped to the `parallax` app:

```python
# You write:
await db.parallax_reports.insert_one(report.dict())

# MDB_RUNTIME stores:
{
    "story_id": 12345,
    "original_title": "...",
    "relevance": {...},
    "product": {...},  # Technical lens (kept for backward compatibility)
    "relevance": {...},
    "app_id": "parallax"  # ← Added automatically
}
```

### LLM Integration

The example uses MDB_RUNTIME's LLM service with LangChain adapters:

```python
from mdb_runtime.llm import LLMService
from parallax import ParallaxEngine

llm_service = engine.get_llm_service("parallax")
parallax = ParallaxEngine(llm_service, db)
reports = await parallax.analyze_feed()
```

## Watchlist Keywords

The default watchlist includes:
- Languages: Python, Rust, Node.js, TypeScript, JavaScript
- Databases: MongoDB, PostgreSQL, Oracle, SQL
- Frameworks: React, Vue, Angular
- Cloud: AWS, Azure, GCP
- DevOps: Kubernetes, Docker
- AI/ML: Vector, RAG, AI

You can customize the watchlist in `parallax.py`.

## Troubleshooting

### LLM API Key Not Set

If you see errors about missing API keys:

1. **Check environment variables:**
   ```bash
   docker-compose exec app env | grep API_KEY
   ```

2. **Set in `.env` file:**
   ```bash
   OPENAI_API_KEY=your-key-here
   ```

3. **Restart the service:**
   ```bash
   docker-compose restart app
   ```

### ModuleNotFoundError

Rebuild the Docker image:

```bash
docker-compose build --no-cache
docker-compose up
```

### MongoDB Connection Issues

1. **Verify MongoDB is running:**
   ```bash
   docker-compose ps mongodb
   ```

2. **Check MongoDB logs:**
   ```bash
   docker-compose logs mongodb
   ```

## Architecture

```
┌─────────────┐
│   Browser   │
│ (Dashboard) │
└──────┬──────┘
       │ HTTP
       ▼
┌─────────────┐
│  FastAPI    │
│  (web.py)   │
└──────┬──────┘
       │
       ├──► ParallaxEngine
       │    ├──► Relevance Agent
       │    └──► Technical Agent
       │
       └──► MongoDB (via MDB_RUNTIME)
            └── parallax_reports
```

## License

Same as MDB_RUNTIME project.
