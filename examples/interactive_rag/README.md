# Interactive RAG Example

An interactive RAG (Retrieval Augmented Generation) system demonstrating MDB_RUNTIME with:
- **EmbeddingService** for semantic text splitting and embeddings
- **LLMService** for chat completions
- **Vector search** with MongoDB Atlas Vector Search
- **Knowledge base management** with sessions
- **Modern FastAPI** web application

## Features

### Core Capabilities

1. **📚 Knowledge Base Management**
   - Ingest documents from text, files, or URLs
   - Automatic semantic chunking with token-aware splitting
   - Multi-session support for isolated knowledge bases
   - Source tracking and browsing

2. **🔍 Vector Search**
   - MongoDB Atlas Vector Search integration
   - Semantic similarity search
   - Configurable result limits and scoring
   - Session-scoped search results

3. **💬 AI-Powered Chat**
   - RAG-based question answering
   - Context-aware responses from knowledge base
   - Chat history management
   - Source attribution

4. **🛠️ Document Processing**
   - Support for text, markdown, PDF, Word, and more (via docling)
   - URL content extraction (via Firecrawl)
   - Chunk preview before ingestion
   - Document conversion and preview

5. **🌐 Web Search Integration**
   - DuckDuckGo web search integration
   - Search result ingestion into knowledge base

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (for containerized setup)
- OR MongoDB running locally (for local setup)
- MDB_RUNTIME installed
- LLM API keys (Azure OpenAI, OpenAI, VoyageAI, or other LiteLLM-supported providers)

## Environment Variables

Create a `.env` file with your API keys:

```bash
# MongoDB (if not using Docker Compose)
MONGO_URI=mongodb://admin:password@mongodb:27017/?authSource=admin
MONGO_DB_NAME=interactive_rag_db

# Azure OpenAI Configuration (recommended)
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/openai/v1/
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_MODEL_NAME=gpt-4o

# Alternative: Standard OpenAI
OPENAI_API_KEY=your-openai-key
VOYAGE_API_KEY=your-voyage-key

# Firecrawl for URL extraction
FIRECRAWL_API_KEY=your-firecrawl-key
```

## Setup

### Using Docker Compose (Recommended)

```bash
# Create .env file with your API keys
cp .env.example .env
# Edit .env and add your API keys

# Start everything
docker-compose up
```

The application will be available at:
- 🌐 **Web Application**: http://localhost:5001

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MONGO_URI="mongodb://localhost:27017/"
export MONGO_DB_NAME="interactive_rag_db"
export OPENAI_API_KEY="your-key"

# Run the application
python web.py
```

## Usage

### 1. Ingest Documents

**Via Text:**
```bash
curl -X POST http://localhost:5001/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Your document text here...",
    "source": "document_1",
    "source_type": "text",
    "session_id": "my_session",
    "chunk_size": 1000,
    "chunk_overlap": 150
  }'
```

**Via File Upload:**
- Use the web UI to upload files (PDF, Word, text, etc.)
- Files are automatically converted and chunked

**Via URL:**
- Use the web UI to enter a URL
- Content is extracted using Firecrawl and ingested

### 2. Chat with Knowledge Base

```bash
curl -X POST http://localhost:5001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic of the documents?",
    "session_id": "my_session",
    "embedding_model": "voyage/voyage-2",
    "rag_params": {
      "num_sources": 3,
      "max_chunk_length": 2000
    }
  }'
```

### 3. Search Knowledge Base

```bash
curl -X POST http://localhost:5001/preview_search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "search query",
    "session_id": "my_session",
    "embedding_model": "voyage/voyage-2",
    "num_sources": 5
  }'
```

## API Endpoints

### Ingestion

- `POST /ingest` - Start ingestion task
- `GET /ingest/status/{task_id}` - Get ingestion status

### Chat

- `POST /chat` - Chat with knowledge base (RAG)

### Search & Preview

- `POST /preview_search` - Preview vector search results
- `POST /preview_file` - Preview file content before ingestion
- `GET /preview_url?url=...` - Preview URL content

### Session Management

- `GET /state` - Get application state (sessions, models)
- `POST /history/clear` - Clear chat history for a session

### Source Management

- `GET /sources?session_id=...` - List all sources in a session
- `GET /chunks?session_id=...&source_url=...` - Get chunks for a source
- `GET /source_content?session_id=...&source=...` - Get full source content

### Chunk Management

- `PUT /chunk/{chunk_id}` - Update a chunk (re-embeds automatically)
- `DELETE /chunk/{chunk_id}` - Delete a chunk

### Utilities

- `POST /search` - Web search (DuckDuckGo)
- `POST /chunk_preview` - Preview how content will be chunked

## Architecture

### Components

1. **RuntimeEngine** - Manages MongoDB connection and app registration
2. **EmbeddingService** - Handles semantic chunking and embedding generation
3. **LLMService** - Provides chat completions via LiteLLM
4. **Vector Search** - MongoDB Atlas Vector Search for semantic retrieval
5. **FastAPI** - Modern async web framework

### Data Flow

```
User Input → EmbeddingService (chunk & embed) → MongoDB
                                                      ↓
User Query → EmbeddingService (embed query) → Vector Search → MongoDB
                                                      ↓
Retrieved Chunks → LLMService (chat with context) → Response
```

### Document Structure

Documents in the knowledge base collection:

```json
{
  "_id": ObjectId("..."),
  "source_id": "document_1",
  "chunk_index": 0,
  "text": "Chunk text content...",
  "embedding": [0.1, 0.2, ...],
  "metadata": {
    "source": "document_1",
    "source_type": "text",
    "session_id": "my_session",
    "model": "voyage/voyage-2",
    "created_at": ISODate("...")
  }
}
```

## Configuration

### Manifest (`manifest.json`)

Key configuration:

```json
{
  "llm_config": {
    "enabled": true,
    "default_chat_model": "gpt-4o",
    "default_embedding_model": "voyage/voyage-2"
  },
  "embedding_config": {
    "enabled": true,
    "max_tokens_per_chunk": 1000,
    "tokenizer_model": "gpt-3.5-turbo"
  },
  "managed_indexes": {
    "knowledge_base_sessions": [
      {
        "type": "vectorSearch",
        "name": "embedding_vector_index",
        "definition": {
          "fields": [
            {
              "type": "vector",
              "path": "embedding",
              "numDimensions": 1024,
              "similarity": "cosine"
            }
          ]
        }
      }
    ]
  }
}
```

### Embedding Models

Supported via LiteLLM:
- `voyage/voyage-2` (default, recommended)
- `text-embedding-3-small`
- `text-embedding-3-large`
- `cohere/embed-english-v3.0`
- And more...

### Chat Models

Supported via LiteLLM:
- `gpt-4o` (default)
- `gpt-4-turbo`
- `claude-3-opus-20240229`
- `gemini/gemini-pro`
- And more...

## Best Practices

### Chunking

- **Chunk Size**: 500-1000 tokens works well for most use cases
- **Overlap**: 10-20% overlap helps maintain context across chunks
- **Semantic Boundaries**: EmbeddingService uses semantic-text-splitter to preserve sentence/paragraph boundaries

### Session Management

- Use separate sessions for different knowledge domains
- Session isolation ensures search results are scoped correctly
- Clear old sessions periodically to manage storage

### Vector Search

- Use `numCandidates = num_sources * 10` for good recall
- Filter by session_id to ensure proper scoping
- Consider hybrid search (vector + text) for better results

### Performance

- Batch embeddings when possible (EmbeddingService does this automatically)
- Use background tasks for large document ingestion
- Monitor MongoDB connection pool usage

## Troubleshooting

### Vector Search Not Working

1. **Check index exists:**
   ```bash
   # In MongoDB shell
   db.knowledge_base_sessions.listSearchIndexes()
   ```

2. **Verify embedding dimensions match index:**
   - VoyageAI: 1024 dimensions
   - OpenAI text-embedding-3-small: 1536 dimensions
   - Update manifest.json if dimensions don't match

### Embedding Generation Fails

1. **Check API keys:**
   ```bash
   echo $OPENAI_API_KEY
   echo $VOYAGE_API_KEY
   ```

2. **Verify LiteLLM configuration:**
   - Model names should use LiteLLM format (e.g., `voyage/voyage-2`)
   - Check LiteLLM documentation for provider-specific setup

### Chunking Issues

1. **Verify semantic-text-splitter is installed:**
   ```bash
   pip install semantic-text-splitter
   ```

2. **Check tokenizer model:**
   - Default `gpt-3.5-turbo` works for most models (uses cl100k_base encoding internally)
   - Must be a valid OpenAI model name (e.g., "gpt-3.5-turbo", "gpt-4", "gpt-4o")
   - Adjust in manifest.json if needed

## Next Steps

- Add LangChain agent tools for more advanced RAG
- Implement hybrid search (vector + text)
- Add document versioning
- Implement source citations in responses
- Add user authentication and multi-user support
- Integrate with more document sources

## Related Examples

- **hello_world** - Basic MDB_RUNTIME usage
- **vector_hacking** - Vector operations and embeddings

## Resources

- [MDB_RUNTIME Documentation](../../README.md)
- [EmbeddingService Documentation](../../mdb_runtime/llm/README.md)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [MongoDB Atlas Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/)

