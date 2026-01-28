#!/usr/bin/env python3
"""
SSO App 3 - AI Chat Application with SSO Authentication & Smart Document Memory
Optimized for Deep Metadata Extraction & Atomic Fact Resolution
"""

import asyncio
import io
import logging
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from bson.objectid import ObjectId
from dotenv import load_dotenv
from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Optional imports
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
try:
    import docx
except ImportError:
    docx = None

from mdb_engine import MongoDBEngine
from mdb_engine.embeddings.service import EmbeddingServiceError, get_embedding_service
from mdb_engine.routing.websockets import broadcast_to_app

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_SLUG = "ai-chat"
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Constants for Concurrency
MAX_CONCURRENT_CHUNKS = 5  # Increased slightly for better throughput
CHUNK_SIZE = 15000  # Large context window usage
CHUNK_OVERLAP = 1000

# --- IMPROVED STRUCTURED DATA MODELS ---


class DocumentMetadata(BaseModel):
    """Rich metadata extracted from the document header/summary."""

    title: str = Field(description="The official title of the document.")
    author: str | None = Field(
        description=(
            "The specific person, department, or entity who wrote the document. "
            "Look for 'Prepared by', 'Author', or bylines."
        )
    )
    organization: str | None = Field(
        description="The company or organization the document belongs to."
    )
    version: str | None = Field(
        description="Version number (e.g., '1.0', 'Draft', 'Final') if available."
    )
    creation_date: str | None = Field(
        description="The specific date the document was created or last modified."
    )
    summary: str = Field(
        description="A comprehensive 3-sentence summary of the document's purpose."
    )
    main_entities: list[str] = Field(
        description=(
            "List of the primary projects, products, or people discussed "
            "(e.g., 'Project Apollo', 'iPhone 15')."
        )
    )


class AtomicFact(BaseModel):
    """A single, self-contained fact optimized for vector retrieval."""

    statement: str = Field(
        description=(
            "The fact statement. MUST be rewritten to be standalone. "
            "Resolve all pronouns to specific names provided in the context."
        )
    )
    category: str = Field(
        description=(
            "Category of the fact: 'Financial', 'Technical', 'Legal', "
            "'Schedule', 'Personnel', or 'General'."
        )
    )
    importance: int = Field(
        description="1-10 score. 10 = Critical decision/deadline/cost. 1 = Minor detail."
    )
    entities: list[str] = Field(
        description="List of specific named entities mentioned in this fact."
    )


class ChunkInsights(BaseModel):
    """Insights extracted from a specific segment of text."""

    facts: list[AtomicFact] = Field(description="List of atomic facts extracted from this segment.")


# --- HELPER FUNCTIONS ---


def semantic_chunking(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Splits text into large chunks for efficient LLM processing."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        if end < text_len:
            # Try to split at paragraph or newline
            last_newline = text.rfind("\n", start, end)
            if last_newline != -1 and last_newline > start + (chunk_size // 2):
                end = last_newline + 1
            else:
                last_space = text.rfind(" ", start, end)
                if last_space != -1:
                    end = last_space + 1

        chunks.append(text[start:end])
        start = end - overlap

    return chunks


# Raw Content Service
class RawContentService:
    """Service for storing and retrieving raw content in a separate MongoDB collection."""

    def __init__(self, engine: MongoDBEngine, app_slug: str, config: dict):
        self.engine = engine
        self.app_slug = app_slug
        self.collection_name = config.get("collection_name", f"{app_slug}_raw_content")
        self.enabled = config.get("enabled", True)

        if not self.enabled:
            logger.info(f"Raw content service disabled for {app_slug}")
            self.embedding_service = None
            return

        embedding_config = {
            "embedding_model": config.get("embedding_model", "text-embedding-3-small"),
            "embedding_model_dims": config.get("embedding_model_dims", 1536),
        }
        try:
            self.embedding_service = get_embedding_service(config=embedding_config)
            logger.info(f"✅ Raw Content Service initialized: {self.collection_name}")
        except (
            EmbeddingServiceError,
            ValueError,
            RuntimeError,
            ImportError,
            AttributeError,
        ) as e:
            logger.warning(f"⚠️ Failed to initialize Raw Content Service: {e}")
            self.embedding_service = None

    async def store_raw_content(
        self, raw_content: str, user_id: str, bucket_id: str, metadata: dict | None = None
    ) -> str | None:
        if not self.enabled or not self.embedding_service:
            return None
        try:
            db = self.engine.get_scoped_db(self.app_slug)
            collection = getattr(db, self.collection_name)
            embeddings = await self.embedding_service.embed(raw_content)
            if not embeddings:
                return None

            doc = {
                "id": str(uuid4()),
                "user_id": str(user_id),
                "bucket_id": bucket_id,
                "associated_bucket_id": bucket_id,
                "content": raw_content,
                "embedding": embeddings[0],
                "metadata": metadata or {},
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
            await collection.insert_one(doc)
            return doc["id"]
        except Exception as e:
            logger.error(f"Failed to store raw content: {e}", exc_info=True)
            return None

    async def get_raw_content(self, bucket_id: str, user_id: str) -> str | None:
        if not self.enabled:
            return None
        try:
            db = self.engine.get_scoped_db(self.app_slug)
            doc = await getattr(db, self.collection_name).find_one(
                {"bucket_id": bucket_id, "user_id": str(user_id)}, sort=[("created_at", -1)]
            )
            return doc.get("content") if doc else None
        except Exception:  # noqa: BLE001
            return None


# Initialize global vars
raw_content_service: RawContentService | None = None
engine = MongoDBEngine(
    mongo_uri=os.getenv("MONGO_URI", "mongodb://mongodb:27017/"),
    db_name=os.getenv("MONGO_DB_NAME", "oblivio_apps"),
)


async def on_startup(app, engine, manifest):
    global raw_content_service
    raw_content_config = manifest.get("raw_content_config", {})
    if raw_content_config.get("enabled", False):
        raw_content_service = RawContentService(engine, APP_SLUG, raw_content_config)
    try:
        engine.register_websocket_routes(app, APP_SLUG)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to register WebSocket routes: {e}")
    logger.info("AI Chat application ready!")


app = engine.create_app(
    slug=APP_SLUG,
    manifest=Path(__file__).parent / "manifest.json",
    title="AI Chat",
    on_startup=on_startup,
)

# --- CORE LOGIC (AI PROCESSING) ---


async def convert_file_to_markdown(file: UploadFile) -> dict:
    filename = file.filename.lower()
    content_bytes = await file.read()
    file_obj = io.BytesIO(content_bytes)
    result = {"filename": file.filename, "content": "", "raw_text": "", "type": "unknown"}

    try:
        if filename.endswith(".docx") and docx:
            result["type"] = "document"
            doc = docx.Document(file_obj)
            text = "\n".join([p.text for p in doc.paragraphs])
            result["raw_text"] = text
            result["content"] = f"### Document: {file.filename}\n\n{text}"
        elif filename.endswith(".pdf") and PdfReader:
            result["type"] = "pdf"
            reader = PdfReader(file_obj)
            text = "\n".join([p.extract_text() or "" for p in reader.pages])
            result["raw_text"] = text
            result["content"] = f"### PDF: {file.filename}\n{text}"
        elif filename.endswith((".xlsx", ".csv")) and pd:
            result["type"] = "spreadsheet"
            df = pd.read_csv(file_obj) if filename.endswith(".csv") else pd.read_excel(file_obj)
            result["raw_text"] = df.to_csv(index=False)
            result["content"] = (
                f"### Data: {file.filename}\n\n{df.head(50).to_markdown(index=False)}"
            )
        else:
            result["type"] = "code"
            text = content_bytes.decode("utf-8", errors="ignore")
            result["raw_text"] = text
            result["content"] = f"### File: {file.filename}\n```\n{text}\n```"
    except Exception as e:
        logger.exception(f"Error reading file {file.filename}")
        result["content"] = f"[Error reading {file.filename}: {e}]"
    return result


async def extract_global_metadata(
    client: genai.Client, text: str, filename: str
) -> DocumentMetadata:
    """Extracts high-level metadata from the beginning of the file."""
    intro_text = text[:20000]  # Increased window to catch metadata at end of intro sections

    prompt = f"""You are an expert document archivist. Analyze this text to extract METADATA.

    CRITICAL: Look for specific details.
    - Author: Look for "Prepared by", "Written by", bylines, or email signatures.
    - Version: Look for "v1.0", "Draft", "Final", "Confidential".
    - Organization: Look for company names in headers or footers.
    - Date: Look for the specific document creation date.

    Filename: {filename}
    Content Snippet:
    {intro_text}
    """

    try:
        resp = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=DocumentMetadata,
            ),
        )
        return resp.parsed
    except Exception:
        logger.exception("Metadata extraction failed")
        # Return safe defaults
        return DocumentMetadata(
            title=filename,
            author="Unknown",
            organization="Unknown",
            version=None,
            creation_date=None,
            summary="Processing failed.",
            main_entities=[],
        )


async def extract_facts_from_chunk(
    client: genai.Client,
    chunk: str,
    chunk_index: int,
    doc_metadata: DocumentMetadata,
    semaphore: asyncio.Semaphore,
) -> list[AtomicFact]:
    """Extracts detailed facts from a specific chunk, constrained by semaphore."""

    async with semaphore:
        # Long prompt string - break to avoid line length issues
        prompt = (
            f"""You are an expert analyst. Extract INDEPENDENT, ATOMIC facts """
            f"""from the text segment below.

        GLOBAL CONTEXT (Use this to resolve pronouns):
        - Document: "{doc_metadata.title}"
        - Author: {doc_metadata.author or 'Unknown'}
        - Organization: {doc_metadata.organization or 'Unknown'}
        - Key Entities: {', '.join(doc_metadata.main_entities)}

        INSTRUCTIONS:
        1. **Resolve Pronouns**: NEVER use "He", "She", "It", "They", """
            + f""""The author", "The company".
           - BAD: "He expects revenue to grow."
           - GOOD: "{doc_metadata.author or 'The author'} expects revenue to grow."
           - BAD: "It will cost $1M."
           - GOOD: (
               f"Project "
               f"{doc_metadata.main_entities[0] if doc_metadata.main_entities else 'X'} "
               "will cost $1M."
           )

        2. **Be Specific**: Include numbers, dates, and exact names.

        3. **Filter**: Only extract facts with importance score > 4. Ignore fluff.

        SEGMENT:
        {chunk}
        """
        )

        try:
            resp = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=ChunkInsights,
                ),
            )
            return resp.parsed.facts if resp.parsed else []
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Chunk {chunk_index} failed: {e}")
            return []


async def process_and_store_file_memory(
    svc, user_id: str, file_data: dict, category: str, associated_bucket_id: str = None
) -> int:
    """
    Orchestrates the parallel processing of a file with enhanced metadata injection.
    """
    filename = file_data["filename"]
    raw_text = file_data["raw_text"]
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # 1. Global Metadata Extraction (First Pass)
    await broadcast_to_app(
        APP_SLUG,
        {
            "type": "memory_progress",
            "stage": "analyzing_metadata",
            "message": f"Identifying author and context for {filename}...",
            "filename": filename,
            "user_id": user_id,
        },
        user_id=None,
    )

    doc_metadata = await extract_global_metadata(gemini_client, raw_text, filename)
    logger.info(
        f"📋 Metadata for {filename}: Author={doc_metadata.author}, Org={doc_metadata.organization}"
    )

    # Bucket IDs
    file_bucket_id = f"file:{filename}:{user_id}"
    cat_bucket_id = associated_bucket_id or (
        f"bucket:{category}:{user_id}" if category != "general" else f"bucket:general:{user_id}"
    )

    # 2. Store Raw Content (Vector DB) with Rich Metadata
    if raw_content_service:
        await raw_content_service.store_raw_content(
            raw_content=raw_text,
            user_id=user_id,
            bucket_id=file_bucket_id,
            metadata={
                "filename": filename,
                "associated_bucket_id": cat_bucket_id,
                "category": category,
                "title": doc_metadata.title,
                "author": doc_metadata.author,
                "organization": doc_metadata.organization,
                "version": doc_metadata.version,
                "summary": doc_metadata.summary,
                "topics": doc_metadata.main_entities,
                "doc_date": doc_metadata.creation_date,
            },
        )

    # 3. Parallel Fact Extraction (Second Pass)
    chunks = semantic_chunking(raw_text)
    total_chunks = len(chunks)

    await broadcast_to_app(
        APP_SLUG,
        {
            "type": "memory_progress",
            "stage": "extracting_facts",
            "message": f"Extracting atomic facts from {total_chunks} segments...",
            "filename": filename,
            "user_id": user_id,
        },
        user_id=None,
    )

    # Create tasks for parallel execution
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)
    tasks = [
        extract_facts_from_chunk(gemini_client, chunk, i, doc_metadata, semaphore)
        for i, chunk in enumerate(chunks)
    ]

    # Run all chunks
    results = await asyncio.gather(*tasks)

    # Flatten results
    all_facts: list[AtomicFact] = [fact for sublist in results for fact in sublist]

    # Filter high value facts and deduplicate
    unique_facts = []
    seen_statements = set()

    for f in all_facts:
        # Strict deduplication
        if f.statement not in seen_statements and f.importance >= 5:
            seen_statements.add(f.statement)
            unique_facts.append(f)

    # CRITICAL: Explicitly add author information as a fact if available
    # This ensures author queries can be found via semantic search
    if doc_metadata.author and doc_metadata.author != "Unknown":
        author_fact = f"The authors of '{doc_metadata.title}' are {doc_metadata.author}."
        if author_fact not in seen_statements:
            unique_facts.append(
                AtomicFact(
                    statement=author_fact,
                    category="Personnel",
                    importance=8,  # High importance for authorship
                    entities=[doc_metadata.author, doc_metadata.title],
                )
            )
            seen_statements.add(author_fact)
            logger.info(f"📝 Added explicit author fact: {author_fact}")

    # 4. Batch Store in Mem0
    stored_count = 0
    if unique_facts:
        await broadcast_to_app(
            APP_SLUG,
            {
                "type": "memory_progress",
                "stage": "storing_memories",
                "message": f"Saving {len(unique_facts)} specific facts...",
                "filename": filename,
                "user_id": user_id,
            },
            user_id=None,
        )

        # Common metadata for all facts in this file
        # We attach the AUTHOR and ORGANIZATION to every single memory!
        common_metadata = {
            "filename": filename,
            "associated_bucket_id": cat_bucket_id,
            "category": category,
            "doc_title": doc_metadata.title,
            "doc_author": doc_metadata.author,
            "doc_org": doc_metadata.organization,
            "doc_version": doc_metadata.version,
            "doc_date": doc_metadata.creation_date,
            "extracted_fact": True,
        }

        # Batch insert simulation (as Mem0 add is singular usually)
        for fact in unique_facts:
            try:
                # We append fact-specific tags to the common metadata
                fact_metadata = common_metadata.copy()
                fact_metadata.update(
                    {
                        "fact_category": fact.category,
                        "fact_importance": fact.importance,
                        "entities": fact.entities,
                    }
                )

                await asyncio.to_thread(
                    svc.add,
                    messages=[{"role": "user", "content": fact.statement}],
                    user_id=user_id,
                    bucket_id=file_bucket_id,
                    bucket_type="file",
                    metadata=fact_metadata,
                    infer=False,  # CRITICAL: We already extracted the atomic fact.
                    # Don't summarize it.
                )
                stored_count += 1
            except Exception:
                logger.exception("Fact storage error")

    # 5. Final Broadcast
    await broadcast_to_app(
        APP_SLUG,
        {
            "type": "memory_stored",
            "memory_count": stored_count,
            "task_completed": False,
            "filename": filename,
            "message": (
                f"Analyzed {filename}: Found {stored_count} facts "
                f"(Author: {doc_metadata.author or 'Unknown'})"
            ),
            "user_id": user_id,
        },
        user_id=None,
    )

    return stored_count


# --- ROUTES ---


def get_current_user(request: Request):
    return getattr(request.state, "user", None)


def get_auth_hub_url() -> str:
    return os.getenv("AUTH_HUB_URL", "http://localhost:8000")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    if get_current_user(request):
        return RedirectResponse("/conversations")
    callback = f"{request.url.scheme}://{request.url.hostname}:{request.url.port}/auth/callback"
    return RedirectResponse(f"{get_auth_hub_url()}/login?redirect_to={callback}")


@app.get("/auth/callback")
async def auth_callback(request: Request, token: str = None):
    if not token:
        token = request.query_params.get("token")
    response = RedirectResponse("/", status_code=302)
    response.set_cookie("mdb_auth_token", token, httponly=True)
    return response


@app.get("/conversations", response_class=HTMLResponse)
async def conversations_list(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/")
    db = engine.get_scoped_db(APP_SLUG)
    convos = (
        await db.conversations.find({"user_id": str(user["_id"])})
        .sort("updated_at", -1)
        .to_list(100)
    )
    return templates.TemplateResponse(
        request, "conversations.html", {"user": user, "conversations": convos}
    )


@app.get("/conversations/{cid}", response_class=HTMLResponse)
async def conversation_view(request: Request, cid: str):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/")
    db = engine.get_scoped_db(APP_SLUG)
    convo = await db.conversations.find_one({"_id": ObjectId(cid), "user_id": str(user["_id"])})
    if not convo:
        return RedirectResponse("/conversations")
    msgs = await db.messages.find({"conversation_id": cid}).sort("created_at", 1).to_list(1000)
    return templates.TemplateResponse(
        request, "conversation.html", {"user": user, "conversation": convo, "messages": msgs}
    )


@app.post("/api/conversations", response_class=JSONResponse)
async def create_convo(request: Request):
    user = get_current_user(request)
    db = engine.get_scoped_db(APP_SLUG)
    res = await db.conversations.insert_one(
        {
            "user_id": str(user["_id"]),
            "title": "New Chat",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    )
    return JSONResponse({"success": True, "conversation": {"_id": str(res.inserted_id)}})


@app.post("/api/conversations/{cid}/messages", response_class=JSONResponse)
async def send_message(
    request: Request,
    cid: str,
    message: str = Form(""),
    category: str = Form("general"),
    files: list[UploadFile] = File(default=[]),
):
    user = get_current_user(request)
    if not user:
        raise HTTPException(401)
    user_id = str(user["_id"])
    db = engine.get_scoped_db(APP_SLUG)
    svc = engine.get_memory_service(APP_SLUG)

    # 1. Process Files for Chat Context
    file_context = ""
    processed_files = []
    file_list = files or []
    if not file_list:
        try:
            form = await request.form()
            file_list = [v for v in form.getlist("files") if isinstance(v, UploadFile)]
        except Exception:  # noqa: BLE001
            pass

    for f in file_list:
        if f.filename:
            data = await convert_file_to_markdown(f)
            if data["raw_text"]:
                processed_files.append(data)
                file_context += f"\n{data['content']}"

    # 2. Store User Msg
    full_input = message + file_context
    await db.messages.insert_one(
        {
            "conversation_id": cid,
            "user_id": user_id,
            "role": "user",
            "content": full_input,
            "created_at": datetime.utcnow(),
        }
    )

    # 3. RAG
    rag_context = []
    retrieved_memories = []
    if svc and message.strip():
        try:
            # Detect if query is about authors
            query_lower = message.lower()
            is_author_query = any(
                keyword in query_lower
                for keyword in ["author", "who wrote", "who created", "who is the author"]
            )

            # Semantic Search
            mems = await asyncio.to_thread(
                svc.search, query=message[:500], user_id=user_id, limit=12
            )

            # If author query and few results, also search by metadata
            if is_author_query and len(mems) < 5:
                # Get all memories and filter for those with author metadata
                all_mems = await asyncio.to_thread(svc.get_all, user_id=user_id, limit=200)
                author_mems = [
                    m
                    for m in all_mems
                    if m.get("metadata", {}).get("doc_author")
                    and m.get("metadata", {}).get("doc_author") != "Unknown"
                ]
                # Merge with semantic results, prioritizing semantic matches
                existing_ids = {
                    m.get("id") or m.get("_id") for m in mems if m.get("id") or m.get("_id")
                }
                for am in author_mems:
                    if (am.get("id") or am.get("_id")) not in existing_ids:
                        mems.append(am)
                mems = mems[:15]  # Limit total results

            # Smart Filtering: Match category OR semantic relevance
            if category != "general" and mems:
                filtered = []
                for m in mems:
                    meta = m.get("metadata", {})
                    # Keep if category matches OR if score is very high (>0.85 approx)
                    if meta.get("category") == category or meta.get("category") == "general":
                        filtered.append(m)

                if filtered:
                    mems = filtered[:10]

            # Build enriched RAG context with memory text AND metadata
            rag_context = []
            for m in mems:
                memory_text = m.get("memory")
                if not memory_text:
                    continue

                meta = m.get("metadata", {})
                # Include relevant metadata in the context for the AI
                enriched_memory = memory_text

                # Add document metadata if available (helps with author/title queries)
                doc_info_parts = []
                if meta.get("doc_author") and meta.get("doc_author") != "Unknown":
                    doc_info_parts.append(f"Author: {meta['doc_author']}")
                if meta.get("doc_title"):
                    doc_info_parts.append(f"Document: {meta['doc_title']}")
                if meta.get("doc_org") and meta.get("doc_org") != "Unknown":
                    doc_info_parts.append(f"Organization: {meta['doc_org']}")

                if doc_info_parts:
                    enriched_memory = (
                        f"[Document Context: {', '.join(doc_info_parts)}]\n{memory_text}"
                    )

                rag_context.append(enriched_memory)

            retrieved_memories = mems
            logger.info(f"🔍 RAG: Found {len(mems)} memories")
        except Exception as e:
            logger.error(f"RAG search failed: {e}", exc_info=True)

    # 4. Generate AI Response
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Enhanced prompt that explicitly instructs the AI to use metadata
    memory_context_str = (
        "\n".join([f"- {mem}" for mem in rag_context])
        if rag_context
        else "No relevant memories found."
    )

    # Long prompt - break IMPORTANT line to avoid E501
    important_note = (
        "IMPORTANT: When answering questions about document authors, titles, or metadata, "
        "pay special attention to the [Document Context] information in the memories above. "
        "This context includes author names, document titles, and organizations."
    )
    prompt = f"""System: You are Orby, an AI assistant with access to stored memories.

MEMORY CONTEXT (use this information to answer questions):
{memory_context_str}

{important_note}

User: {full_input}"""
    try:
        resp = await asyncio.to_thread(
            client.models.generate_content, model="gemini-2.0-flash", contents=prompt
        )
        ai_text = resp.text
    except Exception as e:  # noqa: BLE001
        ai_text = f"AI Error: {e}"

    await db.messages.insert_one(
        {
            "conversation_id": cid,
            "user_id": user_id,
            "role": "assistant",
            "content": ai_text,
            "created_at": datetime.utcnow(),
        }
    )

    # 5. Background Memory Task
    if svc and (processed_files or len(message) > 10):

        async def store_task():
            total_memories = 0
            errors = []
            try:
                # Chat Memory
                if len(message) > 10:
                    chat_bucket_id = (
                        f"bucket:{category}:{user_id}" if category != "general" else None
                    )
                    await asyncio.to_thread(
                        svc.add,
                        messages=[
                            {"role": "user", "content": message},
                            {"role": "assistant", "content": ai_text},
                        ],
                        user_id=user_id,
                        bucket_id=chat_bucket_id,
                        bucket_type="general",
                        metadata={"conversation_id": cid, "category": category},
                    )

                # File Memory
                for pf in processed_files:
                    try:
                        count = await process_and_store_file_memory(
                            svc=svc, user_id=user_id, file_data=pf, category=category
                        )
                        total_memories += count
                    except Exception as e:
                        error_msg = f"Error processing {pf.get('filename')}: {e}"
                        logger.error(f"❌ {error_msg}", exc_info=True)
                        errors.append(error_msg)

                await broadcast_to_app(
                    APP_SLUG,
                    {
                        "type": "memory_stored",
                        "memory_count": total_memories,
                        "task_completed": True,
                        "filename": None,
                        "message": f"Finished processing. {total_memories} insights stored.",
                        "user_id": user_id,
                        "errors": errors if errors else None,
                    },
                    user_id=None,
                )

            except Exception as e:
                logger.error(f"❌ Background task failed: {e}", exc_info=True)

        task = asyncio.create_task(store_task())
        task.add_done_callback(
            lambda t: t.exception() and logger.error(f"Task Error: {t.exception()}")
        )

    return JSONResponse(
        {
            "success": True,
            "message": {"content": ai_text},
            "memory_context": {
                "used_memories": len(rag_context),
                "retrieved_memories": normalize_memories(retrieved_memories),
            },
        }
    )


@app.get("/api/buckets/{bucket_id}/files", response_class=JSONResponse)
async def get_bucket_files(request: Request, bucket_id: str):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"error": "Auth required"}, status_code=401)
    svc = engine.get_memory_service(APP_SLUG)
    if not svc:
        return JSONResponse({"files": []})

    user_id = str(user["_id"])
    files_list = []

    if bucket_id.startswith("file:"):
        parts = bucket_id.split(":")
        if len(parts) >= 2:
            mems = await asyncio.to_thread(
                svc.get_all, user_id=user_id, filters={"bucket_id": bucket_id}, limit=1
            )
            if mems:
                files_list.append(
                    {
                        "filename": parts[1],
                        "bucket_id": bucket_id,
                        "memory_count": len(mems),
                        "upload_date": mems[0].get("created_at") or "Unknown",
                    }
                )
    else:
        # Get by association
        mems = await asyncio.to_thread(
            svc.get_all, user_id=user_id, filters={"associated_bucket_id": bucket_id}, limit=500
        )
        seen = set()
        for m in mems:
            meta = m.get("metadata", {})
            f_bucket = meta.get("bucket_id")
            if (
                meta.get("filename")
                and f_bucket
                and f_bucket.startswith("file:")
                and f_bucket not in seen
            ):
                files_list.append(
                    {
                        "filename": meta.get("filename"),
                        "bucket_id": f_bucket,
                        "upload_date": meta.get("created_at") or "Unknown",
                        "author": meta.get("doc_author", "Unknown"),  # Return author to frontend
                    }
                )
                seen.add(f_bucket)

    return JSONResponse({"success": True, "files": files_list})


@app.post("/api/buckets/{bucket_id}/files", response_class=JSONResponse)
async def add_file_to_bucket(
    request: Request, bucket_id: str, files: list[UploadFile] = File(default=[])
):
    user = get_current_user(request)
    if not user:
        raise HTTPException(401)
    svc = engine.get_memory_service(APP_SLUG)
    user_id = str(user["_id"])

    file_list = files or []
    if not file_list:
        try:
            form = await request.form()
            file_list = [v for v in form.getlist("files") if isinstance(v, UploadFile)]
        except Exception:  # noqa: BLE001
            pass
    if not file_list:
        return JSONResponse({"success": False, "error": "No files provided"})

    parts = bucket_id.split(":")
    category = parts[1] if len(parts) > 1 and parts[0] == "bucket" else "general"
    processed_count = 0

    for f in file_list:
        try:
            data = await convert_file_to_markdown(f)
            if data["raw_text"]:
                await process_and_store_file_memory(
                    svc=svc,
                    user_id=user_id,
                    file_data=data,
                    category=category,
                    associated_bucket_id=bucket_id,
                )
                processed_count += 1
        except Exception:
            logger.exception("Failed to add file to bucket")

    return JSONResponse({"success": True, "processed": processed_count})


@app.get("/api/memories", response_class=JSONResponse)
async def get_all_memories(request: Request, limit: int = 500):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"error": "Auth required"}, status_code=401)
    svc = engine.get_memory_service(APP_SLUG)
    if not svc:
        return JSONResponse({"success": True, "memories": [], "count": 0})

    memories = await asyncio.to_thread(svc.get_all, user_id=str(user["_id"]), limit=limit)
    normalized = normalize_memories(memories)
    return JSONResponse({"success": True, "memories": normalized, "count": len(normalized)})


@app.get("/api/memories/stats", response_class=JSONResponse)
async def get_memory_stats(request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"success": False})
    svc = engine.get_memory_service(APP_SLUG)
    if not svc:
        return JSONResponse({"success": False, "error": "No service"})

    all_mems = await asyncio.to_thread(svc.get_all, user_id=str(user["_id"]), limit=2000)
    stats = {"file_contexts": {}, "general_buckets": {}, "bucket_files": {}}
    buckets = {}

    for m in all_mems:
        meta = m.get("metadata", {})
        bid = meta.get("bucket_id") or meta.get("context_id")
        if bid:
            if bid not in buckets:
                buckets[bid] = {
                    "bucket_id": bid,
                    "bucket_type": meta.get("bucket_type", "general"),
                    "memory_count": 0,
                    "metadata": meta,
                }
            buckets[bid]["memory_count"] += 1

    for b in buckets.values():
        bid = b["bucket_id"]
        meta = b.get("metadata", {})
        if b["bucket_type"] == "file":
            fname = meta.get("filename") or "Unknown"
            stats["file_contexts"][fname] = {
                "context_id": bid,
                "count": b["memory_count"],
                "bucket_type": "file",
            }
            assoc = meta.get("associated_bucket_id")
            if assoc:
                if assoc not in stats["bucket_files"]:
                    stats["bucket_files"][assoc] = {}
                stats["bucket_files"][assoc][bid] = {
                    "filename": fname,
                    "bucket_id": bid,
                    "memory_count": b["memory_count"],
                    "author": meta.get("doc_author", "Unknown"),
                }
        else:
            cat = meta.get("category", "General")
            stats["general_buckets"][bid] = {"name": cat, "count": b["memory_count"]}

    stats["bucket_files"] = {k: list(v.values()) for k, v in stats["bucket_files"].items()}
    return JSONResponse({"success": True, "stats": stats})


@app.get("/api/memories/by-context", response_class=JSONResponse)
async def get_memories_by_context(request: Request, bucket_id: str, limit: int = 100):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"error": "Auth required"}, status_code=401)
    svc = engine.get_memory_service(APP_SLUG)
    mems = await asyncio.to_thread(
        svc.get_all, user_id=str(user["_id"]), filters={"bucket_id": bucket_id}, limit=limit
    )
    normalized = normalize_memories(mems)
    return JSONResponse({"success": True, "memories": normalized, "memoryCount": len(normalized)})


@app.get("/api/memories/search", response_class=JSONResponse)
async def search_memories(
    request: Request, query: str, limit: int = 50, context_id: str | None = None
):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"error": "Auth required"}, status_code=401)
    svc = engine.get_memory_service(APP_SLUG)
    if not svc:
        return JSONResponse({"success": True, "results": [], "count": 0})

    filters = {"bucket_id": context_id} if context_id else None
    results = await asyncio.to_thread(
        svc.search, query=query, user_id=str(user["_id"]), limit=limit, filters=filters
    )

    normalized_results = []
    if isinstance(results, list):
        for res in results:
            if isinstance(res, dict):
                normalized_results.append(
                    {
                        "memory": res.get("memory")
                        or res.get("data", {}).get("memory")
                        or res.get("text")
                        or str(res),
                        "id": res.get("id") or res.get("_id"),
                        "metadata": res.get("metadata", {}),
                        "score": res.get("score"),
                    }
                )
            elif isinstance(res, str):
                normalized_results.append({"memory": res})

    return JSONResponse(
        {
            "success": True,
            "results": normalized_results,
            "count": len(normalized_results),
            "query": query,
        }
    )


@app.get("/api/memories/{memory_id}/raw", response_class=JSONResponse)
async def get_memory_raw(request: Request, memory_id: str):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"error": "Auth required"}, status_code=401)
    svc = engine.get_memory_service(APP_SLUG)
    user_id = str(user["_id"])

    memory = await asyncio.to_thread(svc.get, memory_id=memory_id, user_id=user_id)
    bucket_id = memory_id
    if memory:
        bucket_id = memory.get("metadata", {}).get("bucket_id") or memory.get("metadata", {}).get(
            "associated_bucket_id"
        )

    raw_content = None
    filename = None
    if raw_content_service and bucket_id:
        raw_content = await raw_content_service.get_raw_content(
            bucket_id=bucket_id, user_id=user_id
        )

    if not raw_content and memory:
        raw_content = memory.get("metadata", {}).get("raw_content")
        filename = memory.get("metadata", {}).get("filename")

    if not raw_content:
        return JSONResponse({"success": False, "error": "Raw content not available"})
    return JSONResponse({"success": True, "raw_content": raw_content, "filename": filename})


def normalize_memories(memories):
    norm = []
    if not isinstance(memories, list):
        return norm
    for m in memories:
        if not isinstance(m, dict):
            continue
        metadata = m.get("metadata", {})
        txt = (
            m.get("memory")
            or m.get("text")
            or m.get("content")
            or (
                m.get("messages", [{}])[0].get("content")
                if isinstance(m.get("messages"), list)
                else None
            )
            or m.get("data", {}).get("memory", "")
            or m.get("data", {}).get("text", "")
            or metadata.get("raw_content")
            or str(m)
        )
        if txt:
            norm.append({"memory": txt, "id": m.get("id") or m.get("_id"), "metadata": metadata})
    return norm


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
