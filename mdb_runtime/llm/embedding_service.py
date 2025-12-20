"""
Semantic Text Splitting and Embedding Service

This module provides intelligent text chunking and embedding capabilities:
1. Semantic text splitting using Rust-based semantic-text-splitter
2. Embedding generation via LiteLLM (VoyageAI, OpenAI, Cohere, etc.)
3. MongoDB storage with proper document structure

Key Features:
- Token-aware chunking (never exceeds model limits)
- Semantic boundary preservation (splits on sentences/paragraphs)
- Provider-agnostic embeddings (switch models via config)
- Batch processing for efficiency
- Automatic metadata tracking
- Platform-level defaults (users don't need to configure tokenizer - defaults to "gpt-3.5-turbo")

Dependencies:
    pip install semantic-text-splitter litellm
"""

import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Optional dependencies
try:
    from semantic_text_splitter import TextSplitter
    SEMANTIC_SPLITTER_AVAILABLE = True
except ImportError:
    SEMANTIC_SPLITTER_AVAILABLE = False
    TextSplitter = None

from .service import LLMService, LLMServiceError

logger = logging.getLogger(__name__)


class EmbeddingServiceError(Exception):
    """Base exception for embedding service failures."""
    pass


class EmbeddingService:
    """
    Service for semantic text splitting and embedding generation.
    
    This service combines:
    1. Semantic text splitting (Rust-based, fast and accurate)
    2. Embedding generation (via LiteLLM, provider-agnostic)
    3. MongoDB storage (structured document format)
    
    Example:
        from mdb_runtime.llm import EmbeddingService
        from mdb_runtime.llm import LLMService
        
        # Initialize
        llm_service = LLMService()
        embedding_service = EmbeddingService(llm_service)
        
        # Process and store
        await embedding_service.process_and_store(
            text_content="Your long document here...",
            source_id="doc_101",
            collection=db.knowledge_base,
            max_tokens_per_chunk=1000
        )
    """
    
    def __init__(
        self,
        llm_service: LLMService,
        default_max_tokens: int = 1000,
        default_tokenizer_model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize Embedding Service.
        
        Args:
            llm_service: LLMService instance for embedding generation
            default_max_tokens: Default max tokens per chunk (default: 1000)
            default_tokenizer_model: Tokenizer model name for counting tokens (default: "gpt-3.5-turbo").
                                    This is ONLY for token counting during chunking, NOT for embeddings.
                                    Must be a valid OpenAI model name (e.g., "gpt-3.5-turbo", "gpt-4").
        
        Raises:
            EmbeddingServiceError: If required dependencies are not available
        """
        if not SEMANTIC_SPLITTER_AVAILABLE:
            raise EmbeddingServiceError(
                "semantic-text-splitter not available. Install with: "
                "pip install semantic-text-splitter"
            )
        
        # LLMService already checks for litellm availability
        # We just need to ensure it's initialized
        if not llm_service:
            raise EmbeddingServiceError("LLMService instance is required")
        
        self.llm_service = llm_service
        self.default_max_tokens = default_max_tokens
        self.default_tokenizer_model = default_tokenizer_model
    
    def _create_splitter(
        self,
        max_tokens: int,
        tokenizer_model: Optional[str] = None
    ) -> TextSplitter:
        """
        Create a TextSplitter instance.
        
        Args:
            max_tokens: Maximum tokens per chunk
            tokenizer_model: Tokenizer encoding for counting (default: uses default_tokenizer_model).
                            This is ONLY for token counting, NOT for embeddings.
        
        Returns:
            TextSplitter instance
        """
        # Use provided tokenizer, or fall back to default (gpt-3.5-turbo)
        model = tokenizer_model or self.default_tokenizer_model
        return TextSplitter.from_tiktoken_model(model, max_tokens)
    
    async def chunk_text(
        self,
        text_content: str,
        max_tokens: Optional[int] = None,
        tokenizer_model: Optional[str] = None
    ) -> List[str]:
        """
        Split text into semantic chunks.
        
        Uses Rust-based semantic-text-splitter for fast, accurate chunking
        that respects token limits and semantic boundaries.
        
        Args:
            text_content: The text to chunk
            max_tokens: Max tokens per chunk (default: uses default_max_tokens)
            tokenizer_model: Tokenizer model name for counting (optional, defaults to "gpt-3.5-turbo").
                            This is ONLY for token counting, NOT for embeddings.
                            Must be a valid OpenAI model name (e.g., "gpt-3.5-turbo", "gpt-4").
        
        Returns:
            List of text chunks
        
        Example:
            chunks = await service.chunk_text("Long document...", max_tokens=1000)
            print(f"Generated {len(chunks)} chunks")
        """
        max_tokens = max_tokens or self.default_max_tokens
        splitter = self._create_splitter(max_tokens, tokenizer_model)
        
        try:
            chunks = splitter.chunks(text_content)
            logger.info(f"Generated {len(chunks)} chunks (max_tokens={max_tokens})")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {e}", exc_info=True)
            raise EmbeddingServiceError(f"Chunking failed: {str(e)}") from e
    
    async def embed_chunks(
        self,
        chunks: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for text chunks.
        
        Uses LiteLLM for provider-agnostic embedding generation.
        Supports VoyageAI, OpenAI, Cohere, and more.
        
        Args:
            chunks: List of text chunks to embed
            model: Embedding model (default: uses LLMService default)
                   e.g., "voyage/voyage-2", "text-embedding-3-small"
        
        Returns:
            List of embedding vectors (each is a list of floats)
        
        Example:
            chunks = ["chunk 1", "chunk 2"]
            vectors = await service.embed_chunks(chunks, model="voyage/voyage-2")
        """
        if not chunks:
            return []
        
        try:
            # Use LLMService's embed method (handles retries, logging, etc.)
            vectors = await self.llm_service.embed(chunks, model=model)
            logger.info(f"Generated {len(vectors)} embeddings")
            return vectors
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}", exc_info=True)
            raise EmbeddingServiceError(f"Embedding generation failed: {str(e)}") from e
    
    async def process_and_store(
        self,
        text_content: str,
        source_id: str,
        collection: Any,  # MongoDB collection (AppDB Collection or Motor collection)
        max_tokens: Optional[int] = None,
        tokenizer_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process text and store chunks with embeddings in MongoDB.
        
        This is the main method that:
        1. Chunks the text semantically
        2. Generates embeddings for each chunk
        3. Stores documents in MongoDB with proper structure
        
        Args:
            text_content: The text to process
            source_id: Unique identifier for the source document
            collection: MongoDB collection (AppDB Collection or Motor collection)
            max_tokens: Max tokens per chunk (default: uses default_max_tokens)
            tokenizer_model: Tokenizer model for counting (default: uses default_tokenizer_model)
            embedding_model: Embedding model (default: uses LLMService default)
            metadata: Additional metadata to store with each chunk
        
        Returns:
            Dict with processing results:
            {
                "chunks_created": int,
                "documents_inserted": int,
                "source_id": str
            }
        
        Example:
            result = await service.process_and_store(
                text_content="Long document...",
                source_id="doc_101",
                collection=db.knowledge_base,
                max_tokens=1000
            )
            print(f"Created {result['chunks_created']} chunks")
        """
        logger.info(f"Processing source: {source_id}")
        
        # Step 1: Chunk the text
        chunks = await self.chunk_text(
            text_content,
            max_tokens=max_tokens,
            tokenizer_model=tokenizer_model
        )
        
        if not chunks:
            logger.warning(f"No chunks generated for source: {source_id}")
            return {
                "chunks_created": 0,
                "documents_inserted": 0,
                "source_id": source_id
            }
        
        # Step 2: Generate embeddings (batch for efficiency)
        try:
            vectors = await self.embed_chunks(chunks, model=embedding_model)
        except Exception as e:
            logger.error(f"Failed to generate embeddings for {source_id}: {e}")
            raise EmbeddingServiceError(f"Embedding generation failed: {str(e)}") from e
        
        if len(vectors) != len(chunks):
            raise EmbeddingServiceError(
                f"Mismatch: {len(chunks)} chunks but {len(vectors)} embeddings"
            )
        
        # Step 3: Prepare documents for insertion
        documents_to_insert = []
        for i, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
            doc = {
                "source_id": source_id,
                "chunk_index": i,
                "text": chunk_text,
                "embedding": vector,
                "metadata": {
                    "model": embedding_model or self.llm_service.settings.default_embedding_model,
                    "token_count": len(chunk_text),  # Approximation
                    "created_at": datetime.utcnow()
                }
            }
            
            # Add custom metadata if provided
            if metadata:
                doc["metadata"].update(metadata)
            
            documents_to_insert.append(doc)
        
        # Step 4: Store in MongoDB
        try:
            # Handle both AppDB Collection and Motor collection
            if hasattr(collection, 'insert_many'):
                # AppDB Collection wrapper
                result = await collection.insert_many(documents_to_insert)
                inserted_count = len(result.inserted_ids)
            else:
                # Direct Motor collection
                result = await collection.insert_many(documents_to_insert)
                inserted_count = len(result.inserted_ids)
            
            logger.info(
                f"Successfully inserted {inserted_count} documents for source: {source_id}"
            )
            
            return {
                "chunks_created": len(chunks),
                "documents_inserted": inserted_count,
                "source_id": source_id
            }
            
        except Exception as e:
            logger.error(f"Failed to store documents for {source_id}: {e}", exc_info=True)
            raise EmbeddingServiceError(f"Storage failed: {str(e)}") from e
    
    async def process_text(
        self,
        text_content: str,
        max_tokens: Optional[int] = None,
        tokenizer_model: Optional[str] = None,
        embedding_model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process text and return chunks with embeddings (without storing).
        
        Useful when you want to process text but handle storage yourself.
        
        Args:
            text_content: The text to process
            max_tokens: Max tokens per chunk (default: uses default_max_tokens)
            tokenizer_model: Tokenizer model for counting (default: uses default_tokenizer_model)
            embedding_model: Embedding model (default: uses LLMService default)
        
        Returns:
            List of dicts, each containing:
            {
                "chunk_index": int,
                "text": str,
                "embedding": List[float],
                "metadata": Dict[str, Any]
            }
        
        Example:
            results = await service.process_text("Long document...")
            for result in results:
                print(f"Chunk {result['chunk_index']}: {result['text'][:50]}...")
        """
        # Chunk the text
        chunks = await self.chunk_text(
            text_content,
            max_tokens=max_tokens,
            tokenizer_model=tokenizer_model
        )
        
        if not chunks:
            return []
        
        # Generate embeddings
        vectors = await self.embed_chunks(chunks, model=embedding_model)
        
        if len(vectors) != len(chunks):
            raise EmbeddingServiceError(
                f"Mismatch: {len(chunks)} chunks but {len(vectors)} embeddings"
            )
        
        # Prepare results
        results = []
        for i, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
            results.append({
                "chunk_index": i,
                "text": chunk_text,
                "embedding": vector,
                "metadata": {
                    "model": embedding_model or self.llm_service.settings.default_embedding_model,
                    "token_count": len(chunk_text),
                    "created_at": datetime.utcnow()
                }
            })
        
        return results


# Dependency injection helper
def get_embedding_service(
    llm_service: LLMService,
    config: Optional[Dict[str, Any]] = None
) -> EmbeddingService:
    """
    Create EmbeddingService instance with optional configuration.
    
        Args:
            llm_service: LLMService instance
            config: Optional configuration dict (from manifest.json embedding_config)
                    Supports: max_tokens_per_chunk, tokenizer_model (optional, defaults to "gpt-3.5-turbo")
    
    Returns:
        EmbeddingService instance
    
    Example:
        from mdb_runtime.llm import get_embedding_service, get_llm_service
        
        llm_service = get_llm_service()
        embedding_service = get_embedding_service(
            llm_service,
            config={"max_tokens_per_chunk": 1000}
        )
    """
    # Platform-level defaults (users don't need to think about these)
    default_max_tokens = 1000
    default_tokenizer_model = "gpt-3.5-turbo"  # Model name for tiktoken (uses cl100k_base encoding internally)
    
    # Override from config if provided (but not required)
    if config:
        default_max_tokens = config.get("max_tokens_per_chunk", default_max_tokens)
        # tokenizer_model is optional - only override if explicitly provided
        if "tokenizer_model" in config:
            default_tokenizer_model = config["tokenizer_model"]
    
    return EmbeddingService(
        llm_service=llm_service,
        default_max_tokens=default_max_tokens,
        default_tokenizer_model=default_tokenizer_model
    )

