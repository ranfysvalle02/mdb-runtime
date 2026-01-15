#!/usr/bin/env python3
"""
Snippet Extractor Module

Extracts relevant snippets from call transcripts using RAG (Retrieval-Augmented Generation).
Demonstrates mdb-engine's vector search + LLM capabilities for intelligent snippet extraction.
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Parallax.RAG")


class SnippetExtractor:
    """
    Extract relevant snippets using RAG (vector search + LLM).
    
    This module demonstrates:
    1. Building semantic search queries from lens analysis
    2. Vector search to find relevant transcript chunks
    3. LLM formatting to explain why snippets are relevant
    4. Structured JSON output for consistent results
    """
    
    def __init__(
        self,
        openai_client,
        embedding_service=None,
        db=None,
        app_slug: str = "parallax",
        deployment_name: str = "gpt-4o",
    ):
        """
        Initialize the Snippet Extractor.
        
        Args:
            openai_client: AzureOpenAI or OpenAI client instance
            embedding_service: Optional EmbeddingService for vector search
            db: Scoped MongoDB database instance
            app_slug: App slug (for vector index naming)
            deployment_name: Model deployment name (for Azure) or model name (for OpenAI)
        """
        self.openai_client = openai_client
        self.embedding_service = embedding_service
        self.db = db
        self.app_slug = app_slug
        self.deployment_name = deployment_name
        self.vector_index_name = f"{app_slug}_transcript_vector_idx"
    
    async def extract_snippets(
        self,
        transcript: str,
        call_id: str,
        lens_name: str,
        lens_analysis: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        Extract relevant snippets using RAG pipeline.
        
        Pipeline:
        1. Build search query from lens analysis
        2. Vector search to find relevant chunks
        3. LLM to format and explain snippets
        
        Args:
            transcript: The full call transcript
            call_id: Call identifier for vector search
            lens_name: Lens name (SALES, MARKETING, PRODUCT)
            lens_analysis: The analysis results from the lens
            
        Returns:
            List of dicts with 'snippet' and 'reason' keys
        """
        if not self.openai_client or not self.embedding_service:
            logger.warning(
                f"Missing dependencies for snippet extraction: "
                f"openai_client={bool(self.openai_client)}, "
                f"embedding_service={bool(self.embedding_service)}"
            )
            return []
        
        try:
            # Build a query from the lens analysis to find relevant parts
            lens_focus = {
                "SALES": "sales opportunities, deal value, objections, closing signals, buying intent, pricing discussions",
                "MARKETING": "customer sentiment, messaging effectiveness, brand perception, engagement, competitive mentions",
                "PRODUCT": "feature requests, pain points, use cases, product fit, technical requirements",
            }.get(lens_name, "key insights")
            
            # Create a search query from key analysis points
            key_points = []
            if lens_analysis.get("key_insight"):
                key_points.append(lens_analysis["key_insight"])
            if lens_analysis.get("pain_points"):
                key_points.extend(lens_analysis["pain_points"][:3])
            if lens_analysis.get("products_discussed"):
                key_points.extend(lens_analysis["products_discussed"][:2])
            if lens_analysis.get("deal_value"):
                key_points.append(f"deal value {lens_analysis['deal_value']}")
            
            search_query = f"{lens_name} perspective: {lens_focus}. Key points: {', '.join(key_points[:5])}"
            
            # Use vector search to find relevant chunks from the transcript
            if self.embedding_service and self.db:
                try:
                    # Generate query embedding
                    query_vectors = await self.embedding_service.embed_chunks([search_query])
                    if not query_vectors or len(query_vectors) == 0:
                        logger.warning("Failed to generate query embedding for snippet search")
                        return []
                    
                    query_vector = query_vectors[0]
                    
                    # Vector search in call_transcripts collection
                    pipeline = [
                        {
                            "$vectorSearch": {
                                "index": self.vector_index_name,
                                "path": "embedding",
                                "queryVector": query_vector,
                                "numCandidates": 20,
                                "limit": 5,
                                "filter": {"call_id": call_id},  # Only search within this call
                            }
                        },
                        {
                            "$project": {
                                "_id": 0,
                                "chunk_text": 1,
                                "transcript": {"$ifNull": ["$chunk_text", "$transcript"]},
                                "score": {"$meta": "vectorSearchScore"},
                            }
                        },
                    ]
                    
                    # Try vector search, fallback to full transcript if it fails
                    try:
                        results = await self.db.call_transcripts.aggregate(pipeline).to_list(length=5)
                        if results and len(results) > 0:
                            # Use vector search results - extract relevant chunks
                            relevant_chunks = []
                            for r in results:
                                chunk_text = r.get("chunk_text") or r.get("transcript", "")
                                if chunk_text:
                                    relevant_chunks.append(chunk_text[:800])  # Limit chunk length
                            
                            if relevant_chunks:
                                logger.debug(f"Found {len(relevant_chunks)} relevant chunks via vector search for {lens_name}")
                                # Use LLM to format and explain snippets from vector search results
                                return await self._format_snippets_with_llm(
                                    relevant_chunks, transcript, lens_name, lens_analysis, lens_focus
                                )
                    except Exception as vs_error:
                        logger.debug(f"Vector search failed, using fallback: {vs_error}")
                        # Fallback: use full transcript
                        pass
                        
                except Exception as e:
                    logger.debug(f"Vector search error: {e}, using fallback")
                    pass
            
            # Fallback: Use LLM directly on full transcript (original method)
            return await self._format_snippets_with_llm(
                [transcript], transcript, lens_name, lens_analysis, lens_focus
            )
        
        except Exception as e:
            logger.warning(f"Failed to extract snippets for {lens_name}: {e}", exc_info=True)
            return []
    
    async def _format_snippets_with_llm(
        self,
        relevant_chunks: List[str],
        full_transcript: str,
        lens_name: str,
        lens_analysis: Dict[str, Any],
        lens_focus: str,
    ) -> List[Dict[str, str]]:
        """
        Use LLM to format and explain snippets from relevant chunks.
        
        Args:
            relevant_chunks: List of relevant transcript chunks (from vector search)
            full_transcript: Full transcript for context
            lens_name: Lens name (SALES, MARKETING, PRODUCT)
            lens_analysis: Analysis results from the lens
            lens_focus: Description of what the lens focuses on
            
        Returns:
            List of dicts with 'snippet' and 'reason' keys
        """
        try:
            # Combine relevant chunks for context
            chunks_text = "\n\n---\n\n".join(relevant_chunks[:3])  # Use top 3 chunks
            
            analysis_summary = json.dumps(lens_analysis, indent=2)[:800]
            
            system_prompt = f"""You are a {lens_name} analyst extracting the most relevant snippets from call transcript chunks.

Your task:
1. From the provided transcript chunks, identify 3-5 key snippets MOST relevant to {lens_focus}
2. Each snippet should be 1-3 sentences, preferably exact quotes
3. Explain WHY each snippet is relevant to {lens_name} analysis
4. Focus on the most impactful and actionable insights

Return JSON:
{{
  "snippets": [
    {{
      "snippet": "exact quote from transcript (1-3 sentences)",
      "reason": "why this snippet is relevant to {lens_name} analysis (1 sentence)"
    }}
  ]
}}

Be critical and selective - only include snippets that truly matter."""

            user_prompt = f"""RELEVANT TRANSCRIPT CHUNKS (found via vector search):
{chunks_text[:4000]}

FULL TRANSCRIPT (for context):
{full_transcript[:6000]}

ANALYSIS SUMMARY:
{analysis_summary}

Extract the most relevant snippets for {lens_name} analysis. Return JSON with snippets array."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.deployment_name,
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content
            result_data = json.loads(content)
            
            snippets = result_data.get("snippets", [])
            if not isinstance(snippets, list):
                snippets = []
            
            # Validate and limit snippets
            validated_snippets = []
            for snippet in snippets[:5]:
                if isinstance(snippet, dict) and "snippet" in snippet and "reason" in snippet:
                    validated_snippets.append({
                        "snippet": str(snippet["snippet"])[:500],
                        "reason": str(snippet["reason"])[:300],
                    })
            
            logger.debug(f"Extracted {len(validated_snippets)} snippets for {lens_name} using RAG")
            return validated_snippets
        
        except Exception as e:
            logger.warning(f"Failed to format snippets with LLM: {e}", exc_info=True)
            return []
