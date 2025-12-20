#!/usr/bin/env python3
"""
Parallax Engine - Tech News Analysis Tool

The Parallax Engine orchestrates concurrent analysis of tech news from two focused angles:
1. Relevance: Why this story matters given your watchlist - personalized context and urgency
2. Technical: Concise engineering assessment - performance, complexity, readiness, use cases
"""
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# Setup logger first
logger = logging.getLogger("Parallax")

# Try to import httpx for async HTTP requests
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available, falling back to requests")

# Try to import requests as fallback
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# LangChain imports
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_litellm import ChatLiteLLM
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available")

from schemas import ParallaxReport
from schema_generator import create_dynamic_model, get_default_lens_configs

# The Filter Config - Watchlist for relevant tech keywords
WATCHLIST = ["MongoDB", "vector", "voyageAI"]

# Hacker News API endpoints
HN_TOP_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{}.json"


class ParallaxEngine:
    """
    The Parallax Engine analyzes tech news from Relevance and Technical perspectives.
    
    Architecture:
    1. Fetch and filter Hacker News stories by watchlist keywords
    2. Fan-out to three concurrent agents (Marketing, Sales, Product)
    3. Aggregate results into ParallaxReport
    4. Store in MongoDB for dashboard visualization
    """
    
    def __init__(self, llm_service, db, watchlist: Optional[List[str]] = None):
        """
        Initialize the Parallax Engine.
        
        Args:
            llm_service: MDB_RUNTIME LLMService instance
            db: Scoped MongoDB database instance
            watchlist: Optional list of keywords to watch (defaults to WATCHLIST)
        """
        self.llm_service = llm_service
        self.db = db
        self.llm = None
        self.watchlist = watchlist or WATCHLIST
        self.scan_limit = 50  # Default scan limit
        self.lens_configs = {}  # Cache for lens configurations
        self.lens_models = {}  # Cache for dynamically generated models
        
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError("LangChain dependencies required for Parallax Engine")
        
        # Initialize LangChain LLM adapter
        try:
            model_name = llm_service.settings.default_chat_model
            temperature = 0.0  # Strict adherence to facts for Parallax
            
            # Handle Azure OpenAI configuration
            if llm_service.settings.azure_openai_api_key:
                import os
                os.environ["AZURE_API_KEY"] = llm_service.settings.azure_openai_api_key
                if llm_service.settings.azure_openai_endpoint:
                    os.environ["AZURE_API_BASE"] = llm_service.settings.azure_openai_endpoint
                api_version = getattr(llm_service.settings, 'azure_openai_api_version', None)
                if api_version:
                    os.environ["AZURE_API_VERSION"] = api_version
                
                deployment_name = llm_service.settings.azure_openai_deployment_name or model_name.replace("azure/", "")
                model_name = f"azure/{deployment_name}"
            
            # Handle OpenAI configuration
            elif llm_service.settings.openai_api_key:
                import os
                os.environ["OPENAI_API_KEY"] = llm_service.settings.openai_api_key
            
            self.llm = ChatLiteLLM(
                model=model_name,
                temperature=temperature
            )
            logger.info(f"Parallax Engine initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LangChain LLM: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Parallax Engine: {e}") from e
    
    async def _fetch_story(self, story_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single HN story"""
        if HTTPX_AVAILABLE:
            async with httpx.AsyncClient(timeout=10.0) as client:
                try:
                    response = await client.get(HN_ITEM_URL.format(story_id))
                    if response.status_code == 200:
                        return response.json()
                except Exception as e:
                    logger.debug(f"Failed to fetch story {story_id}: {e}")
        elif REQUESTS_AVAILABLE:
            try:
                response = requests.get(HN_ITEM_URL.format(story_id), timeout=10)
                if response.status_code == 200:
                    return response.json()
            except Exception as e:
                logger.debug(f"Failed to fetch story {story_id}: {e}")
        
        return None
    
    async def _fetch_and_filter(self, limit: int = 50) -> List[Dict]:
        """
        Ingest the feed and apply the 'Buzzword' filter.
        
        Args:
            limit: Maximum number of top stories to check (increased to find more matches)
            
        Returns:
            List of relevant stories that match watchlist keywords
        """
        try:
            # Fetch top story IDs
            if HTTPX_AVAILABLE:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    ids_resp = await client.get(HN_TOP_URL)
                    top_ids = ids_resp.json()[:limit]
            elif REQUESTS_AVAILABLE:
                ids_resp = requests.get(HN_TOP_URL, timeout=10)
                top_ids = ids_resp.json()[:limit]
            else:
                logger.error("No HTTP client available (httpx or requests)")
                return []
            
            # Fetch story details in parallel
            tasks = [self._fetch_story(sid) for sid in top_ids]
            stories = await asyncio.gather(*tasks)
            
            # Filter by watchlist keywords
            relevant_stories = []
            for story in stories:
                if not story or story.get('type') != 'story':
                    continue
                
                title = story.get('title', '').lower()
                text = story.get('text', '').lower() if story.get('text') else ''
                combined_text = f"{title} {text}"
                
                # Check if any watchlist keyword appears in title or text
                # Prioritize MongoDB matches
                matched_keywords = [kw for kw in self.watchlist if kw.lower() in combined_text]
                if matched_keywords:
                    # Prioritize MongoDB stories
                    is_mongodb = any('mongodb' in kw.lower() for kw in matched_keywords)
                    story['matched_keywords'] = matched_keywords
                    story['is_mongodb'] = is_mongodb
                    relevant_stories.append(story)
                    logger.info(f"PARALLAX DETECTED: {story.get('title', 'Unknown')} [Keywords: {', '.join(matched_keywords[:3])}]")
            
            # Sort: MongoDB stories first, then by score
            relevant_stories.sort(key=lambda x: (not x.get('is_mongodb', False), -x.get('score', 0)))
            
            # Limit to top 10 most relevant
            return relevant_stories[:10]
        
        except Exception as e:
            logger.error(f"Error fetching and filtering stories: {e}", exc_info=True)
            return []
    
    async def _load_lens_config(self, lens_name: str) -> Optional[Dict[str, Any]]:
        """Load lens configuration from database"""
        try:
            config = await self.db.lens_configs.find_one({"lens_name": lens_name})
            if config:
                # Remove MongoDB _id for easier handling
                config.pop("_id", None)
                self.lens_configs[lens_name] = config
                return config
        except Exception as e:
            logger.debug(f"Could not load lens config for {lens_name}: {e}")
        return None
    
    async def _get_lens_model(self, lens_name: str) -> Optional[Any]:
        """Get or create the dynamic Pydantic model for a lens"""
        # Check cache first
        if lens_name in self.lens_models:
            return self.lens_models[lens_name]
        
        # Load config
        config = await self._load_lens_config(lens_name)
        if not config:
            logger.error(f"No configuration found for lens: {lens_name}")
            return None
        
        # Generate model from schema fields
        try:
            model = create_dynamic_model(
                f"{lens_name}View",
                config.get("schema_fields", [])
            )
            self.lens_models[lens_name] = model
            return model
        except Exception as e:
            logger.error(f"Failed to create model for {lens_name}: {e}", exc_info=True)
            return None
    
    async def _generate_viewpoint(
        self, 
        title: str, 
        url: str,
        lens_name: str,
        watchlist: Optional[List[str]] = None
    ) -> Optional[Any]:
        """
        The Agent Worker.
        Takes a lens name, loads its config, and returns a filled-out form.
        
        Args:
            title: The news headline
            url: The story URL (for context)
            lens_name: Lens name (Marketing, Sales, Technical, Relevance)
            watchlist: Optional list of watchlist keywords (for Relevance lens)
            
        Returns:
            Parsed schema instance or None on error
        """
        if not self.llm:
            logger.error("LLM not initialized")
            return None
        
        # Load lens configuration
        config = await self._load_lens_config(lens_name)
        if not config:
            logger.error(f"No configuration found for lens: {lens_name}")
            return None
        
        # Get or create the dynamic model
        schema = await self._get_lens_model(lens_name)
        if not schema:
            logger.error(f"Failed to get model for lens: {lens_name}")
            return None
        
        try:
            parser = PydanticOutputParser(pydantic_object=schema)
            
            # Use configurable prompt template
            prompt_template = config.get("prompt_template", 
                "You are the {role} Specialist.\n\n{format_instructions}")
            
            # Format the prompt template with watchlist if provided (for Relevance lens)
            watchlist_str = ", ".join(watchlist) if watchlist else ""
            system_prompt = prompt_template.format(
                role=lens_name,
                watchlist=watchlist_str,
                format_instructions="{format_instructions}"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "NEWS TITLE: {title}\nURL: {url}")
            ])
            
            # Execute chain
            chain = prompt | self.llm | parser
            
            # Use async invoke if available, otherwise sync
            if hasattr(chain, 'ainvoke'):
                result = await chain.ainvoke({
                    "title": title,
                    "url": url,
                    "format_instructions": parser.get_format_instructions()
                })
            else:
                # Fallback to sync execution in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, chain.invoke, {
                    "title": title,
                    "url": url,
                    "format_instructions": parser.get_format_instructions()
                })
            
            return result
        
        except Exception as e:
            logger.error(f"{lens_name} Viewpoint Failed: {e}", exc_info=True)
            return None  # Fail gracefully
    
    async def _load_watchlist_config(self):
        """Load watchlist configuration from database"""
        try:
            config = await self.db.watchlist_config.find_one({"config_type": "watchlist"})
            if config:
                if config.get("keywords"):
                    self.watchlist = config["keywords"]
                    logger.info(f"Loaded watchlist from config: {self.watchlist}")
                if config.get("scan_limit"):
                    self.scan_limit = config["scan_limit"]
                    logger.info(f"Loaded scan limit from config: {self.scan_limit}")
        except Exception as e:
            logger.debug(f"Could not load watchlist config: {e}, using default")
    
    async def update_watchlist(self, keywords: List[str], scan_limit: Optional[int] = None) -> bool:
        """Update the watchlist configuration"""
        try:
            update_data = {
                "keywords": keywords,
                "updated_at": datetime.utcnow()
            }
            if scan_limit is not None:
                update_data["scan_limit"] = scan_limit
                self.scan_limit = scan_limit
            
            await self.db.watchlist_config.update_one(
                {"config_type": "watchlist"},
                {"$set": update_data},
                upsert=True
            )
            self.watchlist = keywords
            logger.info(f"Updated watchlist: {keywords}, scan_limit: {self.scan_limit}")
            return True
        except Exception as e:
            logger.error(f"Failed to update watchlist: {e}")
            return False
    
    async def get_scan_limit(self) -> int:
        """Get current scan limit"""
        await self._load_watchlist_config()
        return self.scan_limit
    
    async def get_watchlist(self) -> List[str]:
        """Get current watchlist"""
        await self._load_watchlist_config()
        return self.watchlist
    
    async def analyze_feed(self) -> List[ParallaxReport]:
        """
        The Main Loop:
        1. Get Stories (filtered by watchlist)
        2. Fan-Out to 3 Agents (Marketing, Sales, Product)
        3. Aggregate & Store
        
        Returns:
            List of ParallaxReport instances
        """
        # Load watchlist and scan limit from DB config if available
        await self._load_watchlist_config()
        
        stories = await self._fetch_and_filter(limit=self.scan_limit)
        reports = []
        
        for story in stories:
            story_id = story.get('id')
            if not story_id:
                continue
            
            # Check DB cache to save money
            cached = await self.db.parallax_reports.find_one({"story_id": story_id})
            if cached:
                logger.debug(f"Skipping cached story: {story.get('title', 'Unknown')}")
                continue
            
            title = story.get('title', 'Unknown')
            url = story.get('url', f"https://news.ycombinator.com/item?id={story_id}")
            
            logger.info(f"Orchestrating Parallax View for: {title}")
            
            # --- THE FAN OUT ---
            # Launch 2 focused agents simultaneously
            task_relevance = self._generate_viewpoint(title, url, "Relevance", watchlist=self.watchlist)
            task_technical = self._generate_viewpoint(title, url, "Technical")
            
            # Wait for both views
            r_view, t_view = await asyncio.gather(task_relevance, task_technical)
            
            if r_view and t_view:
                # Convert Pydantic models to dicts for storage
                relevance_dict = r_view.dict() if hasattr(r_view, 'dict') else r_view
                technical_dict = t_view.dict() if hasattr(t_view, 'dict') else t_view
                
                # Get matched keywords from story
                matched_keywords = story.get('matched_keywords', [])
                
                report = ParallaxReport(
                    story_id=story_id,
                    original_title=title,
                    url=url,
                    marketing={},  # Empty for backward compatibility
                    sales={},  # Empty for backward compatibility
                    product=technical_dict,  # Keep 'product' key for backward compatibility
                    relevance=relevance_dict,
                    timestamp=datetime.utcnow().isoformat(),
                    matched_keywords=matched_keywords
                )
                
                # Store in MongoDB
                await self.db.parallax_reports.insert_one(report.dict())
                reports.append(report)
                logger.info(f"Parallax report stored for: {title}")
            else:
                logger.warning(f"Failed to generate complete Parallax view for: {title}")
        
        return reports

