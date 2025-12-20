#!/usr/bin/env python3
"""
Parallax - Tech News Intelligence Tool

The Parallax Dashboard visualizes tech news from two focused angles:
1. Relevance: Why this story matters given your watchlist - personalized context and urgency
2. Technical: Concise engineering assessment - performance, complexity, readiness, use cases
"""
import os
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from mdb_runtime import RuntimeEngine
from parallax import ParallaxEngine, WATCHLIST
from schema_generator import get_default_lens_configs

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Parallax")

# Initialize FastAPI app
app = FastAPI(
    title="Parallax - Tech News Intelligence",
    description="Focused intelligence tool analyzing tech news from Relevance and Technical perspectives",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates directory
templates = Jinja2Templates(directory="/app/templates")

# Global engine instances
engine: Optional[RuntimeEngine] = None
parallax: Optional[ParallaxEngine] = None
db = None


# Global exception handler to ensure all errors return valid JSON
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions and return valid JSON"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "detail": f"Internal server error: {str(exc)}"
        }
    )


@app.on_event("startup")
async def startup_event():
    """Initialize the runtime engine and Parallax on startup"""
    global engine, parallax, db
    
    logger.info("Starting Parallax...")
    
    # Get MongoDB connection from environment
    mongo_uri = os.getenv(
        "MONGO_URI", 
        "mongodb://admin:password@mongodb:27017/?authSource=admin"
    )
    db_name = os.getenv("MONGO_DB_NAME", "parallax_db")
    
    # Initialize the runtime engine
    engine = RuntimeEngine(
        mongo_uri=mongo_uri,
        db_name=db_name
    )
    
    # Connect to MongoDB
    await engine.initialize()
    logger.info("Engine initialized successfully")
    
    # Load and register the app manifest
    manifest_path = Path("/app/manifest.json")
    if not manifest_path.exists():
        manifest_path = Path(__file__).parent / "manifest.json"
    
    if manifest_path.exists():
        manifest = await engine.load_manifest(manifest_path)
        success = await engine.register_app(manifest, create_indexes=True)
        if success:
            logger.info(f"App '{manifest['slug']}' registered successfully")
        else:
            logger.warning("Failed to register app")
    else:
        logger.warning(f"Manifest not found at {manifest_path}")
    
    # Get scoped database
    db = engine.get_scoped_db("parallax")
    
    # Initialize default watchlist config if it doesn't exist
    try:
        existing_config = await db.watchlist_config.find_one({"config_type": "watchlist"})
        if not existing_config:
            await db.watchlist_config.insert_one({
                "config_type": "watchlist",
                "keywords": WATCHLIST,
                "scan_limit": 50,  # Default: check top 50 stories
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            logger.info(f"Initialized default watchlist: {WATCHLIST}, scan_limit: 50")
    except Exception as e:
        logger.warning(f"Could not initialize watchlist config: {e}")
    
    # Initialize default lens configurations if they don't exist
    try:
        default_configs = get_default_lens_configs()
        for lens_name, config in default_configs.items():
            existing_lens = await db.lens_configs.find_one({"lens_name": lens_name})
            if not existing_lens:
                config["created_at"] = datetime.utcnow()
                config["updated_at"] = datetime.utcnow()
                await db.lens_configs.insert_one(config)
                logger.info(f"Initialized default lens config: {lens_name}")
    except Exception as e:
        logger.warning(f"Could not initialize lens configs: {e}")
    
    # Initialize Parallax Engine
    global parallax
    try:
        llm_service = engine.get_llm_service("parallax")
        if not llm_service:
            logger.error("LLM service not available! Check API keys (OPENAI_API_KEY or AZURE_OPENAI_API_KEY) and manifest configuration.")
            raise RuntimeError("LLM service not available - required for Parallax")
        
        # Load watchlist from DB or use default
        try:
            config = await db.watchlist_config.find_one({"config_type": "watchlist"})
            watchlist = config.get("keywords", WATCHLIST) if config else WATCHLIST
        except Exception:
            watchlist = WATCHLIST
        
        parallax = ParallaxEngine(llm_service, db, watchlist=watchlist)
        logger.info("Parallax Engine initialized successfully")
        
        # Automatically trigger initial scan in background
        logger.info("Triggering initial Parallax scan...")
        asyncio.create_task(parallax.analyze_feed())
    except Exception as e:
        logger.error(f"Failed to initialize Parallax Engine: {e}", exc_info=True)
        raise RuntimeError(f"Parallax Engine initialization failed: {str(e)}") from e
    
        logger.info("Parallax ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global engine
    if engine:
        await engine.shutdown()
        logger.info("Cleaned up and shut down")


def get_db():
    """Get the scoped database"""
    global engine, db
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    if db is None:
        return engine.get_scoped_db("parallax")
    return db


@app.post("/api/refresh", response_class=JSONResponse)
async def trigger_refresh():
    """Trigger the multi-agent analysis manually"""
    global parallax
    
    if not parallax:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "Parallax Engine not initialized"
            }
        )
    
    try:
        logger.info("🔄 Triggering Parallax analysis...")
        reports = await parallax.analyze_feed()
        
        # Update last scan timestamp
        db = get_db()
        try:
            await db.watchlist_config.update_one(
                {"config_type": "watchlist"},
                {"$set": {"last_scan_timestamp": datetime.utcnow()}},
                upsert=True
            )
        except Exception as e:
            logger.warning(f"Could not update last scan timestamp: {e}")
        
        if len(reports) == 0:
            logger.info("No new stories found matching watchlist criteria")
            return {
                "success": True,
                "status": "no_new",
                "new_reports": 0,
                "message": "Found nothing new. All matching stories have already been analyzed."
            }
        else:
            logger.info(f"Generated {len(reports)} new Parallax reports")
            return {
                "success": True,
                "status": "success",
                "new_reports": len(reports),
                "message": f"Analyzed {len(reports)} new stories"
            }
    except Exception as e:
        logger.error(f"Error in Parallax analysis: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "detail": f"Failed to analyze feed: {str(e)}"
            }
        )


@app.get("/api/reports", response_class=JSONResponse)
async def get_reports(limit: int = 10, keyword: str = None, sort_by: str = "date_desc"):
    """
    Get Parallax reports
    
    Args:
        limit: Maximum number of reports to return
        keyword: Optional keyword to filter by (must be in matched_keywords)
        sort_by: Sort order - "date_desc", "date_asc", "relevance"
    """
    db = get_db()
    
    try:
        query = {}
        
        # If filtering by keyword
        if keyword:
            query["matched_keywords"] = {"$in": [keyword]}
        
        # Get last scan timestamp to determine freshness
        scan_config = await db.watchlist_config.find_one({"config_type": "watchlist"})
        last_scan = None
        if scan_config and scan_config.get("last_scan_timestamp"):
            from datetime import datetime
            last_scan = scan_config["last_scan_timestamp"]
            # Convert to datetime if it's a string
            if isinstance(last_scan, str):
                last_scan = datetime.fromisoformat(last_scan.replace('Z', '+00:00'))
        
        # Determine sort order
        sort_order = [("timestamp", -1)]  # Default: newest first
        if sort_by == "date_asc":
            sort_order = [("timestamp", 1)]  # Oldest first
        elif sort_by == "relevance":
            sort_order = [("timestamp", -1)]
        
        reports = await db.parallax_reports.find(query).sort(sort_order).limit(limit * 2).to_list(length=limit * 2)
        
        # Mark reports as fresh
        if last_scan:
            last_scan_iso = last_scan.isoformat()
            for r in reports:
                r["is_fresh"] = r.get("timestamp") and r["timestamp"] > last_scan_iso
        
        # Apply relevance sorting if needed
        if sort_by == "relevance":
            reports.sort(key=lambda x: (len(x.get("matched_keywords", [])), x.get("timestamp", "")), reverse=True)
        
        # Limit to requested amount
        reports = reports[:limit]
        
        # Convert to dict format for JSON serialization
        reports_data = []
        for r in reports:
            # Convert ObjectId to string
            r_dict = dict(r)
            r_dict['_id'] = str(r_dict.get('_id', ''))
            reports_data.append(r_dict)
        
        return {
            "success": True,
            "reports": reports_data,
            "count": len(reports_data),
            "sort_by": sort_by
        }
    except Exception as e:
        logger.error(f"Error fetching reports: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@app.get("/api/watchlist", response_class=JSONResponse)
async def get_watchlist():
    """Get current watchlist configuration"""
    global parallax
    
    if not parallax:
        return JSONResponse(
            status_code=503,
            content={"success": False, "error": "Parallax Engine not initialized"}
        )
    
    try:
        keywords = await parallax.get_watchlist()
        scan_limit = await parallax.get_scan_limit()
        return {
            "success": True,
            "watchlist": keywords,
            "scan_limit": scan_limit
        }
    except Exception as e:
        logger.error(f"Error fetching watchlist: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.post("/api/watchlist", response_class=JSONResponse)
async def update_watchlist(request: Request):
    """Update watchlist configuration"""
    global parallax
    
    if not parallax:
        return JSONResponse(
            status_code=503,
            content={"success": False, "error": "Parallax Engine not initialized"}
        )
    
    try:
        data = await request.json()
        keywords = data.get("watchlist", [])
        scan_limit = data.get("scan_limit")
        
        if not isinstance(keywords, list):
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "watchlist must be a list"}
            )
        
        if scan_limit is not None:
            scan_limit = int(scan_limit)
            if scan_limit < 1 or scan_limit > 500:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "scan_limit must be between 1 and 500"}
                )
        
        success = await parallax.update_watchlist(keywords, scan_limit=scan_limit)
        if success:
            return {
                "success": True,
                "watchlist": keywords,
                "scan_limit": parallax.scan_limit,
                "message": "Watchlist updated successfully"
            }
        else:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Failed to update watchlist"}
            )
    except Exception as e:
        logger.error(f"Error updating watchlist: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/api/lenses", response_class=JSONResponse)
async def get_lenses():
    """Get all lens configurations"""
    db = get_db()
    
    try:
        lenses = await db.lens_configs.find({}).to_list(length=10)
        # Convert ObjectId to string for JSON serialization
        lenses_data = []
        for lens in lenses:
            lens_dict = dict(lens)
            lens_dict['_id'] = str(lens_dict.get('_id', ''))
            lenses_data.append(lens_dict)
        
        return {
            "success": True,
            "lenses": lenses_data
        }
    except Exception as e:
        logger.error(f"Error fetching lenses: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/api/lenses/{lens_name}", response_class=JSONResponse)
async def get_lens(lens_name: str):
    """Get a specific lens configuration"""
    db = get_db()
    
    try:
        lens = await db.lens_configs.find_one({"lens_name": lens_name})
        if not lens:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": f"Lens '{lens_name}' not found"}
            )
        
        lens_dict = dict(lens)
        lens_dict['_id'] = str(lens_dict.get('_id', ''))
        
        return {
            "success": True,
            "lens": lens_dict
        }
    except Exception as e:
        logger.error(f"Error fetching lens {lens_name}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.post("/api/lenses/{lens_name}", response_class=JSONResponse)
async def update_lens(lens_name: str, request: Request):
    """Update a lens configuration"""
    global parallax
    db = get_db()
    
    if not parallax:
        return JSONResponse(
            status_code=503,
            content={"success": False, "error": "Parallax Engine not initialized"}
        )
    
    try:
        data = await request.json()
        
        # Validate required fields
        if "schema_fields" not in data:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "schema_fields is required"}
            )
        
        # Update the lens config
        update_data = {
            "lens_name": lens_name,
            "prompt_template": data.get("prompt_template", ""),
            "schema_fields": data.get("schema_fields", []),
            "updated_at": datetime.utcnow()
        }
        
        result = await db.lens_configs.update_one(
            {"lens_name": lens_name},
            {"$set": update_data},
            upsert=True
        )
        
        # Clear the cache so new config is loaded
        if hasattr(parallax, 'lens_configs'):
            parallax.lens_configs.pop(lens_name, None)
        if hasattr(parallax, 'lens_models'):
            parallax.lens_models.pop(lens_name, None)
        
        logger.info(f"Updated lens configuration: {lens_name}")
        
        return {
            "success": True,
            "lens": update_data,
            "message": f"Lens '{lens_name}' updated successfully"
        }
    except Exception as e:
        logger.error(f"Error updating lens {lens_name}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, keyword: str = None, sort_by: str = "date_desc"):
    """
    The Parallax Dashboard.
    Renders the 3-Column Split View.
    
    Args:
        keyword: Optional keyword to filter by
        sort_by: Sort order - "date_desc", "date_asc", "relevance"
    """
    db = get_db()
    
    try:
        query = {}
        
        # If filtering by keyword
        if keyword:
            query["matched_keywords"] = {"$in": [keyword]}
        
        # Get last scan timestamp to determine freshness
        scan_config = await db.watchlist_config.find_one({"config_type": "watchlist"})
        last_scan = None
        if scan_config and scan_config.get("last_scan_timestamp"):
            from datetime import datetime
            last_scan = scan_config["last_scan_timestamp"]
            # Convert to datetime if it's a string
            if isinstance(last_scan, str):
                last_scan = datetime.fromisoformat(last_scan.replace('Z', '+00:00'))
        
        # Determine sort order
        sort_order = [("timestamp", -1)]  # Default: newest first
        if sort_by == "date_asc":
            sort_order = [("timestamp", 1)]  # Oldest first
        elif sort_by == "relevance":
            # Sort by relevance (based on matched_keywords count, virality score, etc.)
            sort_order = [("timestamp", -1)]
        
        reports = await db.parallax_reports.find(query).sort(sort_order).limit(50).to_list(length=50)
        
        # Mark reports as fresh
        if last_scan:
            last_scan_iso = last_scan.isoformat()
            for r in reports:
                r["is_fresh"] = r.get("timestamp") and r["timestamp"] > last_scan_iso
        
        # Apply relevance sorting if needed
        if sort_by == "relevance":
            # Sort by number of matched keywords (more = more relevant), then by timestamp
            reports.sort(key=lambda x: (len(x.get("matched_keywords", [])), x.get("timestamp", "")), reverse=True)
        
        # Limit to 10 for display
        reports = reports[:10]
        
        # Get current watchlist
        watchlist = WATCHLIST
        global parallax
        if parallax:
            watchlist = await parallax.get_watchlist()
        
        # If no reports and we have a parallax engine, trigger a scan
        if not reports and not keyword:
            if parallax:
                logger.info("No reports found, triggering initial scan...")
                # Trigger scan in background (don't wait)
                asyncio.create_task(parallax.analyze_feed())
    except Exception as e:
        logger.error(f"Error fetching reports for dashboard: {e}")
        reports = []
        watchlist = WATCHLIST
    
    # Render the dashboard template
    return templates.TemplateResponse(
        "parallax_dashboard.html",
        {
            "request": request,
            "reports": reports,
            "watchlist": watchlist,  # Pass as list, not joined string
            "selected_keyword": keyword,
            "sort_by": sort_by
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
