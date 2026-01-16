#!/usr/bin/env python3
"""
Click Tracker Dashboard Application
====================================

A clean example demonstrating cross-app data access with Casbin authorization.
The Dashboard reads click data from ClickTracker app for analytics.

This example shows:
- How to use engine.create_app() for automatic lifecycle management
- How to access data from other apps via read_scopes
- How to perform cross-app queries with proper authorization
- How to separate MDB-Engine specifics from reusable business logic

Key Concepts:
- engine.create_app() - Creates FastAPI app with automatic lifecycle management
- get_scoped_db() - Provides database access scoped to your app
- Cross-app access - Reading data from other apps via read_scopes in manifest
- db.get_collection() - Access collections from other apps
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from bson import ObjectId
from fastapi import Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# Configure logging early
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

try:
    from mdb_engine import MongoDBEngine
    from mdb_engine.dependencies import get_scoped_db, get_authz_provider
    from mdb_engine.auth import (
        authenticate_app_user,
        create_app_session,
        get_app_user,
        logout_user,
    )
except ImportError as e:
    logger.error(f"Failed to import mdb_engine: {e}", exc_info=True)
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================
# Application constants and configuration
# These are reusable across different database backends

APP_SLUG = "click_tracker_dashboard"

# ============================================================================
# STEP 1: INITIALIZE MONGODB ENGINE
# ============================================================================
# The MongoDBEngine is the core of mdb-engine. It manages:
# - Database connections
# - App registration and configuration
# - Authorization providers (Casbin, OSO Cloud, etc.)
# - Cross-app access control via read_scopes
# - Lifecycle management
#
# This is MDB-Engine specific - you would replace this with your own
# database connection logic if not using mdb-engine.

try:
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB", "mdb_runtime")
    logger.info(f"Initializing MongoDBEngine with URI: {mongo_uri[:50]}... (db: {db_name})")
    
    engine = MongoDBEngine(
        mongo_uri=mongo_uri,
        db_name=db_name,
    )
    logger.info("MongoDBEngine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MongoDBEngine: {e}", exc_info=True)
    sys.exit(1)

# ============================================================================
# STEP 2: CREATE FASTAPI APP WITH MDB-ENGINE
# ============================================================================
# engine.create_app() does the heavy lifting:
# - Loads manifest.json configuration
# - Sets up Casbin authorization provider from manifest
# - Configures cross-app access via read_scopes (allows reading click_tracker data)
# - Seeds demo users (if configured)
# - Configures CORS, middleware, etc.
# - Returns a fully configured FastAPI app
#
# This is MDB-Engine specific - the create_app() method handles all the
# boilerplate of FastAPI + MongoDB + Authorization + Cross-App Access setup.

try:
    manifest_path = Path(__file__).parent / "manifest.json"
    logger.info(f"Creating FastAPI app with manifest: {manifest_path}")
    
    app = engine.create_app(
        slug=APP_SLUG,
        manifest=manifest_path,
        title="Click Tracker Dashboard",
        description="Analytics dashboard with cross-app data access",
        version="1.0.0",
    )
    logger.info("FastAPI app created successfully")
except Exception as e:
    logger.error(f"Failed to create FastAPI app: {e}", exc_info=True)
    sys.exit(1)

# Template engine for rendering HTML
# This is reusable - Jinja2 templates work with any FastAPI app
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# ============================================================================
# REUSABLE COMPONENTS
# ============================================================================
# These components are independent of MDB-Engine and can be reused
# in any FastAPI application. They define your business logic.

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

def get_app_config_from_state(request: Request) -> dict:
    """
    Get app configuration from app.state (set by engine.create_app()).
    
    This is a helper function that follows the MDB-Engine pattern of
    storing app configuration in app.state.manifest.
    
    Args:
        request: FastAPI Request object
        
    Returns:
        App manifest configuration dictionary
    """
    app_config = getattr(request.app.state, "manifest", None)
    if not app_config:
        # Fallback to engine.get_app() if not in state (shouldn't happen normally)
        app_config = engine.get_app(APP_SLUG) or {}
    return app_config

# ----------------------------------------------------------------------------
# Authentication Helpers
# ----------------------------------------------------------------------------

async def get_current_user(request: Request, db=Depends(get_scoped_db)):
    """
    Get the currently authenticated user from session cookie.
    
    MDB-Engine specific: Uses get_scoped_db() dependency and app state
    to access app configuration and database scoped to this app.
    
    Reusable: The pattern of getting user from session is standard and
    could work with any session management system.
    
    Args:
        request: FastAPI Request object
        db: Scoped database wrapper (from dependency)
        
    Returns:
        User document from MongoDB, or None if not authenticated
    """
    app_config = get_app_config_from_state(request)

    user = await get_app_user(
        request=request,
        slug_id=APP_SLUG,
        db=db,
        config=app_config,
        allow_demo_fallback=False,
    )
    return user

# ----------------------------------------------------------------------------
# Business Logic Functions
# ----------------------------------------------------------------------------

def convert_for_json(obj: Any) -> Any:
    """
    Recursively convert MongoDB objects to JSON-serializable format.
    
    This is reusable - handles ObjectId, datetime, dict, and list conversion
    for any MongoDB document structure.
    
    Args:
        obj: Object to convert (can be ObjectId, datetime, dict, list, or primitive)
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    return obj


def build_analytics_query(hours: int = 24) -> Dict[str, Any]:
    """
    Build a MongoDB query filter for time-based analytics.
    
    This is reusable query building logic.
    
    Args:
        hours: Number of hours to look back
        
    Returns:
        MongoDB query dictionary with timestamp filter
    """
    since = datetime.utcnow() - timedelta(hours=hours)
    return {"timestamp": {"$gte": since}}


def aggregate_click_stats(clicks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate click statistics from a list of click documents.
    
    This is reusable business logic - it doesn't depend on MDB-Engine.
    You could use this same function with any list of click dictionaries.
    
    Args:
        clicks: List of click documents
        
    Returns:
        Dictionary containing aggregated statistics:
        - total_clicks: Total number of clicks
        - unique_users: Number of unique user IDs
        - unique_sessions: Number of unique session IDs
        - top_urls: List of top URLs with counts
        - top_elements: List of top elements with counts
    """
    total_clicks = len(clicks)
    unique_users = len(set(c.get("user_id") for c in clicks if c.get("user_id")))
    unique_sessions = len(set(c.get("session_id") for c in clicks if c.get("session_id")))

    # Top URLs
    url_counts = {}
    for click in clicks:
        url = click.get("url", "unknown")
        url_counts[url] = url_counts.get(url, 0) + 1
    top_urls = sorted(url_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # Top elements
    element_counts = {}
    for click in clicks:
        element = click.get("element", "unknown")
        element_counts[element] = element_counts.get(element, 0) + 1
    top_elements = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_clicks": total_clicks,
        "unique_users": unique_users,
        "unique_sessions": unique_sessions,
        "top_urls": [{"url": url, "count": count} for url, count in top_urls],
        "top_elements": [{"element": elem, "count": count} for elem, count in top_elements],
    }

# ============================================================================
# ROUTES
# ============================================================================
# Routes combine MDB-Engine dependencies (get_scoped_db, get_authz_provider)
# with reusable business logic functions. The routes demonstrate cross-app
# data access patterns.

# ----------------------------------------------------------------------------
# Web Routes (HTML)
# ----------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    Root endpoint - HTML dashboard page.
    
    Reusable: Template rendering works with any FastAPI app.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    """
    Health check endpoint for container orchestration.
    
    MDB-Engine specific: Uses engine.initialized and app.state.authz_provider
    to report engine and authorization status.
    
    This endpoint is useful for Docker healthchecks and monitoring.
    """
    return {
        "status": "healthy",
        "app": APP_SLUG,
        "engine": "initialized" if engine.initialized else "starting",
        "authz": "configured" if hasattr(app.state, "authz_provider") else "not_configured",
    }


@app.get("/api")
async def api_info():
    """
    API info endpoint - lists available endpoints.
    
    Reusable: Endpoint documentation pattern works with any API.
    """
    return {
        "app": APP_SLUG,
        "endpoints": {
            "GET /": "HTML dashboard page",
            "POST /login": "Authenticate user",
            "GET /logout": "Logout user",
            "GET /api/me": "Get current user info and permissions",
            "GET /analytics": "Get click analytics (requires read permission)",
            "GET /health": "Health check",
        },
    }

# ----------------------------------------------------------------------------
# Authentication Routes
# ----------------------------------------------------------------------------

@app.post("/login")
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db=Depends(get_scoped_db),
):
    """
    Authenticate user and create session.
    
    MDB-Engine specific: Uses authenticate_app_user() and create_app_session()
    which handle app-scoped user authentication and session management.
    
    Reusable: The authentication flow pattern (verify credentials, create session)
    is standard and could work with any auth system.
    """
    user = await authenticate_app_user(
        db=db,
        email=email,
        password=password,
        collection_name="users",
    )

    if not user:
        return JSONResponse(
            status_code=401,
            content={"success": False, "detail": "Invalid credentials"},
        )

    response = JSONResponse(content={"success": True, "user_id": str(user["_id"])})
    app_config = get_app_config_from_state(request)

    await create_app_session(
        request=request,
        slug_id=APP_SLUG,
        user_id=str(user["_id"]),
        config=app_config,
        response=response,
    )

    return response


@app.post("/logout")
async def logout(request: Request):
    """
    Clear session and logout user.
    
    MDB-Engine specific: Uses logout_user() and app configuration to clear
    app-specific session cookies.
    
    Reusable: The logout pattern (clear session, delete cookies) is standard.
    """
    response = JSONResponse(content={"success": True})
    response = await logout_user(request, response)

    app_config = get_app_config_from_state(request)
    if app_config:
        auth = app_config.get("auth", {})
        users_config = auth.get("users", {})
        cookie_name = f"{users_config.get('session_cookie_name', 'app_session')}_{APP_SLUG}"
        response.delete_cookie(key=cookie_name, httponly=True, samesite="lax")

    return response

# ----------------------------------------------------------------------------
# User Info & Permissions Routes
# ----------------------------------------------------------------------------

@app.get("/api/me")
async def get_me(request: Request, authz=Depends(get_authz_provider), db=Depends(get_scoped_db)):
    """
    Get current user info and their permissions.
    
    MDB-Engine specific: Uses get_authz_provider() to get Casbin authorization
    provider, then uses authz.check() to evaluate permissions based on Casbin
    policy rules defined in manifest.json.
    
    Reusable: The permission checking pattern (check multiple actions, return
    list of allowed actions) is standard and could work with any authorization
    system.
    """
    user = await get_current_user(request, db=db)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    email = user.get("email", "unknown")
    permissions = []

    if authz:
        for action in ["read", "export"]:
            try:
                if await authz.check(email, "analytics", action):
                    permissions.append(action)
            except Exception:
                pass

    return {
        "email": email,
        "role": user.get("role", "unknown"),
        "permissions": permissions,
    }

# ----------------------------------------------------------------------------
# Analytics Routes (Cross-App Data Access)
# ----------------------------------------------------------------------------

@app.get("/analytics")
async def get_analytics(
    request: Request,
    hours: int = 24,
    authz=Depends(get_authz_provider),
    db=Depends(get_scoped_db),
):
    """
    Get click analytics from ClickTracker app.
    
    This demonstrates CROSS-APP DATA ACCESS:
    - Dashboard's manifest declares read_scopes: ["click_tracker"]
    - This allows Dashboard to read click_tracker's collections
    - MDB-Engine validates cross-app access permissions
    
    MDB-Engine specific:
    - Uses get_scoped_db() for scoped database access
    - Uses db.get_collection("click_tracker_clicks") to access another app's collection
    - Cross-app access is validated by MDB-Engine based on manifest read_scopes
    
    Reusable:
    - Query building uses reusable build_analytics_query() function
    - Analytics aggregation uses reusable aggregate_click_stats() function
    - JSON conversion uses reusable convert_for_json() function
    """
    user = await get_current_user(request, db=db)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Authorization check: Casbin evaluates policy rules
    if authz and not await authz.check(user.get("email"), "analytics", "read"):
        raise HTTPException(status_code=403, detail="Permission denied: cannot read analytics")

    # Build query using reusable function
    query = build_analytics_query(hours)

    # CROSS-APP ACCESS: Read click_tracker's clicks collection!
    # This works because manifest has read_scopes: ["click_tracker"]
    # Collection name is prefixed: click_tracker_clicks
    clicks_collection = db.get_collection("click_tracker_clicks")
    clicks = await clicks_collection.find(query).sort("timestamp", -1).to_list(length=1000)

    # Convert for JSON using reusable function
    clicks = convert_for_json(clicks)

    # Aggregate statistics using reusable function
    stats = aggregate_click_stats(clicks)

    return JSONResponse(content={
        "period_hours": hours,
        **stats,
        "recent_clicks": clicks[:50],
        "queried_by": user.get("email"),
    })


@app.get("/export")
async def export_analytics(
    request: Request,
    hours: int = 24,
    authz=Depends(get_authz_provider),
    db=Depends(get_scoped_db),
):
    """
    Export full click data (admin only).
    
    Requires 'export' permission on 'analytics'.
    
    MDB-Engine specific:
    - Uses get_scoped_db() for scoped database access
    - Uses db.get_collection() for cross-app access
    - Uses get_authz_provider() for Casbin authorization
    
    Reusable:
    - Query building and JSON conversion use reusable functions
    - The export pattern (fetch data, serialize, return) is standard
    """
    user = await get_current_user(request, db=db)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Authorization check: Only admins can export
    if authz and not await authz.check(user.get("email"), "analytics", "export"):
        raise HTTPException(status_code=403, detail="Permission denied: cannot export analytics")

    # Build query using reusable function
    query = build_analytics_query(hours)

    # CROSS-APP ACCESS: Read click_tracker's clicks collection
    clicks_collection = db.get_collection("click_tracker_clicks")
    clicks = await clicks_collection.find(query).sort("timestamp", -1).to_list(length=10000)

    # Convert for JSON using reusable function
    clicks = convert_for_json(clicks)

    return JSONResponse(content={
        "export_type": "full",
        "period_hours": hours,
        "total_records": len(clicks),
        "exported_by": user.get("email"),
        "exported_at": datetime.utcnow().isoformat(),
        "data": clicks,
    })


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting uvicorn server on 0.0.0.0:8001")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8001, log_level=os.getenv("LOG_LEVEL", "info").lower())
    except Exception as e:
        logger.error(f"Failed to start uvicorn server: {e}", exc_info=True)
        sys.exit(1)
