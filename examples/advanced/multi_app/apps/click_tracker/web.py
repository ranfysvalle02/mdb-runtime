#!/usr/bin/env python3
"""
Click Tracker Application
=========================

A clean example demonstrating click tracking with role-based access control
using Casbin authorization.

This example shows:
- How to use engine.create_app() for automatic lifecycle management
- How to use Casbin authorization with get_authz_provider()
- How to track user events with permission checks
- How to separate MDB-Engine specifics from reusable business logic

Key Concepts:
- engine.create_app() - Creates FastAPI app with automatic lifecycle management
- get_scoped_db() - Provides database access scoped to your app
- get_authz_provider() - Provides Casbin authorization provider
- authz.check() - Checks permissions using Casbin policy evaluation
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

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

APP_SLUG = "click_tracker"

# ============================================================================
# STEP 1: INITIALIZE MONGODB ENGINE
# ============================================================================
# The MongoDBEngine is the core of mdb-engine. It manages:
# - Database connections
# - App registration and configuration
# - Authorization providers (Casbin, OSO Cloud, etc.)
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
# - Seeds demo users (if configured)
# - Configures CORS, middleware, etc.
# - Returns a fully configured FastAPI app
#
# This is MDB-Engine specific - the create_app() method handles all the
# boilerplate of FastAPI + MongoDB + Authorization setup.

try:
    manifest_path = Path(__file__).parent / "manifest.json"
    logger.info(f"Creating FastAPI app with manifest: {manifest_path}")
    
    app = engine.create_app(
        slug=APP_SLUG,
        manifest=manifest_path,
        title="Click Tracker",
        description="Track user clicks with role-based access control",
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

def create_click_document(
    user_id: str,
    url: str = "/",
    element: str = "unknown",
    session_id: str = "default_session",
    tracked_by: Optional[str] = None,
) -> dict:
    """
    Create a click document with standard fields.
    
    This is reusable business logic - it doesn't depend on MDB-Engine.
    You could use this same function with any MongoDB driver.
    
    Args:
        user_id: ID of the user who clicked
        url: URL where the click occurred
        element: Element that was clicked
        session_id: Session identifier
        tracked_by: Email of user who tracked this (for audit)
        
    Returns:
        Dictionary representing a click document
    """
    return {
        "user_id": user_id,
        "timestamp": datetime.utcnow(),
        "session_id": session_id,
        "url": url,
        "element": element,
        "tracked_by": tracked_by or user_id,
    }


def serialize_click(click: dict) -> dict:
    """
    Convert a click document to a JSON-serializable format.
    
    This is reusable - handles ObjectId and datetime conversion for any MongoDB document.
    
    Args:
        click: Click document from MongoDB
        
    Returns:
        Click document with _id converted to string and timestamp to ISO format
    """
    if "_id" in click:
        click["_id"] = str(click["_id"])
    if click.get("timestamp"):
        click["timestamp"] = click["timestamp"].isoformat()
    return click


def build_click_query(user_id: Optional[str] = None) -> dict:
    """
    Build a MongoDB query filter for clicks.
    
    This is reusable query building logic.
    
    Args:
        user_id: Optional filter by user ID
        
    Returns:
        MongoDB query dictionary
    """
    query = {}
    if user_id:
        query["user_id"] = user_id
    return query

# ============================================================================
# ROUTES
# ============================================================================
# Routes combine MDB-Engine dependencies (get_scoped_db, get_authz_provider)
# with reusable business logic functions. The routes themselves are FastAPI-specific
# but the pattern is reusable.

# ----------------------------------------------------------------------------
# Web Routes (HTML)
# ----------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    Root endpoint - HTML demo page.
    
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
    authz_provider = getattr(app.state, "authz_provider", None)
    
    # Get app manifest to check auth config
    app_config = getattr(app.state, "manifest", None)
    auth_config = app_config.get("auth", {}) if app_config else {}
    auth_policy = auth_config.get("policy", {}) if auth_config else {}
    
    return {
        "status": "healthy",
        "app": APP_SLUG,
        "engine": "initialized" if engine.initialized else "starting",
        "authz_provider_exists": authz_provider is not None,
        "authz_provider_type": type(authz_provider).__name__ if authz_provider else None,
        "authz": "configured" if authz_provider else "not_configured",
        "auth_config": {
            "has_auth": "auth" in (app_config or {}),
            "has_policy": "policy" in auth_config,
            "provider": auth_policy.get("provider") if auth_policy else None,
            "has_authorization": "authorization" in auth_policy if auth_policy else False,
        },
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
            "GET /": "HTML demo page",
            "POST /login": "Authenticate user",
            "GET /logout": "Logout user",
            "GET /api/me": "Get current user info and permissions",
            "GET /api/debug": "Debug Casbin state (dev only)",
            "POST /track": "Track a click event (requires write permission)",
            "GET /clicks": "Get click history (requires read permission)",
            "GET /health": "Health check",
        },
    }


@app.get("/api/debug")
async def debug_authz(request: Request, db=Depends(get_scoped_db)):
    """
    Debug endpoint to inspect Casbin authorization state and initialization status.
    """
    user = await get_current_user(request, db=db)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    email = user.get("email", "unknown")
    
    # Check if authz_provider exists on app.state (without requiring it as a dependency)
    authz_provider = getattr(request.app.state, "authz_provider", None)
    
    # Get app manifest to check auth config
    app_config = get_app_config_from_state(request)
    auth_config = app_config.get("auth", {})
    auth_policy = auth_config.get("policy", {})
    
    debug_info = {
        "user_email": email,
        "user_role": user.get("role", "unknown"),
        "authz_provider_exists": authz_provider is not None,
        "authz_provider_type": type(authz_provider).__name__ if authz_provider else None,
        "app_state_keys": [key for key in dir(request.app.state) if not key.startswith("_")],
        "auth_config": {
            "has_auth": "auth" in app_config,
            "has_policy": "policy" in auth_config,
            "provider": auth_policy.get("provider"),
            "has_authorization": "authorization" in auth_policy,
        },
        "engine_initialized": engine.initialized if hasattr(engine, "initialized") else None,
    }
    
    authz = authz_provider
    
    if authz:
        enforcer = authz._enforcer
        try:
            # Check role assignments
            roles = []
            for role in ["admin", "editor", "viewer"]:
                if hasattr(authz, "has_role_for_user"):
                    has_role = await authz.has_role_for_user(email, role)
                    if has_role:
                        roles.append(role)
            
            debug_info["assigned_roles"] = roles
            
            # Check permissions
            permissions = {}
            for action in ["read", "write", "delete"]:
                result = await authz.check(email, "clicks", action)
                permissions[action] = result
            
            debug_info["permissions"] = permissions
            
            # Try to get all policies (if possible)
            try:
                if hasattr(enforcer, "get_all_policies"):
                    all_policies = await enforcer.get_all_policies()
                    debug_info["all_policies"] = all_policies
            except Exception as e:
                debug_info["policy_check_error"] = str(e)
            
            # Try to get grouping policies (role assignments)
            try:
                if hasattr(enforcer, "get_all_grouping_policies"):
                    all_grouping = await enforcer.get_all_grouping_policies()
                    debug_info["all_role_assignments"] = all_grouping
            except Exception as e:
                debug_info["grouping_check_error"] = str(e)
                
        except Exception as e:
            debug_info["error"] = str(e)
            logger.error(f"Debug endpoint error: {e}", exc_info=True)
    else:
        debug_info["error"] = "No authz provider or enforcer available"
    
    return debug_info

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
    role = user.get("role", "unknown")
    permissions = []

    if authz:
        # Check permissions using email (Casbin RBAC resolves email -> role -> permissions)
        # The initial_roles in manifest.json assign roles using email as the subject
        for action in ["read", "write", "delete"]:
            try:
                result = await authz.check(email, "clicks", action)
                logger.debug(f"Permission check: {email} -> clicks:{action} = {result}")
                if result:
                    permissions.append(action)
            except Exception as e:
                logger.error(f"Permission check failed for {email} on clicks:{action}: {e}", exc_info=True)
    else:
        logger.warning(f"No authz provider available for user {email}")
    
    logger.info(f"User {email} (role: {role}) has permissions: {permissions}")

    return {
        "email": email,
        "role": role,
        "permissions": permissions,
    }

# ----------------------------------------------------------------------------
# Click Tracking Routes (Protected by Casbin)
# ----------------------------------------------------------------------------

@app.post("/track")
async def track_click(
    request: Request,
    authz=Depends(get_authz_provider),
    db=Depends(get_scoped_db),
):
    """
    Track a click event. Requires 'write' permission on 'clicks'.
    
    MDB-Engine specific: 
    - Uses get_scoped_db() for scoped database access
    - Uses get_authz_provider() for Casbin authorization
    - Authorization check happens BEFORE database write
    
    Reusable:
    - Click document creation uses reusable create_click_document() function
    - The authorization pattern (check permission, then perform action) is standard
    """
    user = await get_current_user(request, db=db)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Authorization check: Casbin evaluates policy rules
    email = user.get("email", "unknown")
    if authz:
        has_permission = await authz.check(email, "clicks", "write")
        logger.info(f"Track click permission check: {email} -> clicks:write = {has_permission}")
        if not has_permission:
            # Check if user has role assigned
            if hasattr(authz, "has_role_for_user"):
                roles = []
                for role in ["admin", "editor", "viewer"]:
                    if await authz.has_role_for_user(email, role):
                        roles.append(role)
                logger.warning(f"User {email} does not have write permission. Assigned roles: {roles}")
            raise HTTPException(status_code=403, detail="Permission denied: cannot write clicks")
    else:
        logger.error("No authz provider available for track endpoint")
        raise HTTPException(status_code=500, detail="Authorization system not configured")

    # Parse request body (optional - uses defaults if not provided)
    try:
        body = await request.json()
    except Exception:
        body = {}

    # Create click document using reusable function
    click_doc = create_click_document(
        user_id=body.get("user_id", user.get("email")),
        url=body.get("url", "/"),
        element=body.get("element", "unknown"),
        session_id=body.get("session_id", "default_session"),
        tracked_by=user.get("email"),
    )

    # Insert into database - automatically scoped to this app
    result = await db.clicks.insert_one(click_doc)

    return JSONResponse(content={
        "click_id": str(result.inserted_id),
        "status": "tracked",
    })


@app.get("/clicks")
async def get_clicks(
    request: Request,
    user_id: Optional[str] = None,
    limit: int = 100,
    authz=Depends(get_authz_provider),
    db=Depends(get_scoped_db),
):
    """
    Get click history. Requires 'read' permission on 'clicks'.
    
    MDB-Engine specific:
    - Uses get_scoped_db() for scoped database access
    - Uses get_authz_provider() for Casbin authorization
    
    Reusable:
    - Query building uses reusable build_click_query() function
    - Serialization uses reusable serialize_click() function
    - The authorization + query pattern is standard
    """
    user = await get_current_user(request, db=db)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Authorization check: Casbin evaluates policy rules
    if authz and not await authz.check(user.get("email"), "clicks", "read"):
        raise HTTPException(status_code=403, detail="Permission denied: cannot read clicks")

    # Build query using reusable function
    query = build_click_query(user_id)

    # Fetch clicks - database is automatically scoped to this app
    clicks = await db.clicks.find(query).sort("timestamp", -1).limit(limit).to_list(length=limit)

    # Serialize clicks using reusable function
    for click in clicks:
        serialize_click(click)

    return {"clicks": clicks, "count": len(clicks)}


@app.delete("/clicks/{click_id}")
async def delete_click(
    click_id: str,
    request: Request,
    authz=Depends(get_authz_provider),
    db=Depends(get_scoped_db),
):
    """
    Delete a click. Requires 'delete' permission on 'clicks'.
    
    MDB-Engine specific:
    - Uses get_scoped_db() for scoped database access
    - Uses get_authz_provider() for Casbin authorization
    
    Reusable:
    - ObjectId conversion is standard MongoDB operation
    - The authorization + delete pattern is standard
    """
    user = await get_current_user(request, db=db)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Authorization check: Casbin evaluates policy rules
    if authz and not await authz.check(user.get("email"), "clicks", "delete"):
        raise HTTPException(status_code=403, detail="Permission denied: cannot delete clicks")

    # Delete click - database is automatically scoped to this app
    result = await db.clicks.delete_one({"_id": ObjectId(click_id)})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Click not found")

    return {"deleted": True, "click_id": click_id}


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting uvicorn server on 0.0.0.0:8000")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level=os.getenv("LOG_LEVEL", "info").lower())
    except Exception as e:
        logger.error(f"Failed to start uvicorn server: {e}", exc_info=True)
        sys.exit(1)
