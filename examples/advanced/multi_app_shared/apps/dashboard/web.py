#!/usr/bin/env python3
"""
Analytics Dashboard - MDB-Engine SSO Demo
=========================================

A clean example demonstrating Single Sign-On (SSO), cross-app data access,
and admin role management.

This example shows:
- How to use engine.create_app() with shared authentication mode
- How to access data from other apps via read_scopes (cross-app access)
- How to manage user roles across multiple apps (admin features)
- How to separate MDB-Engine specifics from reusable business logic

Key Concepts:
- engine.create_app() - Creates FastAPI app with automatic lifecycle management
- SharedUserPool - Centralized user management (MDB-Engine specific)
- Cross-app access - Reading data from other apps via read_scopes
- db.get_collection() - Access collections from other apps
- Per-app role management - Admins can update roles for any app
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request, Depends, Form
from fastapi.responses import JSONResponse, HTMLResponse
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
    from mdb_engine.dependencies import get_scoped_db
except ImportError as e:
    logger.error(f"Failed to import mdb_engine: {e}", exc_info=True)
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================
# Application constants and configuration
# These are reusable across different database backends

APP_SLUG = "dashboard"

# Shared cookie name for SSO - must match across all apps for SSO to work
# This is a reusable pattern - any SSO system would use shared cookies
AUTH_COOKIE = "mdb_auth_token"

# ============================================================================
# STEP 1: INITIALIZE MONGODB ENGINE
# ============================================================================
# The MongoDBEngine is the core of mdb-engine. It manages:
# - Database connections
# - App registration and configuration
# - SharedUserPool initialization (for auth.mode="shared")
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
# - Detects auth.mode="shared" and initializes SharedUserPool
# - Auto-adds SharedAuthMiddleware (populates request.state.user)
# - Configures cross-app access via read_scopes (allows reading click_tracker data)
# - Seeds demo users to SharedUserPool (from manifest.json demo_users)
# - Configures CORS, middleware, etc.
# - Returns a fully configured FastAPI app
#
# This is MDB-Engine specific - the create_app() method handles all the
# boilerplate of FastAPI + MongoDB + Shared Authentication + Cross-App Access setup.

try:
    manifest_path = Path(__file__).parent / "manifest.json"
    logger.info(f"Creating FastAPI app with manifest: {manifest_path}")
    
    app = engine.create_app(
        slug=APP_SLUG,
        manifest=manifest_path,
        title="Analytics Dashboard (SSO)",
        description="Analytics dashboard with SSO and cross-app data access",
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
# SSO Helper Functions
# ----------------------------------------------------------------------------

def get_user_pool():
    """
    Get the shared user pool from app state.
    
    MDB-Engine specific: SharedUserPool is initialized by engine.create_app()
    when auth.mode="shared" is detected in manifest.json. It's stored in
    app.state.user_pool for access throughout the app.
    
    Reusable: The pattern of accessing shared services from app state is
    standard FastAPI and could work with any shared service.
    
    Returns:
        SharedUserPool instance, or None if not initialized
    """
    return getattr(app.state, "user_pool", None)


def get_current_user(request: Request) -> Optional[dict]:
    """
    Get user from request.state (populated by SharedAuthMiddleware).
    
    MDB-Engine specific: SharedAuthMiddleware (auto-added by create_app())
    validates the SSO JWT token from cookies and populates request.state.user.
    This happens automatically on every request.
    
    Reusable: The pattern of getting user from request.state is standard
    FastAPI middleware pattern and could work with any auth middleware.
    
    Args:
        request: FastAPI Request object
        
    Returns:
        User document, or None if not authenticated
    """
    return getattr(request.state, "user", None)


def get_roles(user: Optional[dict], app_slug: str = APP_SLUG) -> list:
    """
    Get user's roles for specified app.
    
    This is reusable business logic - it extracts per-app roles from a
    user document structure. The pattern works with any role-based system.
    
    Args:
        user: User document (may be None)
        app_slug: App slug to get roles for (defaults to this app)
        
    Returns:
        List of role strings for the specified app
    """
    if not user:
        return []
    app_roles = user.get("app_roles", {})
    return app_roles.get(app_slug, [])


def get_primary_role(user: Optional[dict]) -> str:
    """
    Get user's primary role for display (highest priority role).
    
    This is reusable business logic - role priority logic is standard
    across role-based systems.
    
    Role priority: admin > tracker > clicker
    
    Args:
        user: User document (may be None)
        
    Returns:
        Primary role string, or "clicker" if no roles
    """
    roles = get_roles(user)
    if "admin" in roles:
        return "admin"
    if "tracker" in roles:
        return "tracker"
    return "clicker"


def can_view_analytics(user: Optional[dict]) -> bool:
    """
    Check if user can view analytics (trackers and admins can).
    
    This is reusable permission checking logic - it evaluates role-based
    permissions based on business rules.
    
    Args:
        user: User document (may be None)
        
    Returns:
        True if user can view analytics, False otherwise
    """
    roles = get_roles(user)
    return "tracker" in roles or "admin" in roles


def can_manage_users(user: Optional[dict]) -> bool:
    """
    Check if user can manage users (admins only).
    
    This is reusable permission checking logic - it evaluates role-based
    permissions based on business rules.
    
    Args:
        user: User document (may be None)
        
    Returns:
        True if user can manage users, False otherwise
    """
    return "admin" in get_roles(user)

# ----------------------------------------------------------------------------
# Business Logic Functions
# ----------------------------------------------------------------------------

def build_analytics_query(hours: int = 24) -> dict:
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


def aggregate_click_stats(clicks: list) -> dict:
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
        - clicks_by_role: Dictionary of role counts
        - top_urls: List of top URLs with counts
    """
    role_counts = {}
    url_counts = {}
    
    for c in clicks:
        # Count by role
        r = c.get("user_role", "unknown")
        role_counts[r] = role_counts.get(r, 0) + 1
        
        # Count by URL
        url = c.get("url", "/")
        url_counts[url] = url_counts.get(url, 0) + 1
    
    top_urls = sorted(url_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "total_clicks": len(clicks),
        "unique_users": len(set(c.get("user_id") for c in clicks)),
        "clicks_by_role": role_counts,
        "top_urls": [{"url": u, "count": c} for u, c in top_urls],
    }

# ============================================================================
# ROUTES
# ============================================================================
# Routes combine MDB-Engine dependencies (get_scoped_db) with reusable
# business logic functions. The routes demonstrate SSO + cross-app access patterns.

# ----------------------------------------------------------------------------
# Authentication Routes (SSO)
# ----------------------------------------------------------------------------

@app.post("/login")
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
):
    """
    Authenticate via SharedUserPool and set JWT cookie.
    
    MDB-Engine specific:
    - Uses SharedUserPool.authenticate() to verify credentials
    - SharedUserPool manages users in _mdb_engine_shared_users collection
    - Returns JWT token that works across all apps (SSO)
    
    Reusable:
    - The SSO cookie pattern (set shared cookie, works across apps) is standard
    - The authentication flow (verify credentials, return token) is standard
    """
    pool = get_user_pool()
    if not pool:
        raise HTTPException(500, "User pool not initialized")

    # Authenticate and get JWT token
    token = await pool.authenticate(email, password)

    if not token:
        return JSONResponse(status_code=401, content={
            "success": False,
            "detail": "Invalid credentials"
        })

    # Get user for response
    user = await pool.validate_token(token)
    role = get_primary_role(user) if user else "unknown"

    response = JSONResponse(content={
        "success": True,
        "user": {"email": email, "role": role},
        "can_view_analytics": can_view_analytics(user) if user else False,
        "sso": True,
    })

    # Set shared JWT cookie (works across all apps on same domain/ports)
    # For localhost: set domain="localhost" to work across ports
    # For production: set domain to your domain (e.g., ".example.com")
    host = request.url.hostname
    cookie_domain = None
    if host == "localhost" or host == "127.0.0.1":
        cookie_domain = "localhost"  # Explicit domain for localhost cross-port SSO
    # In production, you might set: cookie_domain = ".yourdomain.com"
    
    logger.info(f"Setting SSO cookie '{AUTH_COOKIE}' for user {email} on {host} (domain={cookie_domain})")
    response.set_cookie(
        key=AUTH_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,  # False for localhost (set True in production with HTTPS)
        max_age=86400,
        path="/",
        domain=cookie_domain,  # "localhost" for cross-port SSO on localhost
    )
    logger.info(f"✅ SSO cookie set successfully - user can now access all apps on {host}")

    return response


@app.post("/logout")
async def logout(request: Request):
    """
    Revoke token and clear cookie.
    
    MDB-Engine specific:
    - Uses SharedUserPool.revoke_token() to invalidate the JWT token
    - Token revocation prevents reuse across all apps (SSO logout)
    
    Reusable:
    - The logout pattern (revoke token, clear cookie) is standard SSO
    """
    pool = get_user_pool()
    token = request.cookies.get(AUTH_COOKIE)

    if pool and token:
        try:
            await pool.revoke_token(token)
        except Exception:
            pass

    response = JSONResponse(content={"success": True})
    # Delete cookie with same settings as set_cookie
    host = request.url.hostname
    cookie_domain = None
    if host == "localhost" or host == "127.0.0.1":
        cookie_domain = "localhost"  # Must match set_cookie settings
    response.delete_cookie(
        AUTH_COOKIE,
        path="/",
        domain=cookie_domain,  # Must match set_cookie settings
        secure=False,  # Must match set_cookie settings
        samesite="lax",  # Must match set_cookie settings
    )
    return response

# ----------------------------------------------------------------------------
# API Routes
# ----------------------------------------------------------------------------

@app.get("/api/me")
async def get_me(request: Request):
    """
    Get current user info.
    
    MDB-Engine specific: User comes from request.state.user (populated by
    SharedAuthMiddleware). No database query needed - user is already loaded.
    
    Reusable: Permission checking uses reusable can_view_analytics() and
    can_manage_users() functions. The response structure is standard for
    user info endpoints.
    """
    # Debug: Check cookie
    cookie_token = request.cookies.get(AUTH_COOKIE)
    logger.debug(f"Cookie '{AUTH_COOKIE}' present: {cookie_token is not None} (length: {len(cookie_token) if cookie_token else 0})")
    
    user = get_current_user(request)
    if not user:
        logger.warning(f"No user found in request.state.user - cookie present: {cookie_token is not None}")
        raise HTTPException(401, "Not authenticated")

    return {
        "email": user["email"],
        "roles": get_roles(user),
        "role": get_primary_role(user),
        "can_view_analytics": can_view_analytics(user),
        "can_manage_users": can_manage_users(user),
        "sso": True,
    }


@app.get("/analytics")
async def get_analytics(request: Request, hours: int = 24, db=Depends(get_scoped_db)):
    """
    Get click analytics from click_tracker's data.
    
    This demonstrates CROSS-APP DATA ACCESS:
    - Dashboard's manifest declares read_scopes: ["click_tracker"]
    - This allows Dashboard to read click_tracker's collections
    - MDB-Engine validates cross-app access permissions
    
    MDB-Engine specific:
    - Uses get_scoped_db() for scoped database access
    - Uses db.get_collection("click_tracker_clicks") to access another app's collection
    - Cross-app access is validated by MDB-Engine based on manifest read_scopes
    
    Reusable:
    - Permission checking uses reusable can_view_analytics() function
    - Query building uses reusable build_analytics_query() function
    - Analytics aggregation uses reusable aggregate_click_stats() function
    """
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    # Permission check using reusable function
    if not can_view_analytics(user):
        raise HTTPException(403, "Tracker or admin role required")

    # Build query using reusable function
    query = build_analytics_query(hours)

    # CROSS-APP ACCESS: Read click_tracker's clicks collection!
    # This works because manifest has read_scopes: ["click_tracker"]
    # Access via attribute: db.click_tracker_clicks (ScopedMongoWrapper handles cross-app access)
    clicks = await db.click_tracker_clicks.find(query).sort("timestamp", -1).to_list(1000)

    # Aggregate statistics using reusable function
    stats = aggregate_click_stats(clicks)

    # Format recent clicks
    recent_clicks = [
        {
            "user_id": c["user_id"],
            "user_role": c.get("user_role", "unknown"),
            "url": c.get("url", "/"),
            "timestamp": c["timestamp"].isoformat(),
        }
        for c in sorted(clicks, key=lambda x: x["timestamp"], reverse=True)[:20]
    ]

    return {
        "period_hours": hours,
        **stats,
        "recent_clicks": recent_clicks,
        "cross_app_access": True,  # Indicate we're reading from click_tracker
    }

# ----------------------------------------------------------------------------
# Admin Routes
# ----------------------------------------------------------------------------

@app.get("/admin/users")
async def list_users(request: Request):
    """
    List all shared users. Requires: admin role.
    
    MDB-Engine specific:
    - Accesses shared users via SharedUserPool from app.state
    - This collection is managed by SharedUserPool
    
    Reusable:
    - Permission checking uses reusable can_manage_users() function
    - User listing and formatting logic is standard
    """
    user = get_current_user(request)
    if not user or not can_manage_users(user):
        raise HTTPException(403, "Admin role required")

    # Get shared users collection via user pool
    pool = get_user_pool()
    if not pool:
        raise HTTPException(500, "User pool not initialized")
    
    # Access the collection through the pool
    shared_users = await pool._collection.find({}).to_list(100)

    return {
        "users": [
            {
                "email": u["email"],
                "app_roles": u.get("app_roles", {}),
                "click_tracker_role": get_roles(u, "click_tracker"),
                "dashboard_role": get_roles(u, "dashboard"),
                "created_at": u.get("created_at").isoformat() if u.get("created_at") else None,
            }
            for u in shared_users
        ],
        "shared_pool": True,
    }


@app.post("/admin/update-role")
async def update_role(
    request: Request,
    email: str = Form(...),
    role: str = Form(...),
    target_app: str = Form("dashboard"),
):
    """
    Update user role for a specific app.
    
    Admins can update roles for ANY app (click_tracker or dashboard)!
    
    MDB-Engine specific:
    - Uses SharedUserPool.update_user_roles() to modify roles
    - SharedUserPool manages role updates in _mdb_engine_shared_users collection
    
    Reusable:
    - Permission checking uses reusable can_manage_users() function
    - Role validation logic is standard
    - The admin role management pattern is standard
    """
    user = get_current_user(request)
    if not user or not can_manage_users(user):
        raise HTTPException(403, "Admin role required")

    if role not in ["clicker", "tracker", "admin"]:
        raise HTTPException(400, "Invalid role")

    if target_app not in ["click_tracker", "dashboard"]:
        raise HTTPException(400, "Invalid target app")

    pool = get_user_pool()
    if not pool:
        raise HTTPException(500, "User pool not initialized")

    # Update the user's role for the target app
    success = await pool.update_user_roles(email, target_app, [role])

    if not success:
        raise HTTPException(404, "User not found")

    return {
        "success": True,
        "message": f"{email} is now {role} on {target_app}",
        "target_app": target_app,
    }

# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Dashboard page.
    
    Reusable: Template rendering works with any FastAPI app.
    """
    user = get_current_user(request)
    role = get_primary_role(user) if user else None

    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": user,
        "role": role,
        "can_view_analytics": can_view_analytics(user) if user else False,
        "can_manage_users": can_manage_users(user) if user else False,
    })


@app.get("/health")
async def health():
    """
    Health check endpoint.
    
    Reusable: Health check pattern works with any app.
    """
    return {"status": "healthy", "app": APP_SLUG, "auth": "sso"}


@app.get("/api")
async def api_info():
    """
    API info endpoint.
    
    Reusable: Endpoint documentation pattern works with any API.
    """
    return {
        "app": APP_SLUG,
        "auth_mode": "shared (SSO)",
        "cross_app_access": ["click_tracker"],
        "demo_users": [
            "alice@example.com (admin)",
            "bob@example.com (tracker)",
            "charlie@example.com (clicker - no access)"
        ],
        "password": "password123",
        "sso_note": "Login on Click Tracker = logged in here too!",
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

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
