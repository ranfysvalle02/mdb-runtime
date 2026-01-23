#!/usr/bin/env python3
"""
Click Tracker - MDB-Engine SSO Demo
===================================

A clean example demonstrating Single Sign-On (SSO) authentication with
shared user pool and per-app role management.

This example shows:
- How to use engine.create_app() with shared authentication mode
- How SharedUserPool provides centralized user management
- How JWT tokens work across multiple apps (SSO)
- How per-app roles enable different permissions per app
- How to separate MDB-Engine specifics from reusable business logic

Key Concepts:
- engine.create_app() - Creates FastAPI app with automatic lifecycle management
- SharedUserPool - Centralized user management (MDB-Engine specific)
- SharedAuthMiddleware - Auto-added for auth.mode="shared" (MDB-Engine specific)
- SSO Cookies - Shared JWT token across apps (reusable pattern)
- Per-App Roles - Users have different roles per app (reusable pattern)
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from fastapi import Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# Configure logging early
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
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

APP_SLUG = "click_tracker"

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
# - Seeds demo users to SharedUserPool (from manifest.json demo_users)
# - Configures CORS, middleware, etc.
# - Returns a fully configured FastAPI app
#
# This is MDB-Engine specific - the create_app() method handles all the
# boilerplate of FastAPI + MongoDB + Shared Authentication setup.

try:
    manifest_path = Path(__file__).parent / "manifest.json"
    logger.info(f"Creating FastAPI app with manifest: {manifest_path}")

    app = engine.create_app(
        slug=APP_SLUG,
        manifest=manifest_path,
        title="Click Tracker (SSO)",
        description="Click tracking with Single Sign-On authentication",
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


def get_related_app_url(app_slug: str) -> str:
    """
    Get URL for a related app from manifest or environment variable.

    Priority:
    1. manifest.auth.related_apps[app_slug]
    2. {APP_SLUG_UPPER}_URL environment variable (e.g., DASHBOARD_URL)
    3. Default based on app_slug

    Returns:
        Related app URL string
    """
    # Try manifest first
    manifest = getattr(app.state, "manifest", None)
    if manifest:
        auth_config = manifest.get("auth", {})
        related_apps = auth_config.get("related_apps", {})
        if app_slug in related_apps:
            return related_apps[app_slug]

    # Fallback to environment variable
    env_var = f"{app_slug.upper().replace('-', '_')}_URL"
    default_urls = {
        "dashboard": "http://localhost:8001",
        "click_tracker": "http://localhost:8000",
    }
    return os.getenv(env_var, default_urls.get(app_slug, "http://localhost:8000"))


def get_current_user(request: Request) -> dict | None:
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


def get_roles(user: dict | None) -> list:
    """
    Get user's roles for this app.

    This is reusable business logic - it extracts per-app roles from a
    user document structure. The pattern works with any role-based system.

    Args:
        user: User document (may be None)

    Returns:
        List of role strings for this app
    """
    if not user:
        return []
    app_roles = user.get("app_roles", {})
    return app_roles.get(APP_SLUG, [])


def get_primary_role(user: dict | None) -> str:
    """
    Get user's primary role for display (highest priority role).

    This is reusable business logic - role priority logic is standard
    across role-based systems.

    Role priority: admin > tracker > clicker

    Args:
        user: User document (may be None)

    Returns:
        Primary role string, or "none" if no roles
    """
    roles = get_roles(user)
    # Priority: admin > tracker > clicker
    if "admin" in roles:
        return "admin"
    if "tracker" in roles:
        return "tracker"
    return "clicker"


def can_view_all(user: dict | None) -> bool:
    """
    Check if user can view all clicks (trackers and admins can).

    This is reusable permission checking logic - it evaluates role-based
    permissions based on business rules.

    Args:
        user: User document (may be None)

    Returns:
        True if user can view all clicks, False otherwise
    """
    roles = get_roles(user)
    return "tracker" in roles or "admin" in roles


# ----------------------------------------------------------------------------
# Business Logic Functions
# ----------------------------------------------------------------------------


def create_click_document(
    user_id: str,
    user_role: str,
    url: str = "/",
    element: str = "button",
) -> dict:
    """
    Create a click document with standard fields.

    This is reusable business logic - it doesn't depend on MDB-Engine.
    You could use this same function with any MongoDB driver.

    Args:
        user_id: ID of the user who clicked
        user_role: Role of the user (for analytics)
        url: URL where the click occurred
        element: Element that was clicked

    Returns:
        Dictionary representing a click document
    """
    return {
        "user_id": user_id,
        "user_role": user_role,
        "url": url,
        "element": element,
        "timestamp": datetime.utcnow(),
    }


def serialize_clicks(clicks: list) -> list:
    """
    Convert click documents to JSON-serializable format.

    This is reusable - handles ObjectId and datetime conversion for any
    MongoDB document list.

    Args:
        clicks: List of click documents from MongoDB

    Returns:
        List of click documents with _id as string and timestamp as ISO format
    """
    return [
        {
            "id": str(c["_id"]),
            "user_id": c["user_id"],
            "user_role": c.get("user_role", "unknown"),
            "url": c.get("url", "/"),
            "timestamp": c["timestamp"].isoformat(),
        }
        for c in clicks
    ]


# ============================================================================
# ROUTES
# ============================================================================
# Routes combine MDB-Engine dependencies (get_scoped_db) with reusable
# business logic functions. The routes demonstrate SSO authentication patterns.

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
        return JSONResponse(
            status_code=401, content={"success": False, "detail": "Invalid credentials"}
        )

    # Get user for response
    user = await pool.validate_token(token)
    role = get_primary_role(user) if user else "unknown"

    response = JSONResponse(
        content={
            "success": True,
            "user": {"email": email, "role": role},
            "sso": True,  # Indicate SSO login
        }
    )

    # Set shared JWT cookie (works across all apps on same domain/ports)
    # This is the SSO magic - same cookie works for all apps!
    # For localhost: set domain="localhost" to work across ports
    # For production: set domain to your domain (e.g., ".example.com")
    host = request.url.hostname
    cookie_domain = None
    if host == "localhost" or host == "127.0.0.1":
        cookie_domain = "localhost"  # Explicit domain for localhost cross-port SSO
    # In production, you might set: cookie_domain = ".yourdomain.com"

    logger.info(
        f"Setting SSO cookie '{AUTH_COOKIE}' for user {email} on {host} (domain={cookie_domain})"
    )
    response.set_cookie(
        key=AUTH_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,  # False for localhost (set True in production with HTTPS)
        max_age=86400,  # 24 hours
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

    # Get token from cookie
    token = request.cookies.get(AUTH_COOKIE)

    # Revoke token if we have pool and token
    if pool and token:
        try:
            await pool.revoke_token(token)
        except (ValueError, KeyError, AttributeError):
            pass  # Token may already be invalid

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
    Get current user info and permissions.

    MDB-Engine specific: User comes from request.state.user (populated by
    SharedAuthMiddleware). No database query needed - user is already loaded.

    Reusable: Permission checking uses reusable get_roles() and can_view_all()
    functions. The response structure is standard for user info endpoints.
    """
    # Debug: Check cookie
    cookie_token = request.cookies.get(AUTH_COOKIE)
    cookie_length = len(cookie_token) if cookie_token else 0
    logger.debug(
        f"Cookie '{AUTH_COOKIE}' present: {cookie_token is not None} " f"(length: {cookie_length})"
    )

    user = get_current_user(request)
    if not user:
        logger.warning(
            f"No user found in request.state.user - cookie present: {cookie_token is not None}"
        )
        raise HTTPException(401, "Not authenticated")

    return {
        "email": user["email"],
        "roles": get_roles(user),
        "role": get_primary_role(user),
        "can_view_all": can_view_all(user),
        "sso": True,
    }


@app.post("/track")
async def track_click(request: Request, db=Depends(get_scoped_db)):
    """
    Track a click event.

    MDB-Engine specific:
    - Uses get_scoped_db() for scoped database access
    - User comes from request.state.user (SSO middleware)

    Reusable:
    - Click document creation uses reusable create_click_document() function
    - The tracking pattern (get user, create document, insert) is standard
    """
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    body = await request.json() if request.headers.get("content-type") == "application/json" else {}

    # Create click document using reusable function
    click_doc = create_click_document(
        user_id=user["email"],
        user_role=get_primary_role(user),
        url=body.get("url", "/"),
        element=body.get("element", "button"),
    )

    # Insert into database - automatically scoped to this app
    result = await db.clicks.insert_one(click_doc)

    return {"success": True, "click_id": str(result.inserted_id)}


@app.get("/clicks")
async def get_clicks(request: Request, limit: int = 50, db=Depends(get_scoped_db)):
    """
    Get clicks - filtered by role permissions.

    MDB-Engine specific:
    - Uses get_scoped_db() for scoped database access
    - User comes from request.state.user (SSO middleware)

    Reusable:
    - Permission checking uses reusable can_view_all() function
    - Query building logic is standard MongoDB
    - Serialization uses reusable serialize_clicks() function
    """
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    # Clickers only see their own clicks (reusable permission logic)
    query = {} if can_view_all(user) else {"user_id": user["email"]}

    # Fetch clicks - database is automatically scoped to this app
    clicks = await db.clicks.find(query).sort("timestamp", -1).limit(limit).to_list(limit)

    # Serialize clicks using reusable function
    serialized_clicks = serialize_clicks(clicks)

    return {
        "clicks": serialized_clicks,
        "count": len(serialized_clicks),
        "viewing_all": can_view_all(user),
    }


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Main page with demo user login.

    Reusable: Template rendering works with any FastAPI app.
    """
    user = get_current_user(request)
    role = get_primary_role(user) if user else None

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user": user,
            "role": role,
            "can_view_all": can_view_all(user) if user else False,
            "dashboard_url": get_related_app_url("dashboard"),
        },
    )


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
        "demo_users": [
            "alice@example.com (admin)",
            "bob@example.com (tracker)",
            "charlie@example.com (clicker)",
        ],
        "password": "password123",
        "sso_note": "Login here = logged into Dashboard too!",
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting uvicorn server on 0.0.0.0:8000")
    try:
        uvicorn.run(
            app, host="0.0.0.0", port=8000, log_level=os.getenv("LOG_LEVEL", "info").lower()
        )
    except Exception as e:
        logger.error(f"Failed to start uvicorn server: {e}", exc_info=True)
        sys.exit(1)
