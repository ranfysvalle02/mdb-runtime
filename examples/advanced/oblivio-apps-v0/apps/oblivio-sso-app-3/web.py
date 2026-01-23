#!/usr/bin/env python3
"""
Oblivio SSO App 3 - Admin Operations
=====================================

SSO-enabled app that demonstrates admin operations.
Requires admin role - validates tokens from auth hub via SSO.
"""

import logging
import os
import sys
from pathlib import Path

from fastapi import Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from mdb_engine.dependencies import get_scoped_db

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

try:
    from mdb_engine import MongoDBEngine
except ImportError as e:
    logger.error(f"Failed to import mdb_engine: {e}", exc_info=True)
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_SLUG = "oblivio-sso-app-3"

# Initialize MongoDB Engine
try:
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB", "oblivio_apps")
    logger.info(f"Initializing MongoDBEngine with URI: {mongo_uri[:50]}... (db: {db_name})")

    engine = MongoDBEngine(
        mongo_uri=mongo_uri,
        db_name=db_name,
    )
    logger.info("MongoDBEngine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MongoDBEngine: {e}", exc_info=True)
    sys.exit(1)

# Create FastAPI app
try:
    manifest_path = Path(__file__).parent / "manifest.json"
    logger.info(f"Creating FastAPI app with manifest: {manifest_path}")

    app = engine.create_app(
        slug=APP_SLUG,
        manifest=manifest_path,
        title="Oblivio SSO App 3",
        description="Admin operations SSO app",
        version="1.0.0",
    )
    logger.info("FastAPI app created successfully")
except Exception as e:
    logger.error(f"Failed to create FastAPI app: {e}", exc_info=True)
    sys.exit(1)

# Template engine
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_current_user(request: Request) -> dict | None:
    """Get user from request.state (populated by SharedAuthMiddleware)."""
    return getattr(request.state, "user", None)


def get_roles(user: dict | None) -> list:
    """Get user's roles for this app."""
    if not user:
        return []
    app_roles = user.get("app_roles", {})
    return app_roles.get(APP_SLUG, [])


def is_admin(user: dict | None) -> bool:
    """Check if user is admin."""
    roles = get_roles(user)
    return "admin" in roles


# ============================================================================
# ROUTES
# ============================================================================


@app.get("/login")
async def login_redirect(request: Request):
    """Redirect to auth hub login with redirect back to this app."""
    current_url = str(request.url)
    callback_url = f"{current_url.split('/login')[0]}/auth/callback"
    return RedirectResponse(
        url=f"http://localhost:8000/login?redirect_to={callback_url}", status_code=302
    )


@app.get("/auth/callback")
async def auth_callback(request: Request, token: str = None):
    """
    Token exchange endpoint - sets cookie for this app after auth hub login.
    Called with ?token=... after successful login at auth hub.
    """
    from urllib.parse import unquote_plus

    # Get token from query parameter (FastAPI auto-decodes, but handle URL-encoded tokens)
    if not token:
        token = request.query_params.get("token")

    if token:
        # URL-decode token in case it was encoded
        token = unquote_plus(token)

    if not token:
        return RedirectResponse(url="http://localhost:8000/login", status_code=302)

    # Validate token by getting user pool
    from mdb_engine.auth.shared_users import SharedUserPool

    pool: SharedUserPool = getattr(app.state, "user_pool", None)

    if not pool:
        return RedirectResponse(
            url="http://localhost:8000/login?error=pool_not_initialized", status_code=302
        )

    # Validate token
    user = await pool.validate_token(token)
    if not user:
        return RedirectResponse(
            url="http://localhost:8000/login?error=invalid_token", status_code=302
        )

    # Set cookie for this app
    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(
        key="mdb_auth_token",
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=86400,  # 24 hours
        path="/",
    )

    return response


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page - requires authentication."""
    # For shared auth, middleware sets request.state.user - check it directly
    user = get_current_user(request)
    if not user:
        # Redirect to auth hub if not authenticated
        return RedirectResponse(url="http://localhost:8000/login", status_code=302)

    if not is_admin(user):
        raise HTTPException(403, "Admin role required")

    roles = get_roles(user)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user": user,
            "roles": roles,
            "is_admin": True,
            "app_name": "Oblivio SSO App 3",
            "app_description": "Admin Operations",
        },
    )


@app.get("/api/admin/stats")
async def get_admin_stats(request: Request, db=Depends(get_scoped_db)):
    """Get admin statistics - requires admin role."""
    # For shared auth, middleware sets request.state.user - check it directly
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    if not is_admin(user):
        raise HTTPException(403, "Admin role required")

    # Get stats from all collections
    items_count = await db.items.count_documents({})
    users_count = (
        await db.users.count_documents({}) if "users" in await db.list_collection_names() else 0
    )

    return JSONResponse(
        {
            "success": True,
            "stats": {
                "items_count": items_count,
                "users_count": users_count,
                "app": APP_SLUG,
            },
            "admin": user.get("email"),
        }
    )


@app.delete("/api/admin/data/{item_id}")
async def delete_data(item_id: str, request: Request, db=Depends(get_scoped_db)):
    """Delete data - requires admin role."""
    # For shared auth, middleware sets request.state.user - check it directly
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    if not is_admin(user):
        raise HTTPException(403, "Admin role required")

    from bson.objectid import ObjectId

    result = await db.items.delete_one({"_id": ObjectId(item_id)})

    if result.deleted_count == 0:
        raise HTTPException(404, "Item not found")

    return JSONResponse(
        {
            "success": True,
            "message": "Item deleted successfully",
        }
    )


@app.get("/api/me")
async def get_me(request: Request):
    """Get current user info."""
    # For shared auth, middleware sets request.state.user - check it directly
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    return {
        "email": user["email"],
        "roles": get_roles(user),
        "is_admin": is_admin(user),
        "app": APP_SLUG,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "app": APP_SLUG, "auth": "shared"}


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Oblivio SSO App 3 on 0.0.0.0:8000")
    try:
        uvicorn.run(
            app, host="0.0.0.0", port=8000, log_level=os.getenv("LOG_LEVEL", "info").lower()
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)
