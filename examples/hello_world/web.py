#!/usr/bin/env python3
"""
FastAPI Web Application for Hello World Example

This demonstrates MDB_RUNTIME with a full web UI including:
- Authentication (login/logout)
- CRUD operations with visual feedback
- Health status dashboard
- Modern, responsive UI
"""
import os
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from bson.objectid import ObjectId

# Setup logger
logger = logging.getLogger(__name__)

from fastapi import FastAPI, Request, Depends, HTTPException, status, Form, Cookie, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any
import asyncio
import json

from mdb_runtime import RuntimeEngine
from mdb_runtime.auth.dependencies import get_current_user, get_authz_provider
import bcrypt
import jwt
from datetime import datetime, timedelta

# Initialize FastAPI app
app = FastAPI(
    title="Hello World - MDB_RUNTIME Demo",
    description="A beautiful demo showcasing MDB_RUNTIME capabilities",
    version="1.0.0"
)

# Add CORS middleware for HTTP requests (WebSockets handle CORS differently)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Note: WebSockets in FastAPI don't go through CORS middleware, they handle it at the protocol level

# Templates directory
templates = Jinja2Templates(directory="/app/templates")

# Global engine instance (will be initialized in startup)
engine: Optional[RuntimeEngine] = None
db = None

# WebSocket connection manager - now using runtime's WebSocket support
# The manager will be initialized after engine is ready
ws_manager = None

# Secret key for JWT (should match what's in dependencies)
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "a_very_bad_dev_secret_key_12345")

# Demo credentials
DEMO_EMAIL = "demo@demo.com"
DEMO_PASSWORD = "password123"


async def periodic_metrics_broadcast():
    """Periodically broadcast metrics updates to all WebSocket clients"""
    await asyncio.sleep(5)  # Wait for server to be ready
    
    while True:
        try:
            # Check if there are any WebSocket connections
            from mdb_runtime.routing.websockets import get_websocket_manager
            manager = await get_websocket_manager("hello_world")
            if engine and manager.get_connection_count() > 0:
                # Get health status
                health = await engine.get_health_status()
                engine_check = next((c for c in health['checks'] if c.get('name') == 'engine'), None)
                health_details = engine_check.get('details', {}) if engine_check else {}
                
                # Get pool metrics
                from mdb_runtime.database.connection import get_pool_metrics
                pool_metrics = await get_pool_metrics(engine._mongo_client)
                
                # Get database stats
                db_stats = {}
                try:
                    stats = await engine._mongo_db.command("dbStats")
                    db_stats = {
                        "collections": stats.get("collections", 0),
                        "data_size_mb": round(stats.get("dataSize", 0) / (1024 * 1024), 2),
                        "storage_size_mb": round(stats.get("storageSize", 0) / (1024 * 1024), 2),
                        "indexes": stats.get("indexes", 0),
                        "objects": stats.get("objects", 0),
                    }
                except Exception:
                    pass
                
                # Get server info
                server_info = {}
                try:
                    server_status = await engine._mongo_client.admin.command("serverStatus")
                    server_info = {
                        "version": server_status.get("version", "unknown"),
                        "uptime_hours": round(server_status.get("uptime", 0) / 3600, 2),
                        "connections": {
                            "current": server_status.get("connections", {}).get("current", 0),
                            "available": server_status.get("connections", {}).get("available", 0),
                            "total_created": server_status.get("connections", {}).get("totalCreated", 0),
                        },
                        "opcounters": {
                            "insert": server_status.get("opcounters", {}).get("insert", 0),
                            "query": server_status.get("opcounters", {}).get("query", 0),
                            "update": server_status.get("opcounters", {}).get("update", 0),
                            "delete": server_status.get("opcounters", {}).get("delete", 0),
                        }
                    }
                except Exception:
                    pass
                
                # Broadcast metrics update using runtime's convenience function
                await broadcast_to_app("hello_world", {
                    "type": "metrics_update",
                    "data": {
                        "health": {
                            "status": health.get("status"),
                            "initialized": health_details.get("initialized", False),
                            "app_count": health_details.get("app_count", 0)
                        },
                        "pool_metrics": pool_metrics,
                        "db_stats": db_stats,
                        "server_info": server_info
                    }
                })
        except Exception as e:
            logger.error(f"Error in periodic metrics broadcast: {e}")
        
        await asyncio.sleep(5)  # Update every 5 seconds


@app.on_event("startup")
async def startup_event():
    """Initialize the runtime engine on startup"""
    global engine, db
    db = None
    
    logger.info("Starting Hello World Web Application...")
    
    # Get MongoDB connection from environment
    mongo_uri = os.getenv(
        "MONGO_URI", 
        "mongodb://admin:password@mongodb:27017/?authSource=admin"
    )
    db_name = os.getenv("MONGO_DB_NAME", "hello_world_db")
    
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
    
    # Get scoped database and store globally
    db = engine.get_scoped_db("hello_world")
    
    # Register message handlers FIRST (before route registration)
    # This ensures handlers are available when routes are created
    register_websocket_message_handlers()
    
    # Register WebSocket routes from manifest (handlers are now available)
    register_websocket_routes_from_manifest()
    
    # Ensure demo user exists
    # Use engine.mongo_db instead of creating new client - leverages runtime's connection pooling
    try:
        # Use the runtime engine's database connection (already pooled and managed)
        top_level_db = engine.mongo_db
        
        # Check if demo user exists
        demo_user = await top_level_db.users.find_one({"email": DEMO_EMAIL})
        if not demo_user:
            # Create demo user
            password_hash = bcrypt.hashpw(DEMO_PASSWORD.encode("utf-8"), bcrypt.gensalt())
            user_doc = {
                "email": DEMO_EMAIL,
                "password_hash": password_hash,
                "full_name": "Demo User",
                "role": "user",
                "date_created": datetime.utcnow()
            }
            await top_level_db.users.insert_one(user_doc)
            logger.info(f"Created demo user: {DEMO_EMAIL}")
        else:
            logger.info(f"Demo user already exists: {DEMO_EMAIL}")
    except Exception as e:
        logger.warning(f"Could not create demo user: {e}")
    
    # Seed some initial data if collection is empty
    try:
        count = await db.greetings.count_documents({})
        if count == 0:
            greetings = [
                {"message": "Hello, World! 🌍", "language": "en", "created_at": datetime.utcnow()},
                {"message": "Hello, MDB_RUNTIME! 🚀", "language": "en", "created_at": datetime.utcnow()},
                {"message": "Hello, Python! 🐍", "language": "en", "created_at": datetime.utcnow()},
                {"message": "¡Hola, Mundo! 👋", "language": "es", "created_at": datetime.utcnow()},
                {"message": "Bonjour le Monde! 🇫🇷", "language": "fr", "created_at": datetime.utcnow()},
            ]
            for greeting in greetings:
                await db.greetings.insert_one(greeting)
            logger.info(f"Seeded {len(greetings)} initial greetings")
    except Exception as e:
        logger.warning(f"Could not seed data: {e}")
    
    logger.info("Web application ready!")
    
    # Start background task for periodic metrics updates
    asyncio.create_task(periodic_metrics_broadcast())


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
        return engine.get_scoped_db("hello_world")
    return db


# ============================================================================
# Authentication Routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root(request: Request, user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Home page - redirects to dashboard if logged in, otherwise to login"""
    if user:
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Login page"""
    if user:
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
):
    """Handle login"""
    if not engine:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Service not available. Please try again later."},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    
    try:
        # Use engine.mongo_db instead of creating new client - leverages runtime's connection pooling
        top_level_db = engine.mongo_db
        
        # Find user by email
        user = await top_level_db.users.find_one({"email": email})
        
        if not user:
            return templates.TemplateResponse(
                "login.html",
                {"request": request, "error": "Invalid email or password"},
                status_code=status.HTTP_401_UNAUTHORIZED
            )
        
        # Verify password
        password_hash = user.get("password_hash")
        if not password_hash:
            return templates.TemplateResponse(
                "login.html",
                {"request": request, "error": "Invalid email or password"},
                status_code=status.HTTP_401_UNAUTHORIZED
            )
        
        # Check password
        try:
            if isinstance(password_hash, str):
                password_hash = password_hash.encode("utf-8")
            if not bcrypt.checkpw(password.encode("utf-8"), password_hash):
                return templates.TemplateResponse(
                    "login.html",
                    {"request": request, "error": "Invalid email or password"},
                    status_code=status.HTTP_401_UNAUTHORIZED
                )
        except Exception as e:
            return templates.TemplateResponse(
                "login.html",
                {"request": request, "error": "Invalid email or password"},
                status_code=status.HTTP_401_UNAUTHORIZED
            )
        
        # Create JWT token
        payload = {
            "user_id": str(user["_id"]),
            "email": user["email"],
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        
        # Redirect to dashboard with cookie
        response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
        response.set_cookie(
            key="token",
            value=token,
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax",
            max_age=86400  # 24 hours
        )
        # Also set a non-httponly cookie for WebSocket access (same domain, so relatively safe)
        # This allows JavaScript to read it for WebSocket authentication
        response.set_cookie(
            key="ws_token",
            value=token,
            httponly=False,  # Allow JS access for WebSocket
            secure=False,
            samesite="lax",
            max_age=86400
        )
        return response
        
    except Exception as e:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": f"Login failed: {str(e)}"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@app.get("/logout")
async def logout():
    """Handle logout"""
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="token")
    response.delete_cookie(key="ws_token")
    return response


# ============================================================================
# Protected Routes
# ============================================================================

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Dashboard page - requires authentication"""
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    db = get_db()
    
    # Get greetings
    greetings = await db.greetings.find({}).sort("created_at", -1).to_list(length=100)
    
    # Get health status
    health = await engine.get_health_status()
    engine_check = next((c for c in health['checks'] if c.get('name') == 'engine'), None)
    health_details = engine_check.get('details', {}) if engine_check else {}
    
    # Get MongoDB metrics
    from mdb_runtime.database.connection import get_pool_metrics
    pool_metrics = await get_pool_metrics(engine._mongo_client)
    if pool_metrics.get("status") == "connected":
        # Add RuntimeEngine's pool config if missing
        if "max_pool_size" not in pool_metrics or pool_metrics.get("max_pool_size") is None:
            pool_metrics["max_pool_size"] = engine.max_pool_size
        if "min_pool_size" not in pool_metrics or pool_metrics.get("min_pool_size") is None:
            pool_metrics["min_pool_size"] = engine.min_pool_size
    
    # Get database stats
    db_stats = {}
    server_info = {}
    try:
        stats = await engine._mongo_db.command("dbStats")
        db_stats = {
            "collections": stats.get("collections", 0),
            "data_size_mb": round(stats.get("dataSize", 0) / (1024 * 1024), 2),
            "storage_size_mb": round(stats.get("storageSize", 0) / (1024 * 1024), 2),
            "indexes": stats.get("indexes", 0),
            "objects": stats.get("objects", 0),
        }
    except Exception:
        pass
    
    # Get server info
    try:
        server_status = await engine._mongo_client.admin.command("serverStatus")
        server_info = {
            "version": server_status.get("version", "unknown"),
            "uptime_hours": round(server_status.get("uptime", 0) / 3600, 2),
            "connections": {
                "current": server_status.get("connections", {}).get("current", 0),
                "available": server_status.get("connections", {}).get("available", 0),
                "total_created": server_status.get("connections", {}).get("totalCreated", 0),
            },
            "network": {
                "bytes_in_mb": round(server_status.get("network", {}).get("bytesIn", 0) / (1024 * 1024), 2),
                "bytes_out_mb": round(server_status.get("network", {}).get("bytesOut", 0) / (1024 * 1024), 2),
                "num_requests": server_status.get("network", {}).get("numRequests", 0),
            },
            "opcounters": {
                "insert": server_status.get("opcounters", {}).get("insert", 0),
                "query": server_status.get("opcounters", {}).get("query", 0),
                "update": server_status.get("opcounters", {}).get("update", 0),
                "delete": server_status.get("opcounters", {}).get("delete", 0),
            }
        }
    except Exception:
        pass
    
    # Get stats
    total_greetings = await db.greetings.count_documents({})
    # Get distinct languages by extracting from all greetings
    all_greetings_for_stats = await db.greetings.find({}, {"language": 1}).to_list(length=1000)
    languages = list(set(g.get("language", "unknown") for g in all_greetings_for_stats))
    
    # Get indexes information
    indexes_info = {}
    collections_info = {}
    try:
        # Get the underlying database from the scoped wrapper
        # Collections are prefixed with app_slug (e.g., "hello_world_greetings")
        underlying_db = db.database if hasattr(db, 'database') else engine._mongo_db
        
        # Get app slug from the scoped wrapper (it's stored in _write_scope)
        app_slug = getattr(db, '_write_scope', 'hello_world') if hasattr(db, '_write_scope') else 'hello_world'
        
        # Get all collections from underlying database
        all_collections = await underlying_db.list_collection_names()
        
        # Filter to only collections for this app (prefixed with app_slug)
        app_collections = [c for c in all_collections if c.startswith(f"{app_slug}_")]
        
        # If no prefixed collections found, try without prefix (for backwards compatibility)
        if not app_collections:
            # Try known collection names
            known_collections = ['greetings']
            for known_coll in known_collections:
                prefixed = f"{app_slug}_{known_coll}"
                if prefixed in all_collections:
                    app_collections.append(prefixed)
        
        # Also get the base collection names (without prefix) for display
        base_collection_map = {}
        for prefixed_name in app_collections:
            # Remove the prefix to get the base name
            if prefixed_name.startswith(f"{app_slug}_"):
                base_name = prefixed_name[len(f"{app_slug}_"):]
            else:
                base_name = prefixed_name
            base_collection_map[base_name] = prefixed_name
        
        # Process each collection
        for base_name, prefixed_name in base_collection_map.items():
            try:
                # Get the scoped collection (using base name)
                coll = getattr(db, base_name)
                
                # Get the real collection for index/stats operations
                real_coll = getattr(underlying_db, prefixed_name)
                
                # Get index information from the real collection
                index_list = await real_coll.index_information()
                indexes_info[base_name] = []
                for index_name, index_spec in index_list.items():
                    index_details = {
                        "name": index_name,
                        "keys": index_spec.get("key", []),
                        "unique": index_spec.get("unique", False),
                        "sparse": index_spec.get("sparse", False),
                        "background": index_spec.get("background", False),
                        "v": index_spec.get("v", 1),
                    }
                    indexes_info[base_name].append(index_details)
                
                # Get collection stats from the real collection
                # Note: collStats is deprecated in MongoDB 7.0+, but still works
                # We'll use it for now and can migrate to $collStats aggregation later if needed
                try:
                    coll_stats = await underlying_db.command({"collStats": prefixed_name})
                    collections_info[base_name] = {
                        "count": coll_stats.get("count", 0),
                        "size": round(coll_stats.get("size", 0) / 1024, 2),  # KB
                        "storage_size": round(coll_stats.get("storageSize", 0) / 1024, 2),  # KB
                        "indexes": coll_stats.get("nindexes", 0),
                        "index_size": round(coll_stats.get("totalIndexSize", 0) / 1024, 2),  # KB
                    }
                except Exception as e:
                    # Fallback to count only using scoped collection
                    try:
                        count = await coll.count_documents({})
                        collections_info[base_name] = {
                            "count": count,
                            "size": 0,
                            "storage_size": 0,
                            "indexes": len(index_list),
                            "index_size": 0,
                        }
                    except Exception as e2:
                        logger.warning(f"Could not get stats for collection {base_name}: {e2}")
                        collections_info[base_name] = {
                            "count": 0,
                            "size": 0,
                            "storage_size": 0,
                            "indexes": len(index_list),
                            "index_size": 0,
                        }
            except Exception as e:
                logger.warning(f"Error processing collection {base_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        logger.debug(f"Found {len(collections_info)} collections: {list(collections_info.keys())}")
        logger.debug(f"Found indexes for {len(indexes_info)} collections: {list(indexes_info.keys())}")
    except Exception as e:
        logger.warning(f"Could not get indexes info: {e}")
        import traceback
        traceback.print_exc()
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
        "greetings": greetings,
        "health": health,
        "health_details": health_details,
        "pool_metrics": pool_metrics,
        "db_stats": db_stats,
        "server_info": server_info,
        "indexes_info": indexes_info,
        "collections_info": collections_info,
        "total_greetings": total_greetings,
        "languages": languages,
    })


@app.post("/api/greetings", response_class=JSONResponse)
async def create_greeting(
    request: Request,
    message: str = Form(...),
    language: str = Form("en"),
    user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Create a new greeting"""
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    
    db = get_db()
    
    greeting = {
        "message": message,
        "language": language,
        "created_at": datetime.utcnow(),
        "created_by": user.get("email", "unknown")
    }
    
    result = await db.greetings.insert_one(greeting)
    
    # Fetch the created greeting for broadcast
    created_greeting = await db.greetings.find_one({"_id": result.inserted_id})
    
    # Broadcast to all WebSocket clients using runtime's convenience function
    # This automatically scopes the message to this app
    await broadcast_to_app("hello_world", {
        "type": "greeting_created",
        "data": {
            "_id": str(created_greeting["_id"]),
            "message": created_greeting["message"],
            "language": created_greeting["language"],
            "created_at": created_greeting["created_at"].isoformat() if created_greeting.get("created_at") else None,
            "created_by": created_greeting.get("created_by", "unknown")
        }
    })
    
    # Also update metrics
    total_count = await db.greetings.count_documents({})
    await broadcast_to_app("hello_world", {
        "type": "metrics_updated",
        "data": {
            "total_greetings": total_count
        }
    })
    
    return {
        "success": True,
        "id": str(result.inserted_id),
        "message": "Greeting created successfully"
    }


@app.put("/api/greetings/{greeting_id}", response_class=JSONResponse)
async def update_greeting(
    greeting_id: str,
    message: str = Form(...),
    language: str = Form("en"),
    user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Update a greeting"""
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    
    db = get_db()
    result = await db.greetings.update_one(
        {"_id": ObjectId(greeting_id)},
        {"$set": {
            "message": message,
            "language": language,
            "updated_at": datetime.utcnow(),
            "updated_by": user.get("email", "unknown")
        }}
    )
    
    if result.matched_count > 0:
        # Fetch updated greeting for broadcast
        updated_greeting = await db.greetings.find_one({"_id": ObjectId(greeting_id)})
        
        # Broadcast to all WebSocket clients using runtime's manager
        await broadcast_to_app("hello_world", {
            "type": "greeting_updated",
            "data": {
                "_id": str(updated_greeting["_id"]),
                "message": updated_greeting["message"],
                "language": updated_greeting["language"],
                "created_at": updated_greeting["created_at"].isoformat() if updated_greeting.get("created_at") else None,
                "updated_at": updated_greeting["updated_at"].isoformat() if updated_greeting.get("updated_at") else None,
            }
        })
        
        return {"success": True, "message": "Greeting updated successfully"}
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Greeting not found")


@app.delete("/api/greetings/{greeting_id}", response_class=JSONResponse)
async def delete_greeting(
    greeting_id: str,
    user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Delete a greeting"""
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    
    db = get_db()
    result = await db.greetings.delete_one({"_id": ObjectId(greeting_id)})
    
    if result.deleted_count > 0:
        # Broadcast to all WebSocket clients using runtime's manager
        await broadcast_to_app("hello_world", {
            "type": "greeting_deleted",
            "data": {
                "_id": greeting_id
            }
        })
        
        # Update metrics
        total_count = await db.greetings.count_documents({})
        await broadcast_to_app("hello_world", {
            "type": "metrics_updated",
            "data": {
                "total_greetings": total_count
            }
        })
        
        return {"success": True, "message": "Greeting deleted successfully"}
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Greeting not found")


@app.get("/api/health", response_class=JSONResponse)
async def health_api(user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Health status API endpoint"""
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    
    health = await engine.get_health_status()
    return health


@app.get("/api/metrics", response_class=JSONResponse)
async def metrics_api(user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """MongoDB metrics API endpoint"""
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    
    if not engine:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not initialized")
    
    metrics = {}
    
    try:
        # Get pool metrics
        from mdb_runtime.database.connection import get_pool_metrics
        pool_metrics = await get_pool_metrics(engine._mongo_client)
        if pool_metrics.get("status") == "connected":
            # Add RuntimeEngine's pool config if missing
            if "max_pool_size" not in pool_metrics or pool_metrics.get("max_pool_size") is None:
                pool_metrics["max_pool_size"] = engine.max_pool_size
            if "min_pool_size" not in pool_metrics or pool_metrics.get("min_pool_size") is None:
                pool_metrics["min_pool_size"] = engine.min_pool_size
        metrics["pool"] = pool_metrics
        
        # Get database stats
        db = engine._mongo_db
        if db is not None:
            try:
                db_stats = await db.command("dbStats")
                metrics["database"] = {
                    "name": db_stats.get("db", engine.db_name),
                    "collections": db_stats.get("collections", 0),
                    "data_size": db_stats.get("dataSize", 0),
                    "storage_size": db_stats.get("storageSize", 0),
                    "indexes": db_stats.get("indexes", 0),
                    "index_size": db_stats.get("indexSize", 0),
                    "objects": db_stats.get("objects", 0),
                }
            except Exception as e:
                logger.debug(f"Could not get database stats: {e}")
                metrics["database"] = {"error": str(e)}
            
            # Get server status
            try:
                server_status = await engine._mongo_client.admin.command("serverStatus")
                metrics["server"] = {
                    "version": server_status.get("version", "unknown"),
                    "uptime": server_status.get("uptime", 0),
                    "connections": {
                        "current": server_status.get("connections", {}).get("current", 0),
                        "available": server_status.get("connections", {}).get("available", 0),
                        "total_created": server_status.get("connections", {}).get("totalCreated", 0),
                    },
                    "network": {
                        "bytes_in": server_status.get("network", {}).get("bytesIn", 0),
                        "bytes_out": server_status.get("network", {}).get("bytesOut", 0),
                        "num_requests": server_status.get("network", {}).get("numRequests", 0),
                    },
                }
            except Exception as e:
                logger.debug(f"Could not get server status: {e}")
                metrics["server"] = {"error": str(e)}
        
        # Get app-specific metrics
        db = get_db()
        try:
            collections = await db.list_collection_names()
            collection_stats = {}
            for coll_name in collections:
                try:
                    coll = getattr(db, coll_name)
                    count = await coll.count_documents({})
                    indexes = await coll.index_information()
                    collection_stats[coll_name] = {
                        "count": count,
                        "indexes": len(indexes),
                    }
                except Exception:
                    pass
            metrics["collections"] = collection_stats
        except Exception as e:
            logger.debug(f"Could not get collection stats: {e}")
            metrics["collections"] = {"error": str(e)}
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}", exc_info=True)
        metrics["error"] = str(e)
    
    return metrics



# Register WebSocket routes from manifest
# The runtime engine will handle WebSocket registration automatically
# For now, we'll use the runtime's WebSocket support
from mdb_runtime.routing.websockets import (
    get_websocket_manager_sync, 
    broadcast_to_app,
    register_message_handler
)

# Use runtime's WebSocket support - register routes from manifest after engine is initialized
# This will be called in the startup event after engine.register_app()
def register_websocket_routes_from_manifest():
    """Register WebSocket routes from manifest configuration."""
    global ws_manager
    if engine:
        logger.info("Registering WebSocket routes for app 'hello_world'...")
        try:
            # Check if WebSocket config exists
            ws_config = engine.get_websocket_config("hello_world")
            if not ws_config:
                logger.warning("No WebSocket configuration found in manifest for 'hello_world'")
                return
            
            # Register routes automatically from manifest
            engine.register_websocket_routes(app, "hello_world")
            logger.info("WebSocket routes registered successfully")
            
            # Get the WebSocket manager for this app (sync version for startup)
            ws_manager = get_websocket_manager_sync("hello_world")
            logger.info("WebSocket manager initialized")
        except Exception as e:
            logger.error(f"Error registering WebSocket routes: {e}", exc_info=True)
    else:
        logger.warning("Engine not initialized, cannot register WebSocket routes")


# Global counter for demo purposes
_websocket_demo_counter = 0


def register_websocket_message_handlers():
    """Register message handlers for bi-directional WebSocket communication demo."""
    
    async def handle_realtime_messages(websocket, message):
        """Handle incoming messages from WebSocket clients."""
        global _websocket_demo_counter
        
        msg_type = message.get("type")
        logger.info(f"📨 Received WebSocket message: {msg_type} for app 'hello_world'")
        
        if msg_type == "echo_test":
            # Echo test - respond directly to the client
            from mdb_runtime.routing.websockets import get_websocket_manager
            manager = await get_websocket_manager("hello_world")
            await manager.send_to_connection(websocket, {
                "type": "echo_response",
                "original_message": message.get("message", ""),
                "timestamp": datetime.utcnow().isoformat(),
                "server_message": f"Echo: {message.get('message', 'Hello!')}"
            })
        
        elif msg_type == "request_server_time":
            # Request server time - broadcast to all clients
            await broadcast_to_app("hello_world", {
                "type": "server_time_response",
                "server_time": datetime.utcnow().isoformat(),
                "requested_by": message.get("user_id", "unknown")
            })
        
        elif msg_type == "increment_counter":
            # Increment global counter and broadcast to all clients
            _websocket_demo_counter += 1
            await broadcast_to_app("hello_world", {
                "type": "counter_updated",
                "counter": _websocket_demo_counter,
                "incremented_by": message.get("user_id", "unknown"),
                "timestamp": datetime.utcnow().isoformat()
            })
        
        elif msg_type == "reset_counter":
            # Reset counter and broadcast
            _websocket_demo_counter = 0
            await broadcast_to_app("hello_world", {
                "type": "counter_reset",
                "counter": 0,
                "reset_by": message.get("user_id", "unknown"),
                "timestamp": datetime.utcnow().isoformat()
            })
        
        elif msg_type == "request_greetings_count":
            # Request current greetings count - respond with broadcast
            global db
            if db is not None:
                count = await db.greetings.count_documents({})
                await broadcast_to_app("hello_world", {
                    "type": "greetings_count_response",
                    "count": count,
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        elif msg_type == "ping":
            # Custom ping (different from keepalive ping)
            from mdb_runtime.routing.websockets import get_websocket_manager
            manager = await get_websocket_manager("hello_world")
            await manager.send_to_connection(websocket, {
                "type": "pong_response",
                "message": "Pong! Server is alive and responding.",
                "timestamp": datetime.utcnow().isoformat()
            })
    
    # Register handler for the "realtime" endpoint
    register_message_handler("hello_world", "realtime", handle_realtime_messages)
    logger.info("Registered WebSocket message handlers for bi-directional communication")

# Note: The WebSocket endpoint is automatically registered by the runtime from manifest.json
# No need for a custom @app.websocket("/ws") endpoint - the runtime handles it via register_websocket_routes()



if __name__ == "__main__":
    import uvicorn
    # WebSocket support is enabled by default in uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, ws="auto")

