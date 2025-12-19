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

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket):
        """Accept and register a WebSocket connection"""
        await websocket.accept()
        async with self._lock:
            if websocket not in self.active_connections:
                self.active_connections.append(websocket)
        print(f"✅ WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        async def _disconnect():
            async with self._lock:
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
            print(f"🔌 WebSocket disconnected. Total connections: {len(self.active_connections)}")
        asyncio.create_task(_disconnect())
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients"""
        if not self.active_connections:
            return
        
        message_json = json.dumps(message)
        disconnected = []
        
        async with self._lock:
            connections = list(self.active_connections)
        
        for connection in connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                print(f"⚠️  Error sending to client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        if disconnected:
            async with self._lock:
                for conn in disconnected:
                    if conn in self.active_connections:
                        self.active_connections.remove(conn)

manager = ConnectionManager()

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
            if engine and len(manager.active_connections) > 0:
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
                
                # Broadcast metrics update
                await manager.broadcast({
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
            print(f"⚠️  Error in periodic metrics broadcast: {e}")
        
        await asyncio.sleep(5)  # Update every 5 seconds


@app.on_event("startup")
async def startup_event():
    # Log all registered routes for debugging
    print("=" * 50)
    print("📋 Registered Routes:")
    for route in app.routes:
        if hasattr(route, 'path'):
            route_type = "WebSocket" if hasattr(route, 'endpoint') and 'websocket' in str(type(route)).lower() else "HTTP"
            print(f"  {route_type}: {route.path}")
    print("=" * 50)
    """Initialize the runtime engine on startup"""
    global engine, db
    db = None
    
    print("🚀 Starting Hello World Web Application...")
    
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
    print("✅ Engine initialized successfully")
    
    # Load and register the app manifest
    manifest_path = Path("/app/manifest.json")
    if not manifest_path.exists():
        manifest_path = Path(__file__).parent / "manifest.json"
    
    if manifest_path.exists():
        manifest = await engine.load_manifest(manifest_path)
        success = await engine.register_app(manifest, create_indexes=True)
        if success:
            print(f"✅ App '{manifest['slug']}' registered successfully")
        else:
            print("⚠️  Failed to register app")
    
    # Get scoped database and store globally
    db = engine.get_scoped_db("hello_world")
    
    # Ensure demo user exists
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        client = AsyncIOMotorClient(mongo_uri)
        top_level_db = client[db_name]
        
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
            print(f"✅ Created demo user: {DEMO_EMAIL}")
        else:
            print(f"✅ Demo user already exists: {DEMO_EMAIL}")
        
        client.close()
    except Exception as e:
        print(f"⚠️  Could not create demo user: {e}")
    
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
            print(f"✅ Seeded {len(greetings)} initial greetings")
    except Exception as e:
        print(f"⚠️  Could not seed data: {e}")
    
    print("✅ Web application ready!")
    
    # Start background task for periodic metrics updates
    asyncio.create_task(periodic_metrics_broadcast())


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global engine
    if engine:
        await engine.shutdown()
        print("🧹 Cleaned up and shut down")


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
    try:
        mongo_uri = os.getenv("MONGO_URI", "mongodb://admin:password@mongodb:27017/?authSource=admin")
        db_name = os.getenv("MONGO_DB_NAME", "hello_world_db")
        
        from motor.motor_asyncio import AsyncIOMotorClient
        client = AsyncIOMotorClient(mongo_uri)
        top_level_db = client[db_name]
        
        # Find user by email
        user = await top_level_db.users.find_one({"email": email})
        
        if not user:
            client.close()
            return templates.TemplateResponse(
                "login.html",
                {"request": request, "error": "Invalid email or password"},
                status_code=status.HTTP_401_UNAUTHORIZED
            )
        
        # Verify password
        password_hash = user.get("password_hash")
        if not password_hash:
            client.close()
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
                client.close()
                return templates.TemplateResponse(
                    "login.html",
                    {"request": request, "error": "Invalid email or password"},
                    status_code=status.HTTP_401_UNAUTHORIZED
                )
        except Exception as e:
            client.close()
            return templates.TemplateResponse(
                "login.html",
                {"request": request, "error": "Invalid email or password"},
                status_code=status.HTTP_401_UNAUTHORIZED
            )
        
        client.close()
        
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
                        print(f"⚠️  Could not get stats for collection {base_name}: {e2}")
                        collections_info[base_name] = {
                            "count": 0,
                            "size": 0,
                            "storage_size": 0,
                            "indexes": len(index_list),
                            "index_size": 0,
                        }
            except Exception as e:
                print(f"⚠️  Error processing collection {base_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"✅ Found {len(collections_info)} collections: {list(collections_info.keys())}")
        print(f"✅ Found indexes for {len(indexes_info)} collections: {list(indexes_info.keys())}")
    except Exception as e:
        print(f"⚠️  Could not get indexes info: {e}")
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
    
    # Broadcast to all WebSocket clients
    await manager.broadcast({
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
    await manager.broadcast({
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
        
        # Broadcast to all WebSocket clients
        await manager.broadcast({
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
        # Broadcast to all WebSocket clients
        await manager.broadcast({
            "type": "greeting_deleted",
            "data": {
                "_id": greeting_id
            }
        })
        
        # Update metrics
        total_count = await db.greetings.count_documents({})
        await manager.broadcast({
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


@app.get("/ws-test")
async def websocket_test():
    """Test endpoint to verify WebSocket route is accessible"""
    return {"message": "WebSocket endpoint is accessible", "path": "/ws", "routes": [str(r.path) for r in app.routes if hasattr(r, 'path')]}

@app.get("/routes")
async def list_routes():
    """List all registered routes for debugging"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            routes.append({"path": route.path, "methods": list(route.methods), "type": "HTTP"})
        elif hasattr(route, 'path'):
            routes.append({"path": route.path, "type": "WebSocket"})
    return {"routes": routes}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    # Log immediately when endpoint is hit - this should print if route is reached
    import sys
    print("=" * 50, file=sys.stderr, flush=True)
    print("🔌 WebSocket endpoint hit - connection attempt", file=sys.stderr, flush=True)
    print("=" * 50, file=sys.stderr, flush=True)
    
    try:
        # Accept the WebSocket connection first - this must happen before anything else
        print("🔌 Attempting to accept WebSocket...", file=sys.stderr, flush=True)
        await websocket.accept()
        print("✅ WebSocket connection accepted", file=sys.stderr, flush=True)
        
        # Then add to manager
        async with manager._lock:
            if websocket not in manager.active_connections:
                manager.active_connections.append(websocket)
        print(f"✅ WebSocket added to manager. Total connections: {len(manager.active_connections)}")
        
        # Send initial connection confirmation
        try:
            await websocket.send_json({
                "type": "connected",
                "message": "WebSocket connected successfully"
            })
            print(f"✅ Sent connection confirmation")
        except Exception as send_error:
            print(f"⚠️  Error sending connection confirmation: {send_error}")
            raise
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages with a timeout
                message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                
                if message.get("type") == "websocket.disconnect":
                    print("🔌 WebSocket client disconnected (disconnect message)")
                    break
                elif message.get("type") == "websocket.receive":
                    # Handle text messages if needed
                    if "text" in message:
                        try:
                            data = json.loads(message["text"])
                            if data.get("type") == "pong":
                                print("🏓 Received pong from client")
                        except json.JSONDecodeError:
                            pass
                            
            except asyncio.TimeoutError:
                # Send a ping to keep connection alive
                try:
                    await websocket.send_json({
                        "type": "ping",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                except Exception as ping_error:
                    print(f"🔌 Connection dead (ping failed): {ping_error}")
                    break
                    
            except WebSocketDisconnect:
                print("🔌 WebSocket disconnected (WebSocketDisconnect exception)")
                break
            except Exception as e:
                error_msg = str(e).lower()
                error_type = type(e).__name__
                if any(keyword in error_msg for keyword in ["disconnect", "closed", "connection", "broken", "reset"]):
                    print(f"🔌 WebSocket disconnected: {error_type}: {e}")
                    break
                print(f"⚠️  WebSocket receive error (continuing): {error_type}: {e}")
                await asyncio.sleep(0.1)
                
    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected (outer WebSocketDisconnect)")
    except Exception as e:
        print(f"⚠️  WebSocket connection error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        # Try to send error to client if connection is still open
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass
    finally:
        # Remove from manager
        try:
            async with manager._lock:
                if websocket in manager.active_connections:
                    manager.active_connections.remove(websocket)
            print(f"🔌 WebSocket cleanup complete. Remaining connections: {len(manager.active_connections)}")
        except Exception as cleanup_error:
            print(f"⚠️  Error during cleanup: {cleanup_error}")


if __name__ == "__main__":
    import uvicorn
    # WebSocket support is enabled by default in uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, ws="auto")

