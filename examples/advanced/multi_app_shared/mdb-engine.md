# MDB-Engine Components in Multi-App Shared (SSO) Example

This document isolates the MDB-Engine specific parts of the Multi-App Shared example, explains their value, and shows how to implement your own versions.

## Overview

The Multi-App Shared example demonstrates Single Sign-On (SSO) features:
- **SharedUserPool** - Centralized user management across apps
- **SharedAuthMiddleware** - Automatic SSO middleware
- **Per-app role management** - Users have different roles per app
- **JWT token sharing** - One login works across all apps
- **Cross-app data access** - Apps can read each other's data

**Without MDB-Engine**, you would need to implement:
- All features from Multi-App, plus:
- Shared user pool database schema
- JWT token generation and validation
- SSO middleware for FastAPI
- Per-app role storage and retrieval
- Token revocation and refresh
- Cookie management for SSO
- User authentication across multiple apps

## MDB-Engine Components

### 1. Basic Components (from Simple App and Multi-App)

All components from previous examples are also used here:
- `MongoDBEngine()` - Database connection management
- `engine.create_app()` - FastAPI app creation
- `get_scoped_db()` - Scoped database access
- `get_authz_provider()` - Casbin authorization (if used)
- Cross-app data access

See [simple_app/mdb-engine.md](../simple_app/mdb-engine.md) and [multi_app/mdb-engine.md](../multi_app/mdb-engine.md) for details.

---

### 2. Shared Authentication Mode

**What it is:**

In `manifest.json`:

```json
{
  "auth": {
    "mode": "shared",
    "roles": ["clicker", "tracker", "admin"],
    "public_routes": ["/", "/health", "/api", "/login"],
    "users": {
      "enabled": true,
      "strategy": "app_users",
      "demo_users": [
        {
          "email": "alice@example.com",
          "password": "password123",
          "app_roles": {
            "click_tracker": ["admin"],
            "dashboard": ["admin"]
          }
        }
      ]
    }
  }
}
```

**What it does:**

- Detects `auth.mode="shared"` in manifest
- Initializes `SharedUserPool` automatically
- Creates `_mdb_engine_shared_users` collection
- Seeds demo users with per-app roles
- Sets up shared authentication infrastructure

**Value:**

- **Declarative**: SSO configured via manifest, not code
- **Automatic setup**: User pool created and configured automatically
- **Demo users**: Test users seeded automatically
- **Per-app roles**: Users can have different roles per app

**How to implement your own:**

```python
async def initialize_shared_auth(db, manifest):
    """Initialize shared authentication from manifest."""
    auth_config = manifest.get("auth", {})
    
    if auth_config.get("mode") != "shared":
        return None
    
    # Create shared users collection
    shared_users_collection = db["_shared_users"]
    
    # Seed demo users
    demo_users = auth_config.get("users", {}).get("demo_users", [])
    for user_data in demo_users:
        email = user_data["email"]
        password = user_data["password"]
        app_roles = user_data.get("app_roles", {})
        
        # Check if user exists
        existing = await shared_users_collection.find_one({"email": email})
        if existing:
            continue
        
        # Hash password
        import bcrypt
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        
        # Create user document
        user_doc = {
            "email": email,
            "password_hash": hashed.decode(),
            "app_roles": app_roles,
            "created_at": datetime.utcnow(),
            "active": True
        }
        
        await shared_users_collection.insert_one(user_doc)
    
    return shared_users_collection
```

---

### 3. SharedUserPool

**What it is:**

```python
def get_user_pool():
    """Get the shared user pool from app state."""
    return getattr(app.state, "user_pool", None)

# Usage
pool = get_user_pool()
token = await pool.authenticate(email, password)
user = await pool.validate_token(token)
```

**What it does:**

- Manages users in `_mdb_engine_shared_users` collection
- Provides `authenticate(email, password)` - Returns JWT token
- Provides `validate_token(token)` - Returns user document
- Provides `revoke_token(token)` - Invalidates token
- Provides `update_user_roles(email, app_slug, roles)` - Updates roles
- Handles password hashing and verification
- Manages JWT token generation and validation

**Value:**

- **Centralized users**: One user pool for all apps
- **JWT tokens**: Secure, stateless authentication
- **Token management**: Automatic token generation and validation
- **Role updates**: Easy per-app role management
- **Password security**: Built-in bcrypt hashing

**How to implement your own:**

```python
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, List

class SharedUserPool:
    """Centralized user pool for SSO."""
    
    def __init__(self, db, jwt_secret: str, jwt_algorithm: str = "HS256"):
        self.db = db
        self.users_collection = db["_shared_users"]
        self.tokens_collection = db["_shared_tokens"]
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
    
    async def authenticate(self, email: str, password: str) -> Optional[str]:
        """Authenticate user and return JWT token."""
        # Find user
        user = await self.users_collection.find_one({"email": email})
        if not user or not user.get("active", True):
            return None
        
        # Verify password
        password_hash = user["password_hash"]
        if not bcrypt.checkpw(password.encode(), password_hash.encode()):
            return None
        
        # Generate JWT token
        token = self._generate_token(user)
        
        # Store token (for revocation)
        await self.tokens_collection.insert_one({
            "token": token,
            "user_email": email,
            "created_at": datetime.utcnow(),
            "revoked": False
        })
        
        return token
    
    def _generate_token(self, user: dict) -> str:
        """Generate JWT token for user."""
        payload = {
            "email": user["email"],
            "app_roles": user.get("app_roles", {}),
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(days=1)
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    async def validate_token(self, token: str) -> Optional[dict]:
        """Validate JWT token and return user."""
        try:
            # Check if token is revoked
            token_doc = await self.tokens_collection.find_one({"token": token})
            if token_doc and token_doc.get("revoked", False):
                return None
            
            # Decode and validate token
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Get user from database
            user = await self.users_collection.find_one({"email": payload["email"]})
            if not user or not user.get("active", True):
                return None
            
            return user
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    async def revoke_token(self, token: str):
        """Revoke a JWT token."""
        await self.tokens_collection.update_one(
            {"token": token},
            {"$set": {"revoked": True, "revoked_at": datetime.utcnow()}}
        )
    
    async def update_user_roles(self, email: str, app_slug: str, roles: List[str]) -> bool:
        """Update user's roles for a specific app."""
        result = await self.users_collection.update_one(
            {"email": email},
            {"$set": {f"app_roles.{app_slug}": roles}}
        )
        return result.modified_count > 0

# Usage
jwt_secret = os.getenv("MDB_ENGINE_JWT_SECRET", "your-secret-key")
user_pool = SharedUserPool(db_manager.get_db(), jwt_secret)
app.state.user_pool = user_pool
```

**Comparison:**

| Feature | MDB-Engine | Custom Implementation |
|---------|-----------|----------------------|
| Lines of code | 1 function | ~120+ |
| User management | Automatic | Manual CRUD operations |
| JWT generation | Built-in | Manual token creation |
| Token validation | Built-in | Manual decoding/verification |
| Token revocation | Built-in | Manual revocation tracking |
| Role updates | Built-in | Manual update operations |
| Password hashing | Built-in | Manual bcrypt usage |

---

### 4. SharedAuthMiddleware

**What it is:**

MDB-Engine automatically adds `SharedAuthMiddleware` when `auth.mode="shared"` is detected. You don't see it in your code - it's added automatically.

**What it does:**

- Intercepts every HTTP request
- Extracts JWT token from `mdb_auth_token` cookie
- Validates token using SharedUserPool
- Populates `request.state.user` with user document
- Handles token expiration and errors gracefully
- Allows public routes (from manifest) to bypass auth

**Value:**

- **Automatic**: No middleware setup code needed
- **Transparent**: User available in `request.state.user`
- **Secure**: Token validation on every request
- **Flexible**: Public routes configured in manifest
- **Error handling**: Graceful handling of expired/invalid tokens

**How to implement your own:**

```python
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

class SharedAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for shared authentication (SSO)."""
    
    def __init__(self, app, user_pool: SharedUserPool, public_routes: List[str]):
        super().__init__(app)
        self.user_pool = user_pool
        self.public_routes = public_routes
    
    async def dispatch(self, request: Request, call_next):
        """Process request and add user to request.state."""
        # Check if route is public
        if request.url.path in self.public_routes:
            return await call_next(request)
        
        # Get token from cookie
        token = request.cookies.get("mdb_auth_token")
        
        if token:
            # Validate token
            user = await self.user_pool.validate_token(token)
            if user:
                # Add user to request state
                request.state.user = user
            else:
                # Invalid token - clear cookie
                request.state.user = None
        else:
            request.state.user = None
        
        # Continue request processing
        response = await call_next(request)
        
        # Clear user from state after response
        if hasattr(request.state, "user"):
            delattr(request.state, "user")
        
        return response

# Usage
public_routes = manifest.get("auth", {}).get("public_routes", [])
app.add_middleware(
    SharedAuthMiddleware,
    user_pool=user_pool,
    public_routes=public_routes
)
```

**Comparison:**

| Feature | MDB-Engine | Custom Implementation |
|---------|-----------|----------------------|
| Lines of code | 0 (automatic) | ~50+ |
| Middleware setup | Automatic | Manual registration |
| Token extraction | Automatic | Manual cookie reading |
| User population | Automatic | Manual state setting |
| Public routes | From manifest | Manual configuration |
| Error handling | Built-in | Manual exception handling |

---

### 5. Per-App Role Management

**What it is:**

User document structure:

```python
{
    "email": "alice@example.com",
    "password_hash": "...",
    "app_roles": {
        "click_tracker": ["admin"],
        "dashboard": ["admin"]
    }
}
```

**What it does:**

- Stores roles per app in `app_roles` dictionary
- Allows users to have different roles in different apps
- Enables role-based access control per app
- Supports role updates without affecting other apps

**Value:**

- **Flexible**: Different permissions per app
- **Isolated**: Role changes don't affect other apps
- **Scalable**: Easy to add new apps with new roles
- **Clear**: Role structure is explicit and readable

**How to implement your own:**

```python
def get_roles(user: Optional[dict], app_slug: str) -> List[str]:
    """Get user's roles for a specific app."""
    if not user:
        return []
    app_roles = user.get("app_roles", {})
    return app_roles.get(app_slug, [])

def get_primary_role(user: Optional[dict], app_slug: str) -> str:
    """Get user's primary role for an app."""
    roles = get_roles(user, app_slug)
    # Priority: admin > tracker > clicker
    if "admin" in roles:
        return "admin"
    if "tracker" in roles:
        return "tracker"
    return "clicker"

def can_perform_action(user: Optional[dict], app_slug: str, action: str) -> bool:
    """Check if user can perform action in app."""
    roles = get_roles(user, app_slug)
    
    # Define role permissions
    permissions = {
        "admin": ["read", "write", "delete", "manage"],
        "tracker": ["read", "write"],
        "clicker": ["read"]
    }
    
    for role in roles:
        if action in permissions.get(role, []):
            return True
    
    return False

# Usage
user = get_current_user(request)
if not can_perform_action(user, APP_SLUG, "write"):
    raise HTTPException(status_code=403, detail="Permission denied")
```

**Comparison:**

| Feature | MDB-Engine | Custom Implementation |
|---------|-----------|----------------------|
| Role storage | Automatic | Manual dictionary structure |
| Role retrieval | Built-in helpers | Manual extraction |
| Role updates | Built-in method | Manual update operations |
| Permission checks | Built-in | Manual permission logic |

---

### 6. SSO Cookie Management

**What it is:**

```python
AUTH_COOKIE = "mdb_auth_token"

# Set cookie after login
response.set_cookie(
    key=AUTH_COOKIE,
    value=token,
    httponly=True,
    samesite="lax",
    max_age=86400,  # 24 hours
    path="/",
)

# Clear cookie on logout
response.delete_cookie(AUTH_COOKIE, path="/")
```

**What it does:**

- Sets shared cookie name across all apps
- Cookie works across all apps on same domain
- HttpOnly flag prevents XSS attacks
- SameSite flag prevents CSRF attacks
- Path="/" makes cookie available to all routes

**Value:**

- **SSO**: One login works across all apps
- **Security**: HttpOnly and SameSite flags protect against attacks
- **Convenience**: Users don't need to login to each app separately

**How to implement your own:**

```python
# Standard FastAPI cookie handling - no custom code needed
# Just use consistent cookie name across apps

AUTH_COOKIE = "mdb_auth_token"  # Must match across all apps

@app.post("/login")
async def login(...):
    token = await user_pool.authenticate(email, password)
    
    response = JSONResponse(...)
    response.set_cookie(
        key=AUTH_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=86400,
        path="/",
    )
    return response

@app.post("/logout")
async def logout(request: Request):
    token = request.cookies.get(AUTH_COOKIE)
    if token:
        await user_pool.revoke_token(token)
    
    response = JSONResponse(...)
    response.delete_cookie(AUTH_COOKIE, path="/")
    return response
```

**Comparison:**

| Feature | MDB-Engine | Custom Implementation |
|---------|-----------|----------------------|
| Cookie name | Configurable | Manual constant |
| Cookie setting | Manual (in login) | Manual (in login) |
| Cookie clearing | Manual (in logout) | Manual (in logout) |
| Security flags | Manual | Manual |
| Cross-app sharing | Automatic | Works if same domain |

---

## Complete Replacement Guide

### Step 1: Add Shared Authentication

**Remove:**
```python
# MDB-Engine handles this automatically when auth.mode="shared"
```

**Replace with:**
```python
# Add SharedUserPool class (from section 3)
# Initialize in lifespan handler
user_pool = SharedUserPool(db, jwt_secret)
app.state.user_pool = user_pool
```

### Step 2: Add SharedAuthMiddleware

**Remove:**
```python
# MDB-Engine adds this automatically
```

**Replace with:**
```python
# Add SharedAuthMiddleware class (from section 4)
# Register middleware
public_routes = manifest.get("auth", {}).get("public_routes", [])
app.add_middleware(
    SharedAuthMiddleware,
    user_pool=user_pool,
    public_routes=public_routes
)
```

### Step 3: Update User Access

**MDB-Engine (automatic):**
```python
def get_current_user(request: Request):
    return getattr(request.state, "user", None)
```

**Custom (same pattern):**
```python
# Same pattern - middleware populates request.state.user
def get_current_user(request: Request):
    return getattr(request.state, "user", None)
```

### Step 4: Add Per-App Role Helpers

**Add:**
```python
# Add get_roles(), get_primary_role(), can_perform_action() helpers
# (from section 5)
```

---

## Migration Guide

### Migrating FROM MDB-Engine TO Custom Implementation

1. **Add SharedUserPool** (from section 3)
2. **Add SharedAuthMiddleware** (from section 4)
3. **Add role helper functions** (from section 5)
4. **Update login/logout** to use SharedUserPool
5. **Initialize user pool** in lifespan handler
6. **Register middleware** in app setup

### Migrating FROM Custom Implementation TO MDB-Engine

1. **Remove SharedUserPool class** - MDB-Engine provides it
2. **Remove SharedAuthMiddleware** - MDB-Engine adds it automatically
3. **Update manifest** - Add `auth.mode="shared"`
4. **Simplify login/logout** - Use `pool.authenticate()` directly
5. **Remove middleware registration** - Automatic in MDB-Engine

---

## Architecture Comparison

### MDB-Engine SSO Architecture

```
┌─────────────────────────────────────────┐
│         FastAPI App (Click Tracker)     │
│  ┌───────────────────────────────────┐  │
│  │  SharedAuthMiddleware (auto)       │  │
│  │  - Validates JWT token             │  │
│  │  - Populates request.state.user    │  │
│  └──────────┬─────────────────────────┘  │
│             │                             │
│  ┌──────────▼─────────────────────────┐  │
│  │  Routes                             │  │
│  │  - get_current_user()               │  │
│  │  - Uses request.state.user          │  │
│  └──────────┬─────────────────────────┘  │
└─────────────┼─────────────────────────────┘
              │
    ┌─────────▼─────────┐
    │  SharedUserPool   │
    │  - authenticate() │
    │  - validate_token()│
    │  - update_roles()  │
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │   MongoDB          │
    │   - _shared_users │
    │   - _shared_tokens│
    └───────────────────┘
              │
    ┌─────────▼─────────┐
    │  JWT Token        │
    │  (in cookie)      │
    │  Works across     │
    │  all apps         │
    └───────────────────┘
```

### Custom SSO Architecture

```
┌─────────────────────────────────────────┐
│         FastAPI App                     │
│  ┌───────────────────────────────────┐  │
│  │  Custom Auth Middleware           │  │
│  │  - Manual token validation        │  │
│  │  - Manual user population         │  │
│  └──────────┬─────────────────────────┘  │
│             │                             │
│  ┌──────────▼─────────────────────────┐  │
│  │  Routes                             │  │
│  │  - Custom user access               │  │
│  │  - Manual role checks               │  │
│  └──────────┬─────────────────────────┘  │
└─────────────┼─────────────────────────────┘
              │
    ┌─────────▼─────────┐
    │  Custom User Pool  │
    │  - Manual JWT      │
    │  - Manual hashing  │
    │  - Manual tokens   │
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │   MongoDB          │
    │   - users          │
    │   - tokens         │
    └───────────────────┘
```

---

## SSO Flow Comparison

### MDB-Engine SSO Flow

```
User logs into Click Tracker
    ↓
SharedUserPool.authenticate()
    ↓
JWT token generated
    ↓
Cookie set: mdb_auth_token
    ↓
User visits Dashboard
    ↓
SharedAuthMiddleware intercepts
    ↓
Validates token from cookie
    ↓
Populates request.state.user
    ↓
User automatically logged in!
```

### Custom SSO Flow

```
User logs into Click Tracker
    ↓
Custom authenticate() function
    ↓
Manual JWT generation
    ↓
Manual cookie setting
    ↓
User visits Dashboard
    ↓
Custom middleware intercepts
    ↓
Manual token validation
    ↓
Manual user lookup
    ↓
Manual request.state.user population
    ↓
User logged in (if all steps work)
```

---

## Summary

MDB-Engine provides significant value for SSO implementations:

| Component | Lines Saved | Complexity Reduced |
|-----------|-------------|-------------------|
| SharedUserPool | ~120 lines | User management, JWT, password hashing |
| SharedAuthMiddleware | ~50 lines | Token validation, user population |
| Per-app roles | ~30 lines | Role storage and retrieval |
| SSO cookie handling | ~20 lines | Cookie management |
| **Total Additional** | **~220 lines** | **Significant SSO complexity** |

Combined with previous features:
- Simple App: ~210 lines
- Multi-App: ~280 lines
- **Total**: **~710 lines saved** with production-ready SSO, security, and error handling.

The custom implementations shown are simplified. Production SSO systems would need:
- Token refresh mechanisms
- Concurrent session management
- Rate limiting for authentication
- Account lockout after failed attempts
- Password reset workflows
- Email verification
- Multi-factor authentication support
- Session timeout handling
- Token blacklisting for logout-all-devices

All of which MDB-Engine can be extended to support, or you can add custom logic on top of the base functionality.
