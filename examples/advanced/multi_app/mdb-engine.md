# MDB-Engine Components in Multi-App Example

This document isolates the MDB-Engine specific parts of the Multi-App example, explains their value, and shows how to implement your own versions.

## Overview

The Multi-App example demonstrates advanced MDB-Engine features for building multi-tenant SaaS platforms:
- **Cross-app data access** - Secure reading of data from other apps with automatic validation
- **Casbin authorization** - Role-based access control (RBAC) with automatic setup
- **App-level authentication** - Envelope encryption for app secrets
- **Automatic policy management** - Policies and roles seeded from manifest
- **Data isolation** - Automatic app_id filtering ensures data separation

**Without MDB-Engine**, you would need to implement:
- All features from Simple App, plus:
- Cross-app access control logic
- App secret encryption/decryption
- Token verification for cross-app requests
- Collection name prefixing and scoping
- Casbin adapter setup and configuration
- Policy and role management
- Secret rotation workflows
- Audit logging for cross-app access

## MDB-Engine Components

### 1. Basic Components (from Simple App)

All components from the Simple App example are also used here:
- `MongoDBEngine()` - Database connection management
- `engine.create_app()` - FastAPI app creation with automatic lifecycle
- `get_scoped_db()` - Scoped database access via dependency injection
- Manifest-driven index creation

See [simple_app/mdb-engine.md](../simple_app/mdb-engine.md) for details on these components.

---

### 2. Cross-App Data Access with Automatic Validation

**What it is:**

In Dashboard's `manifest.json`:
```json
{
  "data_access": {
    "read_scopes": ["click_tracker_dashboard", "click_tracker"],
    "write_scope": "click_tracker_dashboard",
    "cross_app_policy": "explicit"
  }
}
```

In code:
```python
@app.get("/analytics")
async def get_analytics(db=Depends(get_scoped_db)):
    # CROSS-APP ACCESS: Read click_tracker's clicks collection!
    clicks_collection = db.get_collection("click_tracker_clicks")
    clicks = await clicks_collection.find({}).to_list(length=1000)
    return {"clicks": clicks}
```

**What it does:**

- Allows apps to read data from other apps' collections
- Validates cross-app access based on `read_scopes` in manifest
- Prefixes collection names with app slug (e.g., `click_tracker_clicks`)
- Verifies app tokens for cross-app requests automatically
- Enforces data isolation while allowing controlled sharing
- Logs cross-app access for audit trail
- Handles token retrieval from environment or encrypted storage

**Value:**

- **Secure sharing**: Controlled cross-app data access without manual validation
- **Automatic validation**: Access permissions checked automatically from manifest
- **Collection naming**: Automatic prefixing prevents conflicts between apps
- **Audit trail**: All cross-app access logged automatically
- **Type safety**: Collection access is validated at runtime
- **Zero boilerplate**: No manual token management or validation code

**How to implement your own:**

```python
class CrossAppDatabase:
    """Database wrapper that handles cross-app access."""
    
    def __init__(self, db, app_slug: str, read_scopes: list, write_scope: str, app_token: str):
        self._db = db
        self.app_slug = app_slug
        self.read_scopes = read_scopes
        self.write_scope = write_scope
        self.app_token = app_token
    
    def get_collection(self, collection_name: str):
        """Get collection from another app."""
        # Parse app slug from collection name
        # Format: {app_slug}_{collection_name}
        if "_" in collection_name:
            parts = collection_name.split("_", 1)
            source_app = parts[0]
            actual_collection = parts[1]
        else:
            # Default to current app
            source_app = self.app_slug
            actual_collection = collection_name
        
        # Validate read access
        if source_app != self.app_slug:
            if source_app not in self.read_scopes:
                raise HTTPException(
                    status_code=403,
                    detail=f"App {self.app_slug} cannot read from {source_app}"
                )
            
            # Verify app token
            if not self._verify_app_token(source_app, self.app_token):
                raise HTTPException(
                    status_code=401,
                    detail="Invalid app token for cross-app access"
                )
        
        # Return collection with app prefix
        prefixed_name = f"{source_app}_{actual_collection}"
        return self._db[prefixed_name]
    
    def _verify_app_token(self, target_app: str, token: str) -> bool:
        """Verify app token matches stored secret."""
        # Get stored secret from database
        secret_doc = await self._db._mdb_engine_app_secrets.find_one(
            {"app_slug": target_app}
        )
        if not secret_doc:
            return False
        
        # Decrypt and compare
        encrypted_secret = secret_doc["encrypted_secret"]
        stored_secret = self._decrypt_secret(encrypted_secret)
        return secrets.compare_digest(token, stored_secret)
    
    def __getattr__(self, name):
        """Access collections from current app."""
        # Current app collections are automatically prefixed
        prefixed_name = f"{self.app_slug}_{name}"
        return self._db[prefixed_name]

async def get_scoped_db(
    app_slug: str,
    manifest: dict,
    app_token: str
) -> CrossAppDatabase:
    """Get scoped database with cross-app access."""
    db = db_manager.get_db()
    
    data_access = manifest.get("data_access", {})
    read_scopes = data_access.get("read_scopes", [app_slug])
    write_scope = data_access.get("write_scope", app_slug)
    
    return CrossAppDatabase(db, app_slug, read_scopes, write_scope, app_token)
```

**Comparison:**

| Feature | MDB-Engine | Custom Implementation |
|---------|-----------|----------------------|
| Lines of code | 1 method call | ~100+ |
| Access validation | Automatic | Manual checks |
| Collection prefixing | Automatic | Manual string manipulation |
| Token verification | Built-in | Manual verification |
| Token retrieval | Automatic (env or DB) | Manual env/db checks |
| Audit logging | Automatic | Manual logging |
| Error handling | Built-in | Manual exception handling |
| Secret encryption | Built-in | Manual crypto implementation |

---

### 3. Casbin Authorization Provider

**What it is:**

```python
from mdb_engine.dependencies import get_authz_provider

@app.post("/track")
async def track_click(
    request: Request,
    authz=Depends(get_authz_provider),
    db=Depends(get_scoped_db),
):
    user = await get_current_user(request, db=db)
    
    # Authorization check
    if authz and not await authz.check(user.get("email"), "clicks", "write"):
        raise HTTPException(status_code=403, detail="Permission denied")
    
    # ... rest of handler
```

**What it does:**

- Provides a Casbin authorization provider configured from manifest
- Automatically sets up Casbin adapter (MongoDB-based)
- Loads RBAC model from manifest configuration
- Seeds initial policies and roles from manifest
- Provides `authz.check(subject, resource, action)` method
- Handles policy evaluation and caching
- Supports role inheritance and hierarchical permissions

**Value:**

- **Zero setup**: Casbin configured automatically from manifest
- **Policy seeding**: Initial policies loaded from JSON, not code
- **Role management**: Roles assigned automatically from manifest
- **Type safety**: Policy evaluation is type-checked
- **Performance**: Built-in caching for policy checks
- **Consistency**: Same authorization model across all apps

**How to implement your own:**

```python
import casbin
from casbin_motor_adapter import Adapter as MotorAdapter
from motor.motor_asyncio import AsyncIOMotorClient

class AuthorizationProvider:
    def __init__(self, db, model_path: str):
        self.db = db
        self.enforcer = None
        self.model_path = model_path
    
    async def initialize(self):
        """Initialize Casbin enforcer with MongoDB adapter."""
        adapter = MotorAdapter(self.db, "casbin_policies")
        
        # Load RBAC model
        self.enforcer = casbin.Enforcer(self.model_path, adapter)
        await self.enforcer.load_policy()
    
    async def check(self, subject: str, resource: str, action: str) -> bool:
        """Check if subject has permission to perform action on resource."""
        if not self.enforcer:
            await self.initialize()
        
        # Casbin RBAC check: subject, resource, action
        return await self.enforcer.enforce(subject, resource, action)
    
    async def add_policy(self, subject: str, resource: str, action: str):
        """Add a policy rule."""
        if not self.enforcer:
            await self.initialize()
        await self.enforcer.add_policy(subject, resource, action)
    
    async def add_role_for_user(self, user: str, role: str):
        """Assign role to user."""
        if not self.enforcer:
            await self.initialize()
        await self.enforcer.add_role_for_user(user, role)

async def seed_policies(db, manifest):
    """Seed initial policies and roles from manifest."""
    auth_config = manifest.get("auth", {}).get("policy", {})
    authz_config = auth_config.get("authorization", {})
    
    provider = AuthorizationProvider(db, "rbac_model.conf")
    await provider.initialize()
    
    # Seed initial policies
    initial_policies = authz_config.get("initial_policies", [])
    for policy in initial_policies:
        await provider.add_policy(*policy)
    
    # Seed initial roles
    initial_roles = authz_config.get("initial_roles", [])
    for role_assignment in initial_roles:
        await provider.add_role_for_user(
            role_assignment["user"],
            role_assignment["role"]
        )
    
    return provider

async def get_authz_provider():
    """FastAPI dependency for authorization provider."""
    # Get from app state (initialized at startup)
    return app.state.authz_provider

# Usage in lifespan
@asynccontextmanager
async def lifespan(app):
    await db_manager.connect()
    manifest = await load_manifest(Path("manifest.json"))
    
    # Initialize authorization
    authz_provider = await seed_policies(db_manager.get_db(), manifest)
    app.state.authz_provider = authz_provider
    
    yield
    
    await db_manager.disconnect()
```

**Comparison:**

| Feature | MDB-Engine | Custom Implementation |
|---------|-----------|----------------------|
| Lines of code | 1 import | ~100+ |
| Casbin setup | Automatic | Manual adapter setup |
| Model loading | From manifest | Manual config file |
| Policy seeding | Automatic | Manual seeding code |
| Role assignment | Automatic | Manual role management |
| Caching | Built-in | Manual implementation |
| Error handling | Built-in | Manual try/except |

---

### 4. App-Level Authentication with Envelope Encryption

**What it is:**

MDB-Engine automatically handles app-level authentication when `read_scopes` includes other apps. Each app has a secret token that is:
- Generated automatically during app registration
- Encrypted using envelope encryption
- Stored in `_mdb_engine_app_secrets` collection
- Verified on every cross-app database access
- Auto-retrieved from database or environment variables

**What it does:**

- Generates unique 256-bit secrets for each app automatically
- Encrypts secrets using envelope encryption (master key + data encryption key)
- Stores encrypted secrets in MongoDB securely
- Auto-retrieves secrets from database or environment
- Verifies app tokens on cross-app requests
- Rotates secrets securely when needed
- Handles secret migration and backward compatibility

**Value:**

- **Security**: Secrets encrypted at rest with envelope encryption
- **Automatic**: No manual secret management required
- **Secure storage**: Even if database compromised, secrets remain encrypted
- **Easy rotation**: Built-in secret rotation support
- **Environment override**: Can use env vars for better performance
- **Zero configuration**: Works out of the box with sensible defaults

**How to implement your own:**

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import secrets

class EnvelopeEncryption:
    """Envelope encryption for app secrets."""
    
    def __init__(self, master_key: str):
        self.master_key = master_key.encode()
        self.cipher = self._create_cipher()
    
    def _create_cipher(self):
        """Create Fernet cipher from master key."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'mdb_engine_salt',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        return Fernet(key)
    
    def encrypt_secret(self, secret: str) -> str:
        """Encrypt app secret."""
        return self.cipher.encrypt(secret.encode()).decode()
    
    def decrypt_secret(self, encrypted_secret: str) -> str:
        """Decrypt app secret."""
        return self.cipher.decrypt(encrypted_secret.encode()).decode()

class AppSecretManager:
    """Manages app secrets with envelope encryption."""
    
    def __init__(self, db, master_key: str):
        self.db = db
        self.encryption = EnvelopeEncryption(master_key)
        self.secrets_collection = db["_app_secrets"]
    
    async def get_app_secret(self, app_slug: str) -> str:
        """Get app secret (from env or database)."""
        # Check environment first
        env_var = f"{app_slug.upper()}_SECRET"
        secret = os.getenv(env_var)
        if secret:
            return secret
        
        # Get from database
        doc = await self.secrets_collection.find_one({"app_slug": app_slug})
        if not doc:
            # Generate and store new secret
            return await self.generate_app_secret(app_slug)
        
        # Decrypt and return
        encrypted_secret = doc["encrypted_secret"]
        return self.encryption.decrypt_secret(encrypted_secret)
    
    async def generate_app_secret(self, app_slug: str) -> str:
        """Generate and store new app secret."""
        # Generate random 256-bit secret
        secret = secrets.token_urlsafe(32)
        
        # Encrypt
        encrypted_secret = self.encryption.encrypt_secret(secret)
        
        # Store in database
        await self.secrets_collection.update_one(
            {"app_slug": app_slug},
            {
                "$set": {
                    "app_slug": app_slug,
                    "encrypted_secret": encrypted_secret,
                    "created_at": datetime.utcnow(),
                }
            },
            upsert=True
        )
        
        return secret
    
    async def verify_app_token(self, app_slug: str, token: str) -> bool:
        """Verify app token matches stored secret."""
        secret = await self.get_app_secret(app_slug)
        return secrets.compare_digest(token, secret)

# Usage
master_key = os.getenv("MDB_ENGINE_MASTER_KEY")
secret_manager = AppSecretManager(db_manager.get_db(), master_key)

# In cross-app access
async def verify_cross_app_access(source_app: str, target_app: str, token: str):
    """Verify app has permission to access target app's data."""
    # Verify token
    if not await secret_manager.verify_app_token(source_app, token):
        raise HTTPException(status_code=401, detail="Invalid app token")
    
    # Check read_scopes (from manifest)
    # ... (validation logic)
```

**Comparison:**

| Feature | MDB-Engine | Custom Implementation |
|---------|-----------|----------------------|
| Lines of code | 0 (automatic) | ~200+ |
| Secret generation | Automatic | Manual token generation |
| Encryption | Built-in | Manual crypto implementation |
| Storage | Automatic | Manual database operations |
| Retrieval | Automatic (env or DB) | Manual env/db checks |
| Verification | Automatic | Manual comparison logic |
| Rotation | Built-in | Manual rotation code |
| Migration | Built-in | Manual migration scripts |

---

### 5. Dependency Injection Pattern

**What it is:**

```python
from mdb_engine.dependencies import get_scoped_db, get_authz_provider

@app.get("/clicks")
async def get_clicks(
    request: Request,
    authz=Depends(get_authz_provider),
    db=Depends(get_scoped_db),
):
    # db and authz are automatically injected by FastAPI
    # db is scoped to the current app
    # authz is configured from manifest
    clicks = await db.clicks.find({}).to_list(length=100)
    return {"clicks": clicks}
```

**What it does:**

- Provides request-scoped database connections
- Automatically retrieves app slug from `app.state.app_slug`
- Configures database wrapper with correct scopes from manifest
- Provides authorization provider configured from manifest
- Handles connection pooling and error recovery
- Works seamlessly with FastAPI's dependency injection system

**Value:**

- **Request-scoped**: Each request gets its own database connection
- **Automatic configuration**: No manual setup needed
- **Type safety**: Full type hints and IDE support
- **Testability**: Easy to mock dependencies in tests
- **Consistency**: Same pattern across all routes
- **Error handling**: Built-in connection recovery

**How to implement your own:**

```python
from fastapi import Depends, Request
from typing import Annotated

async def get_scoped_db(request: Request) -> ScopedDatabase:
    """FastAPI dependency for scoped database access."""
    # Get engine from app state
    engine = getattr(request.app.state, "engine", None)
    if not engine:
        raise HTTPException(503, "Engine not initialized")
    
    # Get app slug from app state
    app_slug = getattr(request.app.state, "app_slug", None)
    if not app_slug:
        raise HTTPException(503, "App slug not configured")
    
    # Get manifest from app state
    manifest = getattr(request.app.state, "manifest", {})
    
    # Get read_scopes from manifest
    data_access = manifest.get("data_access", {})
    read_scopes = data_access.get("read_scopes", [app_slug])
    write_scope = data_access.get("write_scope", app_slug)
    
    # Get app token (from env or database)
    app_token = await get_app_token(app_slug, engine)
    
    # Return scoped database
    return engine.get_scoped_db(
        app_slug=app_slug,
        app_token=app_token,
        read_scopes=read_scopes,
        write_scope=write_scope,
    )

async def get_authz_provider(request: Request):
    """FastAPI dependency for authorization provider."""
    authz = getattr(request.app.state, "authz_provider", None)
    if not authz:
        raise HTTPException(503, "Authorization not configured")
    return authz

# Usage
@app.get("/clicks")
async def get_clicks(
    db: Annotated[ScopedDatabase, Depends(get_scoped_db)],
    authz: Annotated[AuthorizationProvider, Depends(get_authz_provider)],
):
    clicks = await db.clicks.find({}).to_list(length=100)
    return {"clicks": clicks}
```

**Comparison:**

| Feature | MDB-Engine | Custom Implementation |
|---------|-----------|----------------------|
| Lines of code | 1 import | ~50+ |
| Request scoping | Automatic | Manual request handling |
| App state access | Automatic | Manual app.state access |
| Token retrieval | Automatic | Manual env/db checks |
| Error handling | Built-in | Manual exception handling |
| Type hints | Full support | Manual annotations |

---

## Complete Replacement Guide

### Step 1: Add Cross-App Access

**Remove:**
```python
clicks_collection = db.get_collection("click_tracker_clicks")
```

**Replace with:**
```python
# Add CrossAppDatabase class (from section 2)
# Update get_scoped_db() to return CrossAppDatabase
# Add validation logic
```

### Step 2: Add Casbin Authorization

**Remove:**
```python
from mdb_engine.dependencies import get_authz_provider

authz=Depends(get_authz_provider)
```

**Replace with:**
```python
# Add AuthorizationProvider class (from section 3)
# Add seed_policies function
# Initialize in lifespan handler
```

### Step 3: Add App Secret Management

**Remove:**
```python
# MDB-Engine handles this automatically
```

**Replace with:**
```python
# Add EnvelopeEncryption class (from section 4)
# Add AppSecretManager class
# Add token verification in cross-app access
```

### Step 4: Update Manifest Loading

**Add:**
```python
# Load data_access section
data_access = manifest.get("data_access", {})
read_scopes = data_access.get("read_scopes", [])
write_scope = data_access.get("write_scope", app_slug)

# Pass to database wrapper
```

---

## Migration Guide

### Migrating FROM MDB-Engine TO Custom Implementation

1. **Add Casbin setup** (from section 3)
2. **Add cross-app database wrapper** (from section 2)
3. **Add app secret management** (from section 4)
4. **Update all routes** to use custom dependencies
5. **Add token verification** to cross-app requests
6. **Add audit logging** for cross-app access
7. **Add secret rotation** workflows

### Migrating FROM Custom Implementation TO MDB-Engine

1. **Remove Casbin setup code** - MDB-Engine handles it
2. **Remove cross-app wrapper** - Use `db.get_collection()` directly
3. **Remove secret management** - MDB-Engine handles encryption
4. **Update manifest** - Add `data_access` section
5. **Remove token verification** - Automatic in MDB-Engine
6. **Simplify routes** - Use `get_authz_provider()` dependency

---

## Architecture Comparison

### MDB-Engine Architecture

```
┌─────────────────────────────────────────┐
│         FastAPI App                      │
│  ┌───────────────────────────────────┐  │
│  │  Routes                            │  │
│  │  - get_authz_provider()            │  │
│  │  - get_scoped_db()                 │  │
│  └──────────┬──────────────┬──────────┘  │
│             │              │             │
│  ┌──────────▼──────────┐  │             │
│  │  MongoDBEngine      │  │             │
│  │  - Casbin setup     │  │             │
│  │  - Cross-app access │  │             │
│  │  - Secret mgmt      │  │             │
│  └──────────┬──────────┘  │             │
└─────────────┼──────────────┼─────────────┘
              │              │
    ┌─────────▼─────────┐   │
    │   MongoDB         │   │
    │   - Policies      │   │
    │   - Secrets       │   │
    │   - App data      │   │
    └───────────────────┘   │
                             │
                    ┌────────▼────────┐
                    │  Manifest.json  │
                    │  - Policies     │
                    │  - Roles        │
                    │  - read_scopes   │
                    └─────────────────┘
```

### Custom Implementation Architecture

```
┌─────────────────────────────────────────┐
│         FastAPI App                      │
│  ┌───────────────────────────────────┐  │
│  │  Routes                            │  │
│  │  - Custom authz provider           │  │
│  │  - Custom db wrapper               │  │
│  └──────────┬──────────────┬──────────┘  │
│             │              │             │
│  ┌──────────▼──────────┐  │             │
│  │  Authorization      │  │             │
│  │  Provider            │  │             │
│  │  - Casbin setup      │  │             │
│  │  - Policy mgmt       │  │             │
│  └──────────┬──────────┘  │             │
│             │              │             │
│  ┌──────────▼──────────┐  │             │
│  │  Database Manager    │  │             │
│  │  - Connection        │  │             │
│  │  - Cross-app logic   │  │             │
│  └──────────┬──────────┘  │             │
│             │              │             │
│  ┌──────────▼──────────┐  │             │
│  │  Secret Manager      │  │             │
│  │  - Encryption         │  │             │
│  │  - Token verification │  │             │
│  └──────────┬──────────┘  │             │
└─────────────┼──────────────┼─────────────┘
              │              │
    ┌─────────▼─────────┐   │
    │   MongoDB         │   │
    │   - Policies      │   │
    │   - Secrets       │   │
    │   - App data      │   │
    └───────────────────┘   │
                             │
                    ┌────────▼────────┐
                    │  Config Files   │
                    │  - Casbin model │
                    │  - Policies     │
                    │  - Secrets      │
                    └─────────────────┘
```

---

## Summary

MDB-Engine provides significant value for multi-app architectures:

| Component | Lines Saved | Complexity Reduced |
|-----------|-------------|-------------------|
| Cross-App Access | ~100 lines | Access validation, collection naming, token verification |
| Casbin Authorization | ~100 lines | Policy management, role assignment, adapter setup |
| App Secret Management | ~200 lines | Encryption, storage, verification, rotation |
| Dependency Injection | ~50 lines | Request scoping, app state access, error handling |
| **Total Additional** | **~450 lines** | **Significant security and infrastructure complexity** |

Combined with Simple App features (~210 lines), MDB-Engine saves approximately **~660 lines of infrastructure code** while providing production-ready security, error handling, and audit capabilities.

The custom implementations shown are simplified. Production systems would need additional features like:
- Policy caching and invalidation
- Secret rotation workflows
- Cross-app access rate limiting
- Comprehensive audit logging
- Token refresh mechanisms
- Error recovery and retry logic
- Connection pooling optimization
- Health check endpoints
- Metrics and monitoring

All of which MDB-Engine provides out of the box.

---

## Key Takeaways

1. **Cross-app access is declarative**: Define `read_scopes` in manifest, MDB-Engine handles the rest
2. **Secrets are automatic**: Generated, encrypted, and verified automatically
3. **Authorization is manifest-driven**: Policies and roles defined in JSON, not code
4. **Dependencies are request-scoped**: Each request gets properly configured database and authz
5. **Zero boilerplate**: Focus on business logic, not infrastructure

---

## Next Steps

- Review the [Simple App example](../simple_app/mdb-engine.md) for basic patterns
- Explore the code to see how cross-app access works in practice
- Modify manifests to experiment with different `read_scopes`
- Review [Security documentation](../../docs/SECURITY.md) for production deployment
- Check [App Authentication guide](../../docs/APP_AUTHENTICATION.md) for detailed auth flows
