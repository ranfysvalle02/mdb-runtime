# Hello World Example

A simple example demonstrating MDB_RUNTIME with a single-app setup.

This example shows:
- How to initialize the runtime engine
- How to create and register an app manifest
- Basic CRUD operations with automatic app scoping
- How to use the scoped database wrapper

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (for containerized setup)
- OR MongoDB running locally (for local setup)
- MDB_RUNTIME installed

## Setup

### Using Docker Compose (Recommended)

Everything runs in Docker - MongoDB, the application, and optional services.

**Just run:**
```bash
docker-compose up
```

That's it! The Docker Compose setup will:
1. Build the application Docker image using multi-stage build (installs MDB_RUNTIME automatically)
2. Start MongoDB with authentication and health checks
3. Start the **web application** on http://localhost:8000
4. Start MongoDB Express (Web UI) - optional, use `--profile ui`
5. Show you all the output

**Access the Web UI:**
- 🌐 **Web Application**: http://localhost:8000
- 🔐 **Login Credentials**: 
  - Email: `demo@demo.com`
  - Password: `password123`

**With optional services:**
```bash
# Include MongoDB Express UI
docker-compose --profile ui up

# Include Ray
docker-compose --profile ray up

# Include both
docker-compose --profile ui --profile ray up
```

**View logs:**
```bash
# All services
docker-compose logs -f

# Just the app
docker-compose logs -f app

# Just MongoDB
docker-compose logs -f mongodb
```

**Stop everything:**
```bash
docker-compose down
```

**Rebuild after code changes:**
```bash
docker-compose up --build
```

**Production mode:**
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

**What gets started:**
- ✅ **App Container** - Runs the hello_world example automatically
- ✅ **MongoDB** - Database on port 27017
- ✅ **MongoDB Express** - Web UI on http://localhost:8081
- ✅ **Ray Head Node** - Optional, only with `--profile ray` flag

**Access the services:**
- 🌐 **Web Application**: http://localhost:8000 (login: demo@demo.com / password123)
- MongoDB: `mongodb://admin:password@localhost:27017/?authSource=admin`
- MongoDB Express UI: http://localhost:8081 (login: admin/admin)
- Ray Dashboard: http://localhost:8265 (if Ray is started)

**Run with Ray:**
```bash
docker-compose --profile ray up
```

**Run in detached mode:**
```bash
docker-compose up -d
docker-compose logs -f app  # Follow app logs
```

**Clean up (removes all data):**
```bash
docker-compose down -v
```

## Testing

After running `docker-compose up`, the app automatically runs and tests itself. You'll see the output directly in your terminal.

**To test manually:**

1. **Run the example:**
   ```bash
   docker-compose up
   ```

2. **View the output:**
   The app will automatically:
   - Connect to MongoDB
   - Register the app
   - Create sample data
   - Query the data
   - Update records
   - Display health status

3. **Check logs:**
   ```bash
   # View all logs
   docker-compose logs -f
   
   # View just the app logs
   docker-compose logs -f app
   ```

4. **Run again (after stopping):**
   ```bash
   docker-compose down
   docker-compose up
   ```

5. **Test with MongoDB Express UI:**
   ```bash
   docker-compose --profile ui up
   ```
   Then visit http://localhost:8081 to browse the database and see the created data.

6. **Verify data in MongoDB:**
   ```bash
   # Connect to MongoDB container
   docker-compose exec mongodb mongosh -u admin -p password --authenticationDatabase admin
   
   # In mongosh:
   use hello_world_db
   db.greetings.find()
   ```

## What This Example Does

### Web Application Features

The hello_world example includes a **full-featured web application** with:

1. **🔐 Authentication** - Login/logout with JWT tokens
   - Demo user: `demo@demo.com` / `password123`
   - Secure session management with HTTP-only cookies

2. **📊 Dashboard** - Beautiful, modern UI showing:
   - Real-time greeting management (Create, Read, Delete)
   - System health status
   - Statistics (total greetings, languages)
   - Multi-language support

3. **🎨 Modern UI/UX** - Responsive design with:
   - Gradient backgrounds
   - Smooth animations
   - Mobile-friendly layout
   - Real-time updates

### Backend Features

1. **Initializes the Runtime Engine** - Connects to MongoDB and sets up the runtime
2. **Registers the App** - Loads the manifest and registers the "hello_world" app
3. **Creates Data** - Inserts sample documents (seeded on startup)
4. **Queries Data** - Demonstrates find operations with automatic app scoping
5. **Updates Data** - Shows how updates work with scoped database
6. **Shows Health Status** - Displays engine health information via API

## Expected Output

```
🚀 Initializing MDB_RUNTIME...
✅ Engine initialized successfully
✅ App 'hello_world' registered successfully

📝 Creating sample data...
✅ Created greeting: Hello, World!
✅ Created greeting: Hello, MDB_RUNTIME!
✅ Created greeting: Hello, Python!

🔍 Querying data...
✅ Found 3 greetings
   - Hello, World! (language: en)
   - Hello, MDB_RUNTIME! (language: en)
   - Hello, Python! (language: en)

🔍 Finding English greetings...
✅ Found 3 English greetings

✏️  Updating greeting...
✅ Updated greeting: Hello, Updated World!

🔍 Verifying update...
✅ Found updated greeting: Hello, Updated World!

📊 Health Status:
   - Status: connected
   - App Count: 1
   - Initialized: True

✅ Example completed successfully!
```

## Understanding the Code

### Manifest (`manifest.json`)

The manifest defines your app configuration:
- `slug`: Unique identifier for your app
- `name`: Human-readable name
- `status`: App status (active, draft, etc.)
- `managed_indexes`: Indexes to create automatically

### App Scoping

Notice that when you insert documents, you don't specify `app_id`. MDB_RUNTIME automatically adds it:

```python
# You write:
await db.greetings.insert_one({"message": "Hello, World!", "language": "en"})

# MDB_RUNTIME stores:
{
  "message": "Hello, World!",
  "language": "en",
  "app_id": "hello_world"  # ← Added automatically
}
```

### Automatic Filtering

When you query, MDB_RUNTIME automatically filters by `app_id`:

```python
# You write:
greetings = await db.greetings.find({"language": "en"}).to_list(length=10)

# MDB_RUNTIME executes:
db.greetings.find({
  "$and": [
    {"app_id": {"$in": ["hello_world"]}},  # ← Added automatically
    {"language": "en"}
  ]
})
```

## Docker Compose Services

The `docker-compose.yml` file includes:

The `docker-compose.yml` file includes:

### App Service
- **Container:** `hello_world_app`
- **Purpose:** Runs the hello_world example
- **Build:** Automatically builds from Dockerfile
- **Dependencies:** Waits for MongoDB to be healthy

### MongoDB
- **Port:** 27017
- **Credentials:** admin/password (change in production!)
- **Database:** hello_world_db
- **Health Check:** Enabled (app waits for this)

### MongoDB Express (Web UI)
- **URL:** http://localhost:8081
- **Credentials:** admin/admin
- **Purpose:** Browse and manage your MongoDB data visually

### Ray Head Node (Optional)
- **Dashboard:** http://localhost:8265
- **Port:** 6379 (GCS), 10001 (Client)
- **Purpose:** Distributed computing
- **Note:** Only starts with `--profile ray` flag

### Environment Variables

Environment variables are set in `docker-compose.yml`. You can override them:

```bash
# In docker-compose.yml or via command line
MONGO_URI=mongodb://admin:password@mongodb:27017/?authSource=admin
MONGO_DB_NAME=hello_world_db
APP_SLUG=hello_world
LOG_LEVEL=INFO
```

### Customizing the Setup

1. **Modify `docker-compose.yml`** to change service configurations

2. **Modify `Dockerfile`** to change the build process

3. **Rebuild and restart:**
   ```bash
   docker-compose up --build
   ```

## Troubleshooting

### ModuleNotFoundError: No module named 'mdb_runtime'

If you see this error, the package wasn't installed correctly during the Docker build. Fix it by:

1. **Rebuild the Docker image:**
   ```bash
   docker-compose build --no-cache
   docker-compose up
   ```

2. **Or force a complete rebuild:**
   ```bash
   docker-compose down
   docker-compose build --no-cache
   docker-compose up
   ```

This error was fixed in the Dockerfile by changing from editable install (`-e`) to regular install, which works correctly when copying the virtual environment between Docker stages.

### App Won't Start

1. **Check if MongoDB is healthy:**
   ```bash
   docker-compose ps
   docker-compose logs mongodb
   ```

2. **Check app logs:**
   ```bash
   docker-compose logs app
   ```

3. **Rebuild the app:**
   ```bash
   docker-compose up --build
   ```

### MongoDB Connection Issues

The app connects to MongoDB using the service name `mongodb` (Docker networking).
If you see connection errors:

1. **Verify MongoDB is running:**
   ```bash
   docker-compose ps mongodb
   ```

2. **Check MongoDB is healthy:**
   ```bash
   docker-compose logs mongodb | grep "health"
   ```

3. **Test connection from app container:**
   ```bash
   docker-compose exec app python -c "from motor.motor_asyncio import AsyncIOMotorClient; import asyncio; asyncio.run(AsyncIOMotorClient('mongodb://admin:password@mongodb:27017/?authSource=admin').admin.command('ping'))"
   ```

### Port Conflicts

If ports are already in use, modify `docker-compose.yml`:

```yaml
services:
  mongodb:
    ports:
      - "27018:27017"  # Change host port
```

Then update `MONGO_URI` in the app service to use the new port if accessing from outside Docker.

### Rebuild After Code Changes

If you modify `mdb_runtime` or the example code:

```bash
docker-compose up --build
```

## Docker Best Practices

This example follows enterprise-grade Docker best practices:

- ✅ **Multi-stage builds** - Smaller, more secure images
- ✅ **Non-root user** - Security best practice
- ✅ **Health checks** - Proper service monitoring
- ✅ **Resource limits** - Prevent resource exhaustion
- ✅ **Layer caching** - Optimized build performance
- ✅ **Network isolation** - Secure service communication
- ✅ **Service dependencies** - Proper startup ordering

See [DOCKER_BEST_PRACTICES.md](./DOCKER_BEST_PRACTICES.md) for a comprehensive guide on:
- Why each practice matters
- How to apply them to your own applications
- Production deployment considerations
- Security hardening
- Performance optimization

## Next Steps

- Try adding more collections
- Experiment with different queries
- Add indexes to the manifest
- Check out the FastAPI example for building a real API
- Explore MongoDB Express UI: `docker-compose --profile ui up` then http://localhost:8081
- Check Ray Dashboard: `docker-compose --profile ray up` then http://localhost:8265
- Review [DOCKER_BEST_PRACTICES.md](./DOCKER_BEST_PRACTICES.md) for production deployment

