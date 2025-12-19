# MDB_RUNTIME Examples

This directory contains example applications demonstrating how to use MDB_RUNTIME.

## Available Examples

### [Hello World](./hello_world/)

A simple, beginner-friendly example that demonstrates:
- Initializing the runtime engine
- Creating and registering an app manifest
- Basic CRUD operations
- Automatic app scoping
- Docker Compose setup with MongoDB

**Perfect for:** Getting started with MDB_RUNTIME

**Run it:**
```bash
cd hello_world
./run_with_docker.sh
```

## Docker Compose Setup

Each example includes a `docker-compose.yml` file that provides:

### Standard Services

- **MongoDB** - Database server with authentication
- **MongoDB Express** - Web UI for browsing data (optional)

### Quick Start with Docker

```bash
# Navigate to an example
cd hello_world

# Start all services
docker-compose up -d

# Run the example
python main.py

# Stop services
docker-compose down
```

### Service URLs

When Docker Compose is running:

- **MongoDB:** `mongodb://admin:password@localhost:27017/?authSource=admin`
- **MongoDB Express UI:** http://localhost:8081 (admin/admin, optional with `--profile ui`)

## Running Examples

### Prerequisites

**Just Docker and Docker Compose!** That's it.

- Docker Desktop: https://www.docker.com/products/docker-desktop
- Or install separately: https://docs.docker.com/compose/install/

No need to install Python, MongoDB, or MDB_RUNTIME - everything runs in containers.

### Quick Start

```bash
cd hello_world
docker-compose up
```

The example will:
1. Build the application (installs MDB_RUNTIME automatically)
2. Start MongoDB
3. Start MongoDB Express (Web UI)
4. Run the example automatically
5. Show you all the output

### Example Structure

Each example includes:
- `README.md` - Explanation of what the example does
- `manifest.json` - App configuration manifest
- `main.py` - Main example code
- `Dockerfile` - Builds and runs the example
- `docker-compose.yml` - Orchestrates all services

### Environment Variables

Environment variables are set in `docker-compose.yml`. Common variables:
- `MONGO_URI` - MongoDB connection string (uses Docker service name)
- `MONGO_DB_NAME` - Database name
- `APP_SLUG` - App identifier
- `LOG_LEVEL` - Logging level

### How the Dockerfile Works

The Dockerfile:
1. Copies `mdb_runtime` source code from the project root
2. Installs it with `pip install -e` (editable mode)
3. Installs all dependencies from `pyproject.toml`
4. Copies the example files
5. Runs the example automatically

No manual installation needed!

## Troubleshooting

### App Container Issues

1. **View app logs:**
   ```bash
   docker-compose logs app
   ```

2. **Rebuild after code changes:**
   ```bash
   docker-compose up --build
   ```

3. **Check if MongoDB is ready:**
   ```bash
   docker-compose ps mongodb
   docker-compose logs mongodb
   ```

### Port Conflicts

Modify ports in `docker-compose.yml`:

```yaml
services:
  mongodb:
    ports:
      - "27018:27017"  # Change host port
```

### Services Not Starting

1. **Check Docker is running:**
   ```bash
   docker ps
   ```

2. **View all logs:**
   ```bash
   docker-compose logs
   ```

3. **Check service health:**
   ```bash
   docker-compose ps
   ```

## Contributing Examples

If you've built something cool with MDB_RUNTIME, consider contributing an example! Examples should:

- Be self-contained and runnable
- Include a README explaining what they demonstrate
- Include a `docker-compose.yml` with all required services
- Use clear, well-commented code
- Focus on specific features or use cases
- Include a manifest.json file
- Include environment variable examples

## Need Help?

- Check the [main README](../../README.md) for general documentation
- See the [Quick Start Guide](../../docs/QUICK_START.md) for detailed setup instructions
- Open an issue if you encounter problems
