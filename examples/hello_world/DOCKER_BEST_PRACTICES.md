# Docker Best Practices for MDB_RUNTIME

This document explains the enterprise-grade Docker practices used in the hello_world example, designed to scale from hello world to production applications.

## Architecture Overview

### Multi-Stage Builds

The Dockerfile uses a two-stage build:

1. **Builder Stage**: Installs dependencies and builds the package
2. **Runtime Stage**: Minimal production image with only what's needed

**Benefits:**
- Smaller final image (no build tools)
- Better security (fewer attack surfaces)
- Faster deployments (smaller images transfer faster)

### Security Best Practices

1. **Non-Root User**
   ```dockerfile
   USER appuser  # Runs as non-root
   ```
   - Reduces attack surface
   - Limits damage if container is compromised
   - Required for many production environments

2. **Minimal Base Image**
   - Uses `python:3.11-slim` (Debian-based, minimal)
   - Only installs necessary packages
   - Removes package manager cache after installation

3. **Read-Only Filesystem** (where possible)
   ```yaml
   read_only: true
   tmpfs:
     - /tmp:noexec,nosuid,size=100m
   ```
   - Prevents file system tampering
   - Uses tmpfs for temporary files

4. **Resource Limits**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '1.0'
         memory: 512M
   ```
   - Prevents resource exhaustion
   - Ensures fair resource allocation

## Layer Caching Optimization

The Dockerfile is structured for optimal layer caching:

```dockerfile
# 1. Copy dependency files first (changes rarely)
COPY pyproject.toml setup.py MANIFEST.in ./

# 2. Install dependencies (only rebuilds if dependencies change)
RUN pip install --no-cache-dir -e .

# 3. Copy source code last (changes frequently)
COPY mdb_runtime ./mdb_runtime
```

**Why this matters:**
- Dependency installation is cached until `pyproject.toml` changes
- Source code changes don't invalidate dependency cache
- Faster rebuilds during development

## Docker Compose Best Practices

### Health Checks

All services have health checks:

```yaml
healthcheck:
  test: echo 'db.runCommand("ping").ok' | mongosh localhost:27017/test --quiet
  interval: 10s
  timeout: 5s
  retries: 5
  start_period: 10s
```

**Benefits:**
- Docker knows when services are ready
- `depends_on` with `condition: service_healthy` ensures proper startup order
- Automatic restart on failure

### Service Dependencies

```yaml
depends_on:
  mongodb:
    condition: service_healthy
```

- App waits for MongoDB to be ready
- Prevents connection errors on startup
- Ensures proper initialization order

### Network Isolation

```yaml
networks:
  mdb_runtime_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16
```

- Services communicate on isolated network
- Not exposed to host network
- Better security and organization

### Profiles for Optional Services

```yaml
profiles:
  - ui  # Only start with: docker-compose --profile ui up
```

- MongoDB Express is optional
- Start only what you need
- Reduces resource usage

## Production Considerations

### Using Production Override

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

The `docker-compose.prod.yml` includes:
- `restart: always` for all services
- Log rotation configuration
- Production-appropriate resource limits
- Named volumes instead of bind mounts

### Environment Variables

Never commit secrets. Use:
- `.env` files (not in git)
- Docker secrets (for Docker Swarm)
- External secret management (AWS Secrets Manager, etc.)

### Logging

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

- Prevents disk space issues
- Rotates logs automatically
- Configurable retention

### Volumes

**Development:**
```yaml
volumes:
  mongodb_data:
    driver: local
```

**Production:**
- Use named volumes or external storage
- Consider backup strategies
- Use volume drivers for cloud storage

## Building for Production

### Build Arguments

```dockerfile
ARG APP_USER=appuser
ARG APP_UID=1000
ARG APP_GID=1000
```

Allows customization without modifying Dockerfile:
```bash
docker build --build-arg APP_UID=2000 -t myapp .
```

### Image Labels

```dockerfile
LABEL org.opencontainers.image.title="MDB_RUNTIME Hello World"
```

- Metadata for image management
- Traceability (build date, git commit)
- Useful for compliance and auditing

## Monitoring and Observability

### Health Checks

Every service should have a health check:
- App: Python process check
- MongoDB: Database ping

### Resource Monitoring

```yaml
deploy:
  resources:
    limits:
      cpus: '1.0'
      memory: 512M
```

Monitor actual usage and adjust:
```bash
docker stats hello_world_app
```

## Scaling Considerations

### Horizontal Scaling

For multiple app instances:
```yaml
deploy:
  replicas: 3
```

All instances share the same MongoDB, with automatic app_id scoping.

### Vertical Scaling

Adjust resource limits based on load:
```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 4G
```

## Security Checklist

- [x] Non-root user
- [x] Minimal base image
- [x] No secrets in images
- [x] Health checks configured
- [x] Resource limits set
- [x] Network isolation
- [x] Read-only filesystem where possible
- [x] Log rotation configured
- [x] Dependencies pinned (via pyproject.toml)
- [x] Multi-stage build

## Performance Tips

1. **Use BuildKit** for faster builds:
   ```bash
   DOCKER_BUILDKIT=1 docker-compose build
   ```

2. **Layer caching**: Structure Dockerfile for optimal caching

3. **Parallel builds**: Build multiple services in parallel

4. **Image size**: Multi-stage builds reduce final image size

5. **Startup time**: Health checks prevent premature connections

## Troubleshooting

### View Build Process
```bash
docker-compose build --progress=plain
```

### Inspect Image Layers
```bash
docker history hello_world_app
```

### Check Resource Usage
```bash
docker stats
```

### View Logs
```bash
docker-compose logs -f app
```

## Next Steps

- Add CI/CD pipeline with Docker builds
- Set up image scanning (Trivy, Snyk)
- Configure monitoring (Prometheus, Grafana)
- Set up log aggregation (ELK, Loki)
- Implement backup strategies for volumes

