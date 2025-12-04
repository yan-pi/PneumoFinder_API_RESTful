# Docker Deployment Guide

## Overview

PneumoFinder can be deployed using Docker for simplified dependency management. The Docker setup includes:

- **Flask API** (PneumoFinder) - Main application
- **ChromaDB** - Vector database for semantic search (containerized)
- **SQLite** - Relational database (volume-mounted)
- **Ollama** - LLM server (runs on host machine for GPU access)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Host Machine                          │
│                                                              │
│  ┌──────────────────────┐      ┌──────────────────────────┐ │
│  │   Ollama (Host)      │      │   Docker Containers      │ │
│  │   Port: 11434        │◄─────┤                          │ │
│  │   + GPU Access       │      │  ┌───────────────────┐   │ │
│  │   + macOS/NVIDIA     │      │  │  PneumoFinder API │   │ │
│  └──────────────────────┘      │  │  Port: 5001       │   │ │
│                                │  └─────────┬─────────┘   │ │
│                                │            │             │ │
│                                │  ┌─────────▼─────────┐   │ │
│                                │  │    ChromaDB       │   │ │
│                                │  │    Port: 8000     │   │ │
│                                │  └───────────────────┘   │ │
│                                └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Docker & Docker Compose**
   ```bash
   # macOS (using Homebrew)
   brew install docker docker-compose
   
   # Or install Docker Desktop from https://www.docker.com/products/docker-desktop
   ```

2. **Ollama** (runs on host for GPU access)
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Pull LLaVA model
   ollama pull llava:7b
   
   # Start Ollama server
   ollama serve
   ```

3. **Model File**
   - Ensure `models/pneumonia_model.keras` exists in the project root
   - This file is mounted read-only into the container

## Quick Start

### 1. Environment Setup (Optional)

Create a `.env` file for Twilio credentials (WhatsApp integration):

```bash
cat > .env << EOF
TWILIO_ACCOUNT_SID=your_sid_here
TWILIO_AUTH_TOKEN=your_token_here
EOF
```

### 2. Build and Start Services

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

### 3. Verify Services

```bash
# Check service health
docker-compose ps

# View logs
docker-compose logs -f api

# Test API
curl http://localhost:5001/health
```

### 4. Test Diagnosis

```bash
curl -X POST http://localhost:5001/diagnose \
  -F "image=@path/to/chest_xray.jpg"
```

## Docker Commands

### Service Management

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart a specific service
docker-compose restart api

# View logs
docker-compose logs -f api
docker-compose logs -f chromadb

# Execute commands inside container
docker-compose exec api bash
```

### Data Management

```bash
# Backup database
docker cp pneumofinder-api:/app/database/pneumofinder.db ./backup-$(date +%Y%m%d).db

# View volumes
docker volume ls | grep pneumofinder

# Remove volumes (⚠️ deletes all data)
docker-compose down -v
```

### Rebuild

```bash
# Rebuild after code changes
docker-compose up --build

# Force rebuild (no cache)
docker-compose build --no-cache
```

## Configuration

### Environment Variables

The following environment variables can be configured in `docker-compose.yml`:

#### Flask API

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | `production` | Flask environment |
| `FLASK_DEBUG` | `0` | Debug mode (0=off, 1=on) |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama server URL |
| `CHROMA_HOST` | `chromadb` | ChromaDB hostname |
| `CHROMA_PORT` | `8000` | ChromaDB port |
| `TWILIO_ACCOUNT_SID` | - | Twilio account SID (optional) |
| `TWILIO_AUTH_TOKEN` | - | Twilio auth token (optional) |

#### ChromaDB

| Variable | Default | Description |
|----------|---------|-------------|
| `IS_PERSISTENT` | `TRUE` | Enable data persistence |
| `ANONYMIZED_TELEMETRY` | `FALSE` | Disable telemetry |

### Volumes

```yaml
volumes:
  - ./models:/app/models:ro          # Model files (read-only)
  - api-database:/app/database       # SQLite database (persistent)
  - api-temp:/app/temp               # Temporary uploads
  - chroma-data:/chroma/chroma       # ChromaDB data (persistent)
```

### Ports

- **5001** - PneumoFinder API
- **8000** - ChromaDB (optional, for debugging)

## Troubleshooting

### API Can't Connect to Ollama

**Symptom:** Errors about Ollama connection refused

**Solution:**
```bash
# 1. Ensure Ollama is running on host
ollama serve

# 2. Test from inside container
docker-compose exec api curl http://host.docker.internal:11434

# 3. On Linux, use host network instead:
# Edit docker-compose.yml and change:
# extra_hosts:
#   - "host.docker.internal:172.17.0.1"  # Docker bridge IP
```

### ChromaDB Connection Issues

**Symptom:** API can't connect to ChromaDB

**Solution:**
```bash
# 1. Check ChromaDB health
docker-compose exec chromadb curl http://localhost:8000/api/v1/heartbeat

# 2. View ChromaDB logs
docker-compose logs chromadb

# 3. Restart ChromaDB
docker-compose restart chromadb
```

### Model Not Found

**Symptom:** `FileNotFoundError: models/pneumonia_model.keras`

**Solution:**
```bash
# Ensure model file exists on host
ls -lh models/pneumonia_model.keras

# Check if mounted correctly in container
docker-compose exec api ls -lh /app/models/
```

### Permission Denied

**Symptom:** Permission errors writing to database/temp directories

**Solution:**
```bash
# Adjust volume permissions
docker-compose exec -u root api chown -R pneumofinder:pneumofinder /app/database /app/temp

# Or recreate volumes
docker-compose down -v
docker-compose up -d
```

### Out of Memory

**Symptom:** Container killed due to OOM

**Solution:**
```yaml
# Add memory limits in docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

## Production Deployment

### Recommended Changes

1. **Use Production WSGI Server**
   ```dockerfile
   # In Dockerfile, change:
   CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "4", "--timeout", "120", "app:app"]
   ```

2. **Disable Debug Mode**
   ```yaml
   environment:
     - FLASK_DEBUG=0
     - FLASK_ENV=production
   ```

3. **Add Nginx Reverse Proxy**
   ```yaml
   nginx:
     image: nginx:alpine
     ports:
       - "80:80"
       - "443:443"
     volumes:
       - ./nginx.conf:/etc/nginx/nginx.conf:ro
       - ./certs:/etc/nginx/certs:ro
   ```

4. **Enable HTTPS**
   - Use Let's Encrypt with certbot
   - Mount SSL certificates into Nginx

5. **Resource Limits**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
   ```

6. **Health Monitoring**
   - Set up Prometheus + Grafana
   - Configure log aggregation (ELK stack)

## Development

### Local Development (without Docker)

For development with hot-reload:

```bash
# Use mise (recommended)
mise install
mise run install
mise run run

# Or manual setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

### Hybrid Setup (ChromaDB in Docker, API local)

```bash
# Start only ChromaDB
docker-compose up -d chromadb

# Run API locally
export CHROMA_HOST=localhost
export CHROMA_PORT=8000
mise run run
```

## Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove containers + volumes (⚠️ deletes all data)
docker-compose down -v

# Remove images
docker rmi pneumofinder-api chromadb/chroma

# Clean build cache
docker builder prune -a
```

## Security Notes

1. **Never commit `.env` files** with real credentials
2. **Change default ports** in production
3. **Use secrets management** (Docker secrets, Vault, etc.)
4. **Enable firewall** rules to restrict access
5. **Regular updates** of base images and dependencies
6. **Scan images** for vulnerabilities (`docker scan`)

## Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [ChromaDB Docker Guide](https://docs.trychroma.com/deployment/docker)
- [Ollama Documentation](https://ollama.ai/docs)
