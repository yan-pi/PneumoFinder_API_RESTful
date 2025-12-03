# Development Environment Setup

## Prerequisites Installed ✓
- Python 3.11.14 (via mise)
- uv package manager
- All dependencies (Flask, TensorFlow, Keras, OpenCV, Twilio, etc.)
- ruff linter/formatter

## Project Structure
```
PneumoFinder_API_RESTful/
├── models/              # ML model files (.keras format)
│   ├── pneumonia_model.keras  # ⚠️ Required - pneumonia detection model
│   └── pulmao_model.keras     # ⚠️ Required - lung detection model
├── temp/                # Auto-created - temp storage for uploaded images
├── service/             # Service layer (ML logic)
│   ├── pneumonia_service.py   # DetectorDePneumoniaService
│   ├── pulmao_service.py      # DetectorDePulmao
│   └── chat_bot_service.py    # WhatsApp bot integration
├── app.py               # Main Flask API (port 5001)
├── .env                 # ⚠️ Required for WhatsApp bot - Twilio credentials
├── .env.example         # Template for .env
└── pyproject.toml       # Dependency configuration
```

## Quick Start

### 1. First-Time Setup
```bash
# Clone and enter the project
cd /Users/ybarbara/www/tcc/PneumoFinder_API_RESTful

# Install mise (if not installed)
# On macOS: brew install mise
# Or follow: https://mise.jdx.dev/getting-started.html

# Trust and install Python + dependencies
mise trust && mise install
mise run install
```

### 2. Add Model Files (Required!)
Place your trained Keras models in the `models/` directory:
- `models/pneumonia_model.keras` - Pneumonia detection (ResNet50-based)
- `models/pulmao_model.keras` - Lung detection

**The API will NOT start without these files!**

### 3. Configure Environment (Optional - for WhatsApp bot only)
```bash
# Copy template
cp .env.example .env

# Edit .env and add your Twilio credentials:
# TWILIO_ACCOUNT_SID=your_actual_sid
# TWILIO_AUTH_TOKEN=your_actual_token
```

## Development Commands

### Running the API
```bash
mise run run                # Start Flask API on http://localhost:5001
mise run bot                # Start WhatsApp bot service
mise run test-model         # Test model with sample images
```

### Code Quality
```bash
mise run lint               # Check code with ruff
mise run format             # Auto-format code with ruff
mise run format-check       # Verify formatting without changes
```

### API Endpoints
Once running, the API provides three endpoints:

1. **POST /verificar_pulmao** - Check if image is a lung X-ray
   - Input: multipart/form-data with "imagem" file
   - Returns: `{"classe": "PULMÃO" | "NÃO É PULMÃO", "confianca": 0.95}`

2. **POST /diagnosticar_pneumonia** - Diagnose pneumonia from X-ray
   - Input: multipart/form-data with "imagem" file
   - Returns: `{"classe": "PNEUMONIA" | "NORMAL", "confianca": 0.87}`

3. **POST /diagnostico_completo** - Full pipeline (lung check + pneumonia diagnosis)
   - Input: multipart/form-data with "imagem" file
   - Returns: Full diagnosis with both classifications

## Development Workflow

### Making Changes
1. Activate environment: `mise exec -- python`
2. Make your changes to Python files
3. Run formatter: `mise run format`
4. Check for issues: `mise run lint`
5. Test locally: `mise run run`

### Testing with cURL
```bash
# Test lung detection
curl -X POST -F "imagem=@imgs/1_normal1.jpeg" http://localhost:5001/verificar_pulmao

# Test pneumonia diagnosis  
curl -X POST -F "imagem=@imgs/3_pneumonia1.jpeg" http://localhost:5001/diagnosticar_pneumonia

# Test complete pipeline
curl -X POST -F "imagem=@imgs/4_pneumonia2.jpeg" http://localhost:5001/diagnostico_completo
```

### Code Style Guidelines
- Follow ruff rules (configured in `pyproject.toml`)
- Portuguese naming conventions are allowed (N802-N816 rules ignored)
- Max line length: 100 characters
- Use double quotes for strings
- Target Python 3.11

## Troubleshooting

### "Module not found" errors
```bash
# Reinstall dependencies
mise run install
```

### Models not loading
```bash
# Verify models exist
ls -la models/

# Should show:
# - pneumonia_model.keras
# - pulmao_model.keras
```

### Port 5001 already in use
```bash
# Find and kill process using port 5001
lsof -ti:5001 | xargs kill -9
```

### Twilio errors (WhatsApp bot)
- Verify `.env` file exists with correct credentials
- Check `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` are set
- WhatsApp bot only works with valid Twilio account

## Next Steps
- Add your trained models to `models/` directory
- Test API endpoints with sample images from `imgs/` directory
- Review `AGENTS.md` for AI agent collaboration guidelines
- See `MIGRATION.md` for details on the modern tooling migration
