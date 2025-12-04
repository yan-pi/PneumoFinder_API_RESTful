# Architecture Documentation

## ✅ Current Implementation (Post-Refactoring)

### Architecture Overview
The PneumoFinder API follows a **functional programming approach** with clear separation of concerns. The codebase is organized into distinct modules with pure functions wherever possible.

### Directory Structure
```
src/
├── api/
│   └── app.py                      # Flask routes and HTTP handling
├── core/
│   ├── diagnosis.py                # CNN inference functions
│   ├── visualization.py            # Grad-CAM generation functions
│   └── clinical_description.py     # LLM integration functions
├── db/
│   ├── database.py                 # SQLite connection management
│   ├── repositories.py             # CRUD operations + deduplication
│   └── vector_store.py             # ChromaDB vector search
├── utils/
│   ├── config.py                   # Configuration management
│   ├── file_utils.py               # File handling utilities
│   └── image_utils.py              # Image processing utilities
└── bots/
    └── whatsapp_bot.py             # WhatsApp webhook integration
```

### API Endpoints

#### Primary Endpoints (English)
- **`POST /diagnose`** - Simple pneumonia diagnosis (returns diagnosis_id, diagnosis, confidence)
- **`POST /diagnose/explained`** - Complete diagnosis with Grad-CAM + LLM explanation
- **`GET /health`** - Health check

#### Database Endpoints
- **`GET /api/diagnoses/<id>`** - Retrieve diagnosis by ID (includes heatmap/overlay in base64)
- **`POST /api/search/similar`** - Semantic search for similar diagnoses
- **`GET /api/diagnoses/recent`** - List recent diagnoses

#### Legacy Endpoints (Portuguese - Backward Compatibility)
These endpoints redirect to the primary endpoints:
- `POST /diagnosticar_pneumonia` → redirects to `/diagnose`
- `POST /diagnostico_completo` → redirects to `/diagnose`
- `POST /diagnosticar_com_descricao` → redirects to `/diagnose/explained`

#### WhatsApp Integration
- **`POST /webhook`** - Twilio webhook for WhatsApp messages

### Key Design Principles

1. **Functional Programming**
   - Pure functions for image processing, diagnosis, and visualization
   - Stateless operations wherever possible
   - Composable functions for complex workflows
   - Minimal side effects (isolated to I/O operations)

2. **Separation of Concerns**
   - `diagnosis.py`: CNN model loading and inference only
   - `visualization.py`: Grad-CAM generation only
   - `clinical_description.py`: LLM API integration only
   - `image_utils.py`: Image transformations only
   - `file_utils.py`: File operations only
   - `database.py`: Database connection management
   - `repositories.py`: CRUD operations and business logic
   - `vector_store.py`: Vector embeddings and semantic search

3. **Configuration Management**
   - All configuration centralized in `src/utils/config.py`
   - Environment variable support via `.env`
   - Easy modification without touching code
   - No hardcoded paths

4. **Code Quality**
   - All code passes `ruff` linting (configured in pyproject.toml)
   - Type hints throughout the codebase
   - Clear docstrings with parameter descriptions
   - Consistent naming conventions (snake_case for functions, PascalCase for classes)
   - English-only codebase for international collaboration

## 🧪 Testing

### Verified Components
- ✅ Configuration loading from `config.py`
- ✅ CNN model loading (ResNet50 base + last convolutional layer extraction)
- ✅ Simple diagnosis workflow: `diagnose_from_path()`
- ✅ API module initialization
- ✅ All endpoints registered correctly (primary + legacy + database)
- ✅ Database initialization (SQLite + ChromaDB)
- ✅ Image deduplication with SHA-256 hashing
- ✅ Semantic search with sentence-transformers embeddings
- ✅ BLOB storage for heatmap/overlay visualizations

### Example Workflow
```python
from src.core.diagnosis import load_cnn_model, diagnose_from_path
from src.utils.config import config

# Load model once at startup
model = load_cnn_model(config.cnn_model_path)

# Perform diagnosis
diagnosis, confidence = diagnose_from_path(model, "radiografia.jpg")
# Output: ("PNEUMONIA", 0.999) or ("NORMAL", 0.876)
```

### Database Operations Example
```python
from src.db.repositories import DiagnosisRepository
from src.db.database import get_db_connection

conn = get_db_connection()
repo = DiagnosisRepository(conn)

# Save diagnosis (with automatic deduplication)
diagnosis_id = repo.create_diagnosis(
    image_hash="a3f2b1c4...",
    diagnosis="PNEUMONIA",
    confidence=0.87,
    clinical_description="...",
    heatmap_blob=heatmap_bytes,
    overlay_blob=overlay_bytes
)

# Retrieve diagnosis
diagnosis_data = repo.get_diagnosis(diagnosis_id)
```

## 📝 Running the Application

### Main API Server
```bash
# Run with mise
mise run run

# Or directly with uv
uv run python app.py

# API will be available at http://localhost:5001
```

### WhatsApp Bot Integration
The WhatsApp bot is integrated directly into the main API via the `/webhook` endpoint. Configure Twilio webhook URL to point to `https://your-domain.com/webhook`.

**Configuration:**
- Add Twilio credentials to `.env`:
  ```env
  TWILIO_ACCOUNT_SID=your_sid
  TWILIO_AUTH_TOKEN=your_token
  ```

### Database Initialization
The database (SQLite + ChromaDB) is initialized automatically on first API startup:
- SQLite database created at `database/pneumofinder.db`
- ChromaDB vector store created at `database/vectors/`
- No manual setup required

## 🔄 Request Flow

### Simple Diagnosis Flow (`POST /diagnose`)
```
1. Client sends POST with image file
2. Flask receives multipart/form-data
3. Save image to temp/ directory
4. Calculate SHA-256 hash of image
5. Check database for existing diagnosis (deduplication)
6. If exists → return cached diagnosis_id
7. If not exists:
   - Load image with image_utils
   - Preprocess with ResNet50 preprocessing
   - Run CNN inference (diagnosis.py)
   - Save to database (repositories.py)
   - Return JSON: {diagnosis_id, diagnosis, confidence}
8. Clean up temp file
```

### Complete Diagnosis with Explanation (`POST /diagnose/explained`)
```
1-7. Same as simple diagnosis flow
8. Generate Grad-CAM heatmap (visualization.py)
9. Generate overlay (heatmap + original image)
10. Generate clinical description with LLaVA (clinical_description.py)
11. Save visualizations as BLOBs in database
12. Save clinical description to ChromaDB for semantic search
13. Return JSON with:
    - diagnosis_id, diagnosis, confidence
    - clinical_description
    - visualizations.heatmap (base64)
    - visualizations.overlay (base64)
14. Clean up temp files
```

### Semantic Search Flow (`POST /api/search/similar`)
```
1. Client sends query text (e.g., "infiltrates in lower lung field")
2. Generate embedding with sentence-transformers (384-dim)
3. Query ChromaDB for top_k similar vectors
4. Retrieve diagnosis records from SQLite
5. Return ranked results with similarity scores
```

### Deduplication Flow
```
Image → SHA-256 hash → Check DB → If exists: return cached result
                                → If not exists: process + save
```

## 🏗️ Technology Stack

### Backend Framework
- **Flask 3.1.0** - Lightweight web framework for REST API
- **Flask-CORS** - Cross-origin resource sharing support

### Machine Learning
- **TensorFlow 2.19.0** - Deep learning framework
- **Keras** - High-level neural networks API (integrated with TensorFlow)
- **ResNet50** - Pre-trained CNN architecture (transfer learning base)

### Database Layer
- **SQLite** - Lightweight relational database for structured data
- **ChromaDB** - Vector database for semantic search
- **sentence-transformers** - Generate 384-dim embeddings for similarity search

### Computer Vision
- **OpenCV (cv2)** - Grad-CAM visualization generation
- **Pillow (PIL)** - Image processing and manipulation
- **NumPy** - Numerical operations on arrays

### LLM Integration
- **Ollama** - Local LLM inference server
- **LLaVA 7B** - Multimodal language model (vision + text)
- **requests** - HTTP client for Ollama API calls

### External Integrations
- **Twilio API** - WhatsApp webhook integration

### Development Tools
- **mise** - Development environment management (Python 3.11)
- **uv** - Fast Python package installer
- **ruff** - Linter and code formatter
- **python-dotenv** - Environment variable management

## 🚀 Future Improvements

### Potential Enhancements

1. **Testing Suite**
   - Unit tests for pure functions (diagnosis, visualization, utils)
   - Integration tests for API endpoints
   - End-to-end tests for complete workflows

2. **Performance Optimization**
   - Model quantization for faster inference
   - Async processing for Grad-CAM generation
   - Caching layer for frequently accessed diagnoses

3. **Feature Additions**
   - Multi-model ensemble for improved accuracy
   - Support for additional lung pathologies (tuberculosis, COVID-19, etc.)
   - Batch processing endpoint for multiple images
   - User authentication and authorization

4. **Deployment**
   - Docker containerization
   - Kubernetes deployment configuration
   - CI/CD pipeline setup
   - Production-grade monitoring and logging

5. **Documentation**
   - OpenAPI/Swagger documentation
   - Interactive API documentation
   - Deployment guides

## 💡 Benefits of Current Architecture

1. **Readability**: Functions are small, focused, and do one thing well
2. **Testability**: Pure functions are easy to test in isolation
3. **Maintainability**: Changes to one module don't affect others
4. **Extensibility**: Easy to add new features (models, endpoints, data sources)
5. **Performance**: Models loaded once at startup (not per-request)
6. **International**: English codebase enables global collaboration
7. **Simplicity**: No unnecessary abstractions, just clean functional code
8. **Scalability**: Database layer supports growth with deduplication and semantic search
