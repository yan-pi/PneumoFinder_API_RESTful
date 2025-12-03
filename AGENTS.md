# Agent Guidelines for PneumoFinder API

## Prerequisites
- Install [mise](https://mise.jdx.dev/) for tool management
- mise will handle Python 3.11 and uv installation automatically

## Build/Run Commands
- **Setup environment**: `mise install` (installs Python 3.11 + creates .venv)
- **Install dependencies**: `mise run install` or `uv sync`
- **Run API**: `mise run run` or `uv run python app.py` (port 5001)
- **Run WhatsApp bot**: `mise run bot` or `uv run python service/chat_bot_service.py`
- **Test model**: `mise run test-model` or `uv run python testar_modelo.py`
- **Lint code**: `mise run lint` or `uv run ruff check .`
- **Format code**: `mise run format` or `uv run ruff format .`
- **Check formatting**: `mise run format-check` or `uv run ruff format --check .`

## Code Style
- **Language**: Python 3.11 with Flask, TensorFlow/Keras, managed by mise + uv
- **Formatting**: Use ruff for linting and formatting (configured in pyproject.toml)
- **Imports**: Standard libs first, then third-party (Flask, TensorFlow, etc.), then local (`from service import ...`)
- **Naming**: snake_case for functions/variables, PascalCase for classes (e.g., `DetectorDePneumonia`)
- **Portuguese**: Variable/function names in Portuguese (e.g., `diagnosticar_imagem`, `classe`, `confianca`)
- **Error handling**: Try-except blocks return JSON errors with 400/500 status codes
- **File handling**: Always clean up temp files with `os.remove()` after processing
- **Image preprocessing**: Normalize to 0-1 range or use ResNet50 preprocessing, target size 224x224
- **Models**: Load once at startup, stored in `models/` directory (.keras format)
- **Endpoints**: POST methods with multipart/form-data for image uploads
- **Environment**: Use `python-dotenv` for Twilio credentials in `.env`
- **Line length**: 100 characters max (enforced by ruff)

## Architecture
- Flask API (`app.py`) with 3 main endpoints: `/verificar_pulmao`, `/diagnosticar_pneumonia`, `/diagnostico_completo`
- Service layer pattern: `DetectorDePulmao` and `DetectorDePneumonia` classes handle ML logic
- Twilio webhook integration for WhatsApp bot (`service/chat_bot_service.py`)
- Temp files stored in `temp/` directory, created on startup
