# Agent Guidelines for PneumoFinder API

## Commands
- **Install:** `mise run install` (or `uv sync --extra dev`)
- **Run API:** `mise run run` (starts Flask on port 5001)
- **Lint:** `mise run lint` (ruff check)
- **Format:** `mise run format` (ruff auto-format)
- **Tests:** `uv run pytest tests/` (single test: `uv run pytest tests/test_api.py::test_health`)
- **Docker:** `docker-compose up --build` (API + ChromaDB)

## Code Style
- **Language:** English for all code (functions, variables, comments); Portuguese allowed in UI/docs
- **Formatting:** Ruff with line-length=100, double quotes, 4-space indent
- **Imports:** Sort with isort (E, F, I checks enabled) - stdlib → third-party → local
- **Types:** Use type hints for all functions: `def func(arg: str) -> tuple[str, float]:`
- **Naming:** `snake_case` for functions/variables, `PascalCase` for classes
- **Error Handling:** Specific exceptions with context, avoid bare `except:`
- **Functions:** Pure functions preferred, keep under 50 lines, single responsibility
- **Docstrings:** Google style with Args/Returns sections for all public functions

## Architecture
- **Functional approach:** `src/core/` has pure functions, no classes unless needed
- **Config:** Use `src/utils/config.py`, support env vars (OLLAMA_BASE_URL, CHROMA_HOST)
- **Database:** SQLite (structured) + ChromaDB (vectors), deduplication with SHA-256
- **Never modify:** `models/`, `database/` (except via repositories)
