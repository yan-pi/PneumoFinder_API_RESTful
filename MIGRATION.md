# Migration to mise + uv + ruff - Summary

## What Changed

### New Files Created
1. **`.mise.toml`** - Manages Python 3.11 version and defines tasks
2. **`pyproject.toml`** - Modern Python project configuration with:
   - All dependencies from requirements.txt
   - uv configuration
   - ruff linting/formatting rules (respects Portuguese naming)
   - Project metadata

### Updated Files
1. **`.gitignore`** - Added mise/uv artifacts (.venv/, uv.lock, .ruff_cache/)
2. **`AGENTS.md`** - Updated with mise/uv/ruff commands
3. **`README.md`** - Added modern setup instructions with mise

## Key Benefits

### mise
- **Automatic Python version management**: Ensures Python 3.11 for all developers
- **Task runner**: Simple commands like `mise run run` instead of remembering full paths
- **Automatic venv creation**: No more manual virtualenv setup

### uv
- **10-100x faster** than pip for dependency installation
- **Better dependency resolution**: Prevents conflicts
- **Lock file support**: `uv.lock` ensures reproducible builds
- **Drop-in replacement**: Works with existing requirements

### ruff
- **Lightning fast**: 10-100x faster than pylint/flake8/black combined
- **All-in-one**: Linting + formatting in one tool
- **Configured for project**: Respects Portuguese naming conventions

## Quick Start (New Setup)

```bash
# 1. Install mise
curl https://mise.run | sh

# 2. Setup project (installs Python 3.11 + creates .venv)
mise install

# 3. Install dependencies
mise run install

# 4. Run the API
mise run run
```

## Migration (Existing Setup)

```bash
# Remove old virtualenv
rm -rf venv/

# Setup with mise + uv
mise install
mise run install

# Everything else works the same!
mise run run
```

## Available Commands

| Command | Description |
|---------|-------------|
| `mise run install` | Install dependencies |
| `mise run run` | Run Flask API (port 5001) |
| `mise run bot` | Run WhatsApp bot |
| `mise run test-model` | Test model with samples |
| `mise run lint` | Check code with ruff |
| `mise run format` | Format code with ruff |
| `mise run format-check` | Check formatting without modifying |

## Notes

- **requirements.txt preserved**: For backward compatibility
- **Portuguese naming**: Ruff configured to allow Portuguese variable names
- **Line length**: 100 characters (configured in pyproject.toml)
- **Python 3.11**: Required for TensorFlow 2.19.0 compatibility
