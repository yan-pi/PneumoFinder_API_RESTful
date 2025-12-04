"""Main entry point for PneumoFinder API."""

# Run the refactored app from src/api/app.py
if __name__ == "__main__":
    from src.api.app import app, config, ensure_directory

    # Ensure temp directory exists
    ensure_directory(config.temp_dir)

    # Run app
    print(f"\n🚀 Starting PneumoFinder API on port {config.api_port}...")
    print(f"   - Simple diagnosis: POST /diagnose")
    print(f"   - With explanation: POST /diagnose/explained")
    print(f"   - Health check: GET /health")
    print(f"\n   Legacy endpoints (backward compatibility):")
    print(f"   - POST /diagnosticar_pneumonia")
    print(f"   - POST /diagnostico_completo")
    print(f"   - POST /diagnosticar_com_descricao\n")

    app.run(debug=config.debug_mode, port=config.api_port)
