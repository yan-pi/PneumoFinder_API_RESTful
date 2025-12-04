"""Configuration settings for PneumoFinder API."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration."""

    # Model paths
    cnn_model_path: str = "models/pneumonia_model.keras"

    # Directories
    temp_dir: str = "temp"
    uploads_dir: str = "imgs_pulmoes"
    database_dir: str = "database"
    vector_db_dir: str = "database/vectors"

    # Model settings
    image_size: tuple[int, int] = (224, 224)
    prediction_threshold: float = 0.5
    model_version: str = "1.0"

    # LLM settings
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llava:7b"
    llm_max_tokens: int = 500
    llm_temperature: float = 0.3
    llm_enabled: bool = True

    # ChromaDB settings (for Docker deployment)
    chroma_host: str = "localhost"
    chroma_port: int = 8000

    # Vector search configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_collection: str = "clinical_descriptions"

    # Prompts
    medical_prompt_path: str = "prompts/medical_analysis.txt"

    # API settings
    api_port: int = 5001
    debug_mode: bool = True

    @classmethod
    def from_env(cls) -> "Config":
        """
        Load configuration from environment variables.

        Returns:
            Config instance with values from environment
        """
        # Handle OLLAMA_BASE_URL for Docker compatibility
        ollama_host = os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_HOST", cls.ollama_host))

        return cls(
            cnn_model_path=os.getenv("CNN_MODEL_PATH", cls.cnn_model_path),
            temp_dir=os.getenv("TEMP_DIR", cls.temp_dir),
            ollama_host=ollama_host,
            ollama_model=os.getenv("OLLAMA_MODEL", cls.ollama_model),
            llm_enabled=os.getenv("LLM_ENABLED", "true").lower() == "true",
            chroma_host=os.getenv("CHROMA_HOST", cls.chroma_host),
            chroma_port=int(os.getenv("CHROMA_PORT", str(cls.chroma_port))),
            api_port=int(os.getenv("API_PORT", str(cls.api_port))),
            debug_mode=os.getenv("FLASK_DEBUG", os.getenv("DEBUG_MODE", "true")).lower()
            in ("true", "1"),
        )


# Global config instance
config = Config.from_env()
