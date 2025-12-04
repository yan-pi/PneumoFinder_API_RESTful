"""Configuration settings for PneumoFinder API."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration."""

    # Model paths
    cnn_model_path: str = "models/pneumonia_model.keras"
    lung_model_path: str = "models/pulmao_model.keras"

    # Directories
    temp_dir: str = "temp"
    reports_dir: str = "relatorios"
    uploads_dir: str = "imgs_pulmoes"

    # Model settings
    image_size: tuple[int, int] = (224, 224)
    prediction_threshold: float = 0.5

    # LLM settings
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llava:7b"
    llm_max_tokens: int = 500
    llm_temperature: float = 0.3
    llm_enabled: bool = True

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
        return cls(
            cnn_model_path=os.getenv("CNN_MODEL_PATH", cls.cnn_model_path),
            lung_model_path=os.getenv("LUNG_MODEL_PATH", cls.lung_model_path),
            temp_dir=os.getenv("TEMP_DIR", cls.temp_dir),
            ollama_host=os.getenv("OLLAMA_HOST", cls.ollama_host),
            ollama_model=os.getenv("OLLAMA_MODEL", cls.ollama_model),
            llm_enabled=os.getenv("LLM_ENABLED", "true").lower() == "true",
            api_port=int(os.getenv("API_PORT", str(cls.api_port))),
            debug_mode=os.getenv("DEBUG_MODE", "true").lower() == "true",
        )


# Global config instance
config = Config.from_env()
