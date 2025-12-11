#!/usr/bin/env python3
"""
Sabiá-7B Model Setup Script

Downloads and tests maritaca-ai/sabia-7b (Brazilian Portuguese specialist)
Tries multiple sources: Ollama, HuggingFace transformers, GGUF
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Model sources
OLLAMA_MODEL = "sabia-7b"
HF_MODEL_ID = "maritaca-ai/sabia-7b"
GGUF_MODEL_ID = "TheBloke/sabia-7b-GGUF"


def check_ollama_available() -> bool:
    """Check if Ollama is installed."""
    try:
        subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            check=True,
            timeout=5,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def setup_via_ollama() -> bool:
    """
    Try to setup Sabiá via Ollama (fastest method).

    Returns:
        True if successful, False otherwise
    """
    logger.info("Attempting to install via Ollama...")

    try:
        # Try to pull model
        result = subprocess.run(
            ["ollama", "pull", OLLAMA_MODEL],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes max
        )

        if result.returncode == 0:
            logger.info("✓ Model installed via Ollama")
            return True
        else:
            logger.warning(f"Ollama pull failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.warning("Ollama pull timed out")
        return False
    except Exception as e:
        logger.warning(f"Ollama installation failed: {e}")
        return False


def test_ollama_model() -> None:
    """Test Ollama model with simple inference."""
    logger.info("Testing Ollama model...")

    prompt = (
        "Descreva os principais achados radiológicos de pneumonia bacteriana em raio-X de tórax."
    )

    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL, prompt],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            logger.info("✓ Test inference successful")
            logger.info(f"Output:\n{result.stdout[:300]}...")
        else:
            logger.error(f"Test failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        logger.error("Test inference timed out")
    except Exception as e:
        logger.error(f"Test failed: {e}")


def setup_via_transformers() -> bool:
    """
    Try to setup via HuggingFace transformers.

    Returns:
        True if successful, False otherwise
    """
    logger.info("Attempting to install via HuggingFace transformers...")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

        logger.info(f"Downloading from {HF_MODEL_ID}...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, cache_dir=cache_dir)

        # Load model with 4-bit if possible
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        if device == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                HF_MODEL_ID, torch_dtype=torch.float16, cache_dir=cache_dir
            )
            model = model.to("mps")
        else:
            model = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID, cache_dir=cache_dir)

        logger.info("✓ Model downloaded via transformers")

        # Quick test
        prompt = "Pneumonia bacteriana apresenta os seguintes achados:"
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=50)

        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logger.info(f"Test output: {output[:200]}...")

        return True

    except ImportError:
        logger.warning("transformers/torch not available")
        return False
    except Exception as e:
        logger.warning(f"HuggingFace installation failed: {e}")
        return False


def main() -> None:
    """Main setup script."""
    parser = argparse.ArgumentParser(description="Setup Sabiá-7B model")
    parser.add_argument(
        "--method",
        choices=["ollama", "transformers", "auto"],
        default="auto",
        help="Installation method",
    )
    parser.add_argument("--skip-test", action="store_true", help="Skip test inference")
    args = parser.parse_args()

    success = False

    if args.method == "auto":
        # Try Ollama first (fastest)
        if check_ollama_available():
            logger.info("Ollama detected, trying Ollama first...")
            success = setup_via_ollama()

        # Fallback to transformers
        if not success:
            logger.info("Falling back to HuggingFace transformers...")
            success = setup_via_transformers()

    elif args.method == "ollama":
        if not check_ollama_available():
            logger.error("Ollama not installed. Install from: https://ollama.ai")
            sys.exit(1)
        success = setup_via_ollama()

    elif args.method == "transformers":
        success = setup_via_transformers()

    if not success:
        logger.error("All installation methods failed")
        logger.error("Options:")
        logger.error("  1. Install Ollama: https://ollama.ai")
        logger.error("  2. Manually download GGUF from TheBloke/sabia-7b-GGUF")
        sys.exit(1)

    # Test model
    if not args.skip_test and args.method in ["ollama", "auto"]:
        if check_ollama_available():
            test_ollama_model()

    logger.info("✓ Sabiá-7B setup complete!")


if __name__ == "__main__":
    main()
