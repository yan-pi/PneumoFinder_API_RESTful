#!/usr/bin/env python3
"""
BioMistral-7B Model Setup Script

Downloads and tests BioMistral/BioMistral-7B
Medical English specialist trained on PubMed abstracts
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Model configuration
MODEL_ID = "BioMistral/BioMistral-7B"
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


def get_device() -> str:
    """Determine best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def download_model(use_4bit: bool = True) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Download and load BioMistral model.

    Args:
        use_4bit: Use 4-bit quantization (recommended)

    Returns:
        Tuple of (model, tokenizer)
    """
    device = get_device()
    logger.info(f"Using device: {device}")

    logger.info(f"Downloading BioMistral from {MODEL_ID}...")
    logger.info("This may take 10-20 minutes depending on connection speed")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

    # Load model with optimizations
    if use_4bit and device != "mps":
        # BitsAndBytes 4-bit quantization
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
            cache_dir=CACHE_DIR,
        )
    else:
        # MPS: Use float16
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, cache_dir=CACHE_DIR
        )
        if device == "mps":
            model = model.to("mps")

    logger.info("✓ Model downloaded and loaded successfully")
    return model, tokenizer


def test_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
    """
    Test model with medical terminology task.

    Args:
        model: Loaded BioMistral model
        tokenizer: Model tokenizer
    """
    logger.info("Running test inference...")

    # Medical test prompt
    prompt = """Question: What are the primary radiological findings of bacterial pneumonia on chest X-ray?

Answer:"""

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    start_time = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.time() - start_time

    # Decode
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    logger.info(f"✓ Test inference completed in {elapsed:.2f}s")
    logger.info(f"Output:\n{output_text}")


def get_model_info(model: AutoModelForCausalLM) -> dict[str, any]:
    """Get model statistics."""
    num_params = sum(p.numel() for p in model.parameters())
    param_size_mb = num_params * 2 / (1024**2)

    return {
        "total_parameters": num_params,
        "estimated_size_mb": param_size_mb,
        "device": str(next(model.parameters()).device),
    }


def main() -> None:
    """Main setup script."""
    parser = argparse.ArgumentParser(description="Setup BioMistral model")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--skip-test", action="store_true", help="Skip test inference")
    args = parser.parse_args()

    try:
        # Download model
        model, tokenizer = download_model(use_4bit=not args.no_4bit)

        # Get info
        info = get_model_info(model)
        logger.info("Model Information:")
        logger.info(f"  Total parameters: {info['total_parameters']:,}")
        logger.info(f"  Estimated size: {info['estimated_size_mb']:.1f} MB")
        logger.info(f"  Device: {info['device']}")

        # Test
        if not args.skip_test:
            test_model(model, tokenizer)

        logger.info("✓ BioMistral setup complete!")
        logger.info(f"Model cached at: {CACHE_DIR}")

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
