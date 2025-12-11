#!/usr/bin/env python3
"""
LLaVA-Med v1.5 Model Setup Script

Downloads and tests microsoft/llava-med-v1.5-mistral-7b
Optimized for M4 Pro 24GB with MPS (Metal Performance Shaders)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Model configuration
MODEL_ID = "microsoft/llava-med-v1.5-mistral-7b"
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


def get_device() -> str:
    """Determine best available device (MPS for M4 Pro, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def download_model(use_4bit: bool = True) -> tuple[LlavaForConditionalGeneration, AutoProcessor]:
    """
    Download and load LLaVA-Med model with optimal quantization.

    Args:
        use_4bit: Use 4-bit quantization to reduce VRAM (recommended for 24GB)

    Returns:
        Tuple of (model, processor)
    """
    device = get_device()
    logger.info(f"Using device: {device}")

    logger.info(f"Downloading LLaVA-Med from {MODEL_ID}...")
    logger.info("This may take 10-20 minutes depending on connection speed")

    # Load processor (tokenizer + image processor)
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

    # Load model with optimizations
    if use_4bit and device != "mps":
        # BitsAndBytes 4-bit quantization (CUDA only)
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
            cache_dir=CACHE_DIR,
        )
    else:
        # MPS or CPU: Use float16 for efficiency
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, cache_dir=CACHE_DIR
        )
        if device == "mps":
            model = model.to("mps")

    logger.info("✓ Model downloaded and loaded successfully")
    return model, processor


def test_model(model: LlavaForConditionalGeneration, processor: AutoProcessor) -> None:
    """
    Run a simple test inference to verify model works.

    Args:
        model: Loaded LLaVA-Med model
        processor: Model processor
    """
    from PIL import Image

    logger.info("Running test inference...")

    # Create a dummy image (black square)
    test_image = Image.new("RGB", (224, 224), color="black")

    # Medical analysis prompt
    prompt = "USER: <image>\nDescribe any pathological findings in this chest X-ray.\nASSISTANT:"

    # Process inputs
    inputs = processor(text=prompt, images=test_image, return_tensors="pt")

    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    start_time = time.time()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False, temperature=0.0)

    elapsed = time.time() - start_time

    # Decode output
    output_text = processor.decode(output_ids[0], skip_special_tokens=True)

    logger.info(f"✓ Test inference completed in {elapsed:.2f}s")
    logger.info(f"Output: {output_text[:200]}...")


def get_model_info(model: LlavaForConditionalGeneration) -> dict[str, any]:
    """
    Get model size and memory usage information.

    Args:
        model: Loaded model

    Returns:
        Dictionary with model statistics
    """
    num_params = sum(p.numel() for p in model.parameters())
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate memory usage (rough approximation)
    param_size_mb = num_params * 2 / (1024**2)  # float16 = 2 bytes

    return {
        "total_parameters": num_params,
        "trainable_parameters": num_params_trainable,
        "estimated_size_mb": param_size_mb,
        "device": str(next(model.parameters()).device),
    }


def main() -> None:
    """Main setup script."""
    parser = argparse.ArgumentParser(description="Setup LLaVA-Med model")
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (use float16 instead)",
    )
    parser.add_argument("--skip-test", action="store_true", help="Skip test inference")
    args = parser.parse_args()

    try:
        # Download model
        model, processor = download_model(use_4bit=not args.no_4bit)

        # Get model info
        info = get_model_info(model)
        logger.info("Model Information:")
        logger.info(f"  Total parameters: {info['total_parameters']:,}")
        logger.info(f"  Estimated size: {info['estimated_size_mb']:.1f} MB")
        logger.info(f"  Device: {info['device']}")

        # Test model
        if not args.skip_test:
            test_model(model, processor)

        logger.info("✓ LLaVA-Med setup complete!")
        logger.info(f"Model cached at: {CACHE_DIR}")

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
