#!/usr/bin/env python3
"""
Model Benchmarking Script

Tests VRAM usage, inference latency, and output quality for all models:
- LLaVA models (7b, 13b, llama3) via Ollama
- LLaVA-Med via transformers
- BioMistral via transformers
- Sabiá via Ollama/transformers

Optimized for M4 Pro 24GB RAM with MPS acceleration
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Store benchmark results for a single model."""

    model_name: str
    model_type: str  # "vision" or "text"
    backend: str  # "ollama" or "transformers"
    load_time_s: float
    inference_time_s: float
    tokens_per_second: Optional[float]
    memory_mb: Optional[float]
    output_length_chars: int
    success: bool
    error_message: Optional[str] = None


class OllamaBenchmark:
    """Benchmark Ollama models."""

    @staticmethod
    def check_model_exists(model_name: str) -> bool:
        """Check if Ollama model is installed."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return model_name in result.stdout
        except Exception:
            return False

    @staticmethod
    def benchmark_vision_model(model_name: str, image_path: Path, prompt: str) -> BenchmarkResult:
        """Benchmark Ollama vision model (LLaVA variants)."""
        logger.info(f"Benchmarking {model_name} (Ollama vision)...")

        if not OllamaBenchmark.check_model_exists(model_name):
            return BenchmarkResult(
                model_name=model_name,
                model_type="vision",
                backend="ollama",
                load_time_s=0.0,
                inference_time_s=0.0,
                tokens_per_second=None,
                memory_mb=None,
                output_length_chars=0,
                success=False,
                error_message=f"Model {model_name} not installed",
            )

        try:
            # Warm-up (load model into memory)
            start_load = time.time()
            subprocess.run(
                ["ollama", "run", model_name, "test"],
                capture_output=True,
                timeout=60,
            )
            load_time = time.time() - start_load

            # Actual benchmark
            start_inference = time.time()
            result = subprocess.run(
                ["ollama", "run", model_name, prompt, str(image_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            inference_time = time.time() - start_inference

            if result.returncode != 0:
                raise RuntimeError(f"Inference failed: {result.stderr}")

            output = result.stdout.strip()

            return BenchmarkResult(
                model_name=model_name,
                model_type="vision",
                backend="ollama",
                load_time_s=load_time,
                inference_time_s=inference_time,
                tokens_per_second=len(output.split()) / inference_time
                if inference_time > 0
                else None,
                memory_mb=None,  # Ollama doesn't expose this easily
                output_length_chars=len(output),
                success=True,
            )

        except Exception as e:
            logger.error(f"Benchmark failed for {model_name}: {e}")
            return BenchmarkResult(
                model_name=model_name,
                model_type="vision",
                backend="ollama",
                load_time_s=0.0,
                inference_time_s=0.0,
                tokens_per_second=None,
                memory_mb=None,
                output_length_chars=0,
                success=False,
                error_message=str(e),
            )

    @staticmethod
    def benchmark_text_model(model_name: str, prompt: str) -> BenchmarkResult:
        """Benchmark Ollama text model (Sabiá)."""
        logger.info(f"Benchmarking {model_name} (Ollama text)...")

        if not OllamaBenchmark.check_model_exists(model_name):
            return BenchmarkResult(
                model_name=model_name,
                model_type="text",
                backend="ollama",
                load_time_s=0.0,
                inference_time_s=0.0,
                tokens_per_second=None,
                memory_mb=None,
                output_length_chars=0,
                success=False,
                error_message=f"Model {model_name} not installed",
            )

        try:
            # Warm-up
            start_load = time.time()
            subprocess.run(
                ["ollama", "run", model_name, "test"],
                capture_output=True,
                timeout=60,
            )
            load_time = time.time() - start_load

            # Benchmark
            start_inference = time.time()
            result = subprocess.run(
                ["ollama", "run", model_name, prompt],
                capture_output=True,
                text=True,
                timeout=120,
            )
            inference_time = time.time() - start_inference

            if result.returncode != 0:
                raise RuntimeError(f"Inference failed: {result.stderr}")

            output = result.stdout.strip()

            return BenchmarkResult(
                model_name=model_name,
                model_type="text",
                backend="ollama",
                load_time_s=load_time,
                inference_time_s=inference_time,
                tokens_per_second=len(output.split()) / inference_time
                if inference_time > 0
                else None,
                memory_mb=None,
                output_length_chars=len(output),
                success=True,
            )

        except Exception as e:
            logger.error(f"Benchmark failed for {model_name}: {e}")
            return BenchmarkResult(
                model_name=model_name,
                model_type="text",
                backend="ollama",
                load_time_s=0.0,
                inference_time_s=0.0,
                tokens_per_second=None,
                memory_mb=None,
                output_length_chars=0,
                success=False,
                error_message=str(e),
            )


class TransformersBenchmark:
    """Benchmark HuggingFace transformers models."""

    @staticmethod
    def benchmark_llava_med(image_path: Path, prompt: str) -> BenchmarkResult:
        """Benchmark LLaVA-Med via transformers."""
        logger.info("Benchmarking LLaVA-Med (transformers)...")

        try:
            import torch
            from PIL import Image
            from transformers import AutoProcessor, LlavaForConditionalGeneration

            model_id = "microsoft/llava-med-v1.5-mistral-7b"

            # Load model
            start_load = time.time()
            processor = AutoProcessor.from_pretrained(model_id)
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float16
            )

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model = model.to(device)
            load_time = time.time() - start_load

            # Prepare inputs
            image = Image.open(image_path)
            full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
            inputs = processor(text=full_prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Benchmark inference
            start_inference = time.time()
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=200)
            inference_time = time.time() - start_inference

            output = processor.decode(output_ids[0], skip_special_tokens=True)

            # Estimate memory (rough)
            memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

            return BenchmarkResult(
                model_name="llava-med-v1.5-mistral-7b",
                model_type="vision",
                backend="transformers",
                load_time_s=load_time,
                inference_time_s=inference_time,
                tokens_per_second=len(output.split()) / inference_time
                if inference_time > 0
                else None,
                memory_mb=memory_mb,
                output_length_chars=len(output),
                success=True,
            )

        except Exception as e:
            logger.error(f"LLaVA-Med benchmark failed: {e}")
            return BenchmarkResult(
                model_name="llava-med-v1.5-mistral-7b",
                model_type="vision",
                backend="transformers",
                load_time_s=0.0,
                inference_time_s=0.0,
                tokens_per_second=None,
                memory_mb=None,
                output_length_chars=0,
                success=False,
                error_message=str(e),
            )

    @staticmethod
    def benchmark_biomistral(prompt: str) -> BenchmarkResult:
        """Benchmark BioMistral via transformers."""
        logger.info("Benchmarking BioMistral (transformers)...")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = "BioMistral/BioMistral-7B"

            # Load model
            start_load = time.time()
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model = model.to(device)
            load_time = time.time() - start_load

            # Prepare inputs
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Benchmark
            start_inference = time.time()
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    pad_token_id=tokenizer.eos_token_id,
                )
            inference_time = time.time() - start_inference

            output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

            return BenchmarkResult(
                model_name="BioMistral-7B",
                model_type="text",
                backend="transformers",
                load_time_s=load_time,
                inference_time_s=inference_time,
                tokens_per_second=len(output.split()) / inference_time
                if inference_time > 0
                else None,
                memory_mb=memory_mb,
                output_length_chars=len(output),
                success=True,
            )

        except Exception as e:
            logger.error(f"BioMistral benchmark failed: {e}")
            return BenchmarkResult(
                model_name="BioMistral-7B",
                model_type="text",
                backend="transformers",
                load_time_s=0.0,
                inference_time_s=0.0,
                tokens_per_second=None,
                memory_mb=None,
                output_length_chars=0,
                success=False,
                error_message=str(e),
            )


def run_full_benchmark(image_path: Path) -> list[BenchmarkResult]:
    """Run complete benchmark suite."""
    results = []

    # Test prompts
    vision_prompt = (
        "Describe any pathological findings in this chest X-ray. "
        "Focus on consolidations, infiltrates, and pleural abnormalities."
    )
    text_prompt = (
        "Question: What are the primary radiological findings of bacterial pneumonia "
        "on chest X-ray?\n\nAnswer:"
    )

    # Ollama vision models
    for model in ["llava:7b", "llava:13b", "llava-llama3:latest"]:
        results.append(OllamaBenchmark.benchmark_vision_model(model, image_path, vision_prompt))

    # Ollama text models
    results.append(OllamaBenchmark.benchmark_text_model("sabia-7b", text_prompt))

    # Transformers models
    results.append(TransformersBenchmark.benchmark_llava_med(image_path, vision_prompt))
    results.append(TransformersBenchmark.benchmark_biomistral(text_prompt))

    return results


def print_results(results: list[BenchmarkResult]) -> None:
    """Print formatted benchmark results."""
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 80)

    for result in results:
        status = "✓" if result.success else "✗"
        logger.info(f"\n{status} {result.model_name} ({result.backend})")
        if result.success:
            logger.info(f"  Load time: {result.load_time_s:.2f}s")
            logger.info(f"  Inference time: {result.inference_time_s:.2f}s")
            if result.tokens_per_second:
                logger.info(f"  Speed: {result.tokens_per_second:.1f} tokens/s")
            if result.memory_mb:
                logger.info(f"  Memory: {result.memory_mb:.0f} MB")
            logger.info(f"  Output length: {result.output_length_chars} chars")
        else:
            logger.info(f"  Error: {result.error_message}")


def save_results(results: list[BenchmarkResult], output_path: Path) -> None:
    """Save results to JSON."""
    data = [asdict(result) for result in results]
    with output_path.open("w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"\n✓ Results saved to {output_path}")


def main() -> None:
    """Main benchmark script."""
    parser = argparse.ArgumentParser(description="Benchmark all models")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("imgs/person75_bacteria_365.jpeg"),
        help="Test image path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    if not args.image.exists():
        logger.error(f"Image not found: {args.image}")
        sys.exit(1)

    # Run benchmarks
    results = run_full_benchmark(args.image)

    # Display results
    print_results(results)

    # Save to file
    save_results(results, args.output)


if __name__ == "__main__":
    main()
