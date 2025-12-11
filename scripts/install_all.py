#!/usr/bin/env python3
"""
Master Installation Script

Installs all medical AI models sequentially with progress tracking.
Optimized for M4 Pro 24GB with automatic error handling.
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_script(script_name: str, args: list[str] = None) -> bool:
    """
    Run a setup script and return success status.

    Args:
        script_name: Name of the script to run
        args: Optional arguments to pass to the script

    Returns:
        True if successful, False otherwise
    """
    script_path = Path(__file__).parent / script_name
    cmd = ["uv", "run", str(script_path)]

    if args:
        cmd.extend(args)

    logger.info(f"Running: {' '.join(cmd)}")
    logger.info("=" * 80)

    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time

        logger.info("=" * 80)
        logger.info(f"✓ {script_name} completed in {elapsed / 60:.1f} minutes")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {script_name} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        logger.warning(f"✗ {script_name} interrupted by user")
        return False
    except Exception as e:
        logger.error(f"✗ {script_name} failed: {e}")
        return False


def main() -> None:
    """Main installation orchestrator."""
    parser = argparse.ArgumentParser(
        description="Install all medical AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Install all models
  python install_all.py

  # Install only LLaVA-Med and BioMistral
  python install_all.py --models llava-med biomistral

  # Skip tests (faster)
  python install_all.py --skip-tests

  # Install specific models
  python install_all.py --models llava-med  # Only LLaVA-Med
        """,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["llava-med", "biomistral", "sabia", "all"],
        default=["all"],
        help="Models to install (default: all)",
    )
    parser.add_argument("--skip-tests", action="store_true", help="Skip test inference")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark after installation")
    args = parser.parse_args()

    # Determine which models to install
    if "all" in args.models:
        models_to_install = ["llava-med", "biomistral", "sabia"]
    else:
        models_to_install = args.models

    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║              PNEUMOFINDER MODEL INSTALLATION                 ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info(f"\nModels to install: {', '.join(models_to_install)}")
    logger.info(f"Skip tests: {args.skip_tests}")
    logger.info(f"Run benchmark: {args.benchmark}")
    logger.info("")

    # Track results
    results = {}
    total_start = time.time()

    # Install each model
    script_args = ["--skip-test"] if args.skip_tests else []

    if "llava-med" in models_to_install:
        logger.info("\n[1/3] Installing LLaVA-Med v1.5...")
        results["llava-med"] = run_script("setup_llava_med.py", script_args)
        logger.info("")

    if "biomistral" in models_to_install:
        logger.info("\n[2/3] Installing BioMistral-7B...")
        results["biomistral"] = run_script("setup_biomistral.py", script_args)
        logger.info("")

    if "sabia" in models_to_install:
        logger.info("\n[3/3] Installing Sabiá-7B...")
        # Force transformers backend (Ollama version not available)
        sabia_args = script_args + ["--method", "transformers"]
        results["sabia"] = run_script("setup_sabia.py", sabia_args)
        logger.info("")

    # Run benchmark if requested
    if args.benchmark and any(results.values()):
        logger.info("\nRunning benchmark...")
        benchmark_success = run_script("benchmark_models.py")
        results["benchmark"] = benchmark_success

    # Print summary
    total_elapsed = time.time() - total_start
    logger.info("\n╔══════════════════════════════════════════════════════════════╗")
    logger.info("║                    INSTALLATION SUMMARY                      ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")

    for model, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"{status} {model}")

    logger.info(f"\nTotal time: {total_elapsed / 60:.1f} minutes")

    # Exit with error if any installation failed
    if not all(results.values()):
        logger.error("\n⚠ Some installations failed. Check logs above.")
        sys.exit(1)
    else:
        logger.info("\n✓ All installations completed successfully!")
        logger.info("\nNext steps:")
        logger.info("  1. Review benchmark results (if run)")
        logger.info("  2. Move to Day 2: RAG infrastructure")
        logger.info("  3. Create advanced prompts")


if __name__ == "__main__":
    main()
