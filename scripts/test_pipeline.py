"""Test script for medical pipeline validation.

Tests the complete 3-stage pipeline on sample X-rays:
1. Normal X-ray (must NOT hallucinate findings)
2. Pneumonia X-ray (must detect correctly)

Run with: python scripts/test_pipeline.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.medical_pipeline import MedicalPipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_normal_xray(pipeline: MedicalPipeline, output_dir: Path) -> dict:
    """Test pipeline on a normal chest X-ray.

    Args:
        pipeline: MedicalPipeline instance
        output_dir: Directory to save results

    Returns:
        Test result dictionary
    """
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Normal Chest X-ray (Hallucination Test)")
    logger.info("=" * 80)

    # Find normal X-ray from test dataset
    test_cases = list((project_root / "tests/medical_evaluation").glob("*/case_*_normal_*"))

    if not test_cases:
        logger.warning("No normal test cases found, using sample image")
        image_path = str(project_root / "imgs/person75_bacteria_365.jpeg")
        cnn_diagnosis = {"prediction": "NORMAL", "confidence": 0.95}
    else:
        case_dir = test_cases[0]
        image_path = str(case_dir / "1_original.jpeg")

        # Load CNN diagnosis if available
        cnn_file = case_dir / "4_cnn_diagnosis.json"
        if cnn_file.exists():
            with open(cnn_file) as f:
                cnn_data = json.load(f)
                cnn_diagnosis = {
                    "prediction": "NORMAL",
                    "confidence": cnn_data.get("probability_normal", 0.95),
                }
        else:
            cnn_diagnosis = {"prediction": "NORMAL", "confidence": 0.95}

    # Run pipeline
    result = pipeline.analyze_xray(image_path, cnn_diagnosis)

    # Save result
    output_file = output_dir / "test_normal_xray.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Validate: NO hallucinated findings
    if result["success"]:
        report = result["final_report_pt_br"].lower()

        # Check for hallucination keywords
        hallucination_keywords = [
            "consolidação",
            "infiltrado",
            "opacidade",
            "pneumonia",
            "derrame",
            "massa",
            "nódulo",
        ]

        hallucinated = [kw for kw in hallucination_keywords if kw in report]

        if hallucinated:
            logger.warning(f"⚠️  HALLUCINATION DETECTED: {hallucinated}")
            result["hallucination_test"] = "FAILED"
            result["hallucinated_terms"] = hallucinated
        else:
            logger.info("✅ NO HALLUCINATION - Normal X-ray described correctly")
            result["hallucination_test"] = "PASSED"

    logger.info(f"\nResults saved to: {output_file}")
    return result


def test_pneumonia_xray(pipeline: MedicalPipeline, output_dir: Path) -> dict:
    """Test pipeline on a pneumonia chest X-ray.

    Args:
        pipeline: MedicalPipeline instance
        output_dir: Directory to save results

    Returns:
        Test result dictionary
    """
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Pneumonia Chest X-ray (Detection Test)")
    logger.info("=" * 80)

    # Find pneumonia X-ray from test dataset
    test_cases = list((project_root / "tests/medical_evaluation").glob("*/case_*_pneumonia_*"))

    if not test_cases:
        logger.warning("No pneumonia test cases found, using sample image")
        image_path = str(project_root / "imgs/person75_bacteria_365.jpeg")
        cnn_diagnosis = {"prediction": "PNEUMONIA", "confidence": 0.89}
    else:
        case_dir = test_cases[0]
        image_path = str(case_dir / "1_original.jpeg")

        # Load CNN diagnosis if available
        cnn_file = case_dir / "4_cnn_diagnosis.json"
        if cnn_file.exists():
            with open(cnn_file) as f:
                cnn_data = json.load(f)
                cnn_diagnosis = {
                    "prediction": "PNEUMONIA",
                    "confidence": cnn_data.get("probability_pneumonia", 0.89),
                }
        else:
            cnn_diagnosis = {"prediction": "PNEUMONIA", "confidence": 0.89}

    # Run pipeline
    result = pipeline.analyze_xray(image_path, cnn_diagnosis)

    # Save result
    output_file = output_dir / "test_pneumonia_xray.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Validate: Should detect pneumonia
    if result["success"]:
        report = result["final_report_pt_br"].lower()

        # Check for pneumonia indicators
        pneumonia_keywords = ["pneumonia", "consolidação", "infiltrado", "opacidade"]

        detected = [kw for kw in pneumonia_keywords if kw in report]

        if detected:
            logger.info(f"✅ PNEUMONIA DETECTED: {detected}")
            result["detection_test"] = "PASSED"
            result["detected_terms"] = detected
        else:
            logger.warning("⚠️  PNEUMONIA NOT DETECTED - False negative")
            result["detection_test"] = "FAILED"

    logger.info(f"\nResults saved to: {output_file}")
    return result


def test_custom_image(pipeline: MedicalPipeline, image_path: str, output_dir: Path) -> dict:
    """Test pipeline on a custom image.

    Args:
        pipeline: MedicalPipeline instance
        image_path: Path to custom X-ray image
        output_dir: Directory to save results

    Returns:
        Test result dictionary
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"TEST: Custom Image - {image_path}")
    logger.info("=" * 80)

    # Use unknown diagnosis (let vision model decide)
    cnn_diagnosis = {"prediction": "UNKNOWN", "confidence": 0.0}

    # Run pipeline
    result = pipeline.analyze_xray(image_path, cnn_diagnosis)

    # Save result
    image_name = Path(image_path).stem
    output_file = output_dir / f"test_{image_name}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to: {output_file}")
    return result


def main() -> None:
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test medical pipeline")
    parser.add_argument(
        "--image", type=str, help="Path to custom X-ray image (optional)", default=None
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        choices=["llava-med", "llava-llama3"],
        default="llava-llama3",
        help="Vision model to use",
    )
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG context grounding")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/pipeline_tests",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("MEDICAL PIPELINE TESTING")
    logger.info("=" * 80)
    logger.info(f"Vision Model: {args.vision_model}")
    logger.info(f"RAG Enabled: {not args.no_rag}")
    logger.info(f"Output Directory: {output_dir}")

    # Initialize pipeline
    try:
        pipeline = MedicalPipeline(use_rag=not args.no_rag, vision_model=args.vision_model)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)

    # Run tests
    results = {}

    if args.image:
        # Test custom image
        results["custom"] = test_custom_image(pipeline, args.image, output_dir)
    else:
        # Run standard tests
        results["normal_xray"] = test_normal_xray(pipeline, output_dir)
        results["pneumonia_xray"] = test_pneumonia_xray(pipeline, output_dir)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    for test_name, result in results.items():
        if result.get("success"):
            latency = result.get("total_latency_s", 0)
            logger.info(f"\n{test_name.upper()}:")
            logger.info(f"  Status: ✅ Success ({latency:.1f}s)")

            # Show test-specific results
            if "hallucination_test" in result:
                status = result["hallucination_test"]
                logger.info(f"  Hallucination Test: {status}")

            if "detection_test" in result:
                status = result["detection_test"]
                logger.info(f"  Detection Test: {status}")

            logger.info(f"  Final Report:\n    {result['final_report_pt_br']}")
        else:
            logger.error(f"\n{test_name.upper()}: ❌ Failed - {result.get('error')}")

    logger.info("\n" + "=" * 80)
    logger.info("Tests complete! Results saved to: " + str(output_dir))


if __name__ == "__main__":
    main()
