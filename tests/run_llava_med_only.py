#!/usr/bin/env python3
"""
LLaVA-Med Isolated Evaluation Script

Runs llava-med pipeline evaluation in isolated environment with protobuf 3.20.3.
Reuses CNN diagnosis and Grad-CAM from llava-llama3 reference model.

Usage:
    .venv_llava_med/bin/python3 tests/run_llava_med_only.py
    .venv_llava_med/bin/python3 tests/run_llava_med_only.py --only-case case_01_normal_clear
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Project imports
from src.core.medical_pipeline import MedicalPipeline
from src.utils.file_utils import ensure_directory

# ============================================================================
# CONFIGURATION
# ============================================================================

VISION_MODEL = "llava-med"  # Only test llava-med
REFERENCE_MODEL = "llava-llama3"  # Source for CNN + Grad-CAM reuse

# Test cases (same as main evaluation)
EVALUATION_CASES = {
    "case_01_normal_clear": ("6_normal3.jpeg", "NORMAL"),
    "case_02_pneumonia_severe": ("3_pneumonia1.jpeg", "PNEUMONIA"),
    "case_03_normal_challenging": ("1_normal1.jpeg", "NORMAL"),
    "case_04_pneumonia_moderate": ("5_pneumonia3.jpeg", "PNEUMONIA"),
    "case_05_false_negative": ("person72_bacteria_354.jpeg", "PNEUMONIA"),
}

# Paths
SAMPLES_DIR = Path("data/samples")
OUTPUT_DIR = Path("tests/pipeline_evaluation_3stage")
REFERENCE_DIR = OUTPUT_DIR / f"vision_{REFERENCE_MODEL}"
LLAVA_MED_DIR = OUTPUT_DIR / "vision_llava-med"

# ============================================================================
# HELPER FUNCTIONS (copied from main evaluation for 100% parity)
# ============================================================================


def count_medical_terms(text: str) -> tuple[int, list[str]]:
    """Count medical terminology in text."""
    medical_terms = [
        # Anatomical
        "lung",
        "pulmonary",
        "cardiac",
        "mediastin",
        "diaphragm",
        "thoracic",
        "bilateral",
        "unilateral",
        "right",
        "left",
        "apex",
        "base",
        # Findings
        "opacity",
        "opacities",
        "consolidation",
        "infiltrate",
        "infiltration",
        "effusion",
        "pneumothorax",
        "atelectasis",
        "nodule",
        "mass",
        # Pathology
        "pneumonia",
        "infection",
        "inflammation",
        "edema",
        "congestion",
        # Quality
        "clear",
        "normal",
        "abnormal",
        "increased",
        "decreased",
        # Clinical
        "radiograph",
        "x-ray",
        "chest",
        "findings",
        "assessment",
    ]

    text_lower = text.lower()
    found_terms = []
    for term in medical_terms:
        if term in text_lower:
            found_terms.append(term)

    return len(found_terms), found_terms


def analyze_content_quality(text: str) -> dict:
    """Analyze text quality metrics."""
    words = text.split()
    sentences = text.split(".")

    # Count medical terms
    medical_count, medical_list = count_medical_terms(text)

    # Check structural elements
    has_anatomy = any(term in text.lower() for term in ["lung", "cardiac", "mediastin"])
    has_findings = any(term in text.lower() for term in ["opacity", "consolidation", "clear"])
    has_assessment = any(term in text.lower() for term in ["normal", "abnormal", "demonstrates"])

    return {
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "char_count": len(text),
        "medical_term_count": medical_count,
        "medical_terms_found": medical_list,
        "medical_density": medical_count / len(words) if words else 0,
        "has_anatomy_description": has_anatomy,
        "has_findings_description": has_findings,
        "has_assessment": has_assessment,
        "structural_completeness": sum([has_anatomy, has_findings, has_assessment]) / 3.0,
    }


def process_llava_med_case(
    case_id: str,
    image_filename: str,
    ground_truth: str,
    output_case_dir: Path,
    reference_case_dir: Path,
) -> dict:
    """
    Process a single case with llava-med, reusing CNN + Grad-CAM from reference.

    Pipeline:
    1. Copy original image from reference
    2. Copy CNN diagnosis from reference
    3. Copy Grad-CAM overlay from reference
    4. Run Stage 1 (Vision) with llava-med
    5. Run Stage 2 (Medical + RAG) with BioMistral
    6. Generate reports and save metrics

    Returns:
        Dictionary with all metrics
    """
    print(f"\n  📄 Processing: {case_id}")
    case_start_time = time.time()

    # Setup paths
    ensure_directory(str(output_case_dir))

    # Reference files (source)
    ref_original = reference_case_dir / "1_original.jpeg"
    ref_cnn_json = reference_case_dir / "2_cnn_diagnosis.json"
    ref_gradcam = reference_case_dir / "3_gradcam_overlay.png"

    # Verify reference files exist
    if not ref_original.exists():
        raise FileNotFoundError(f"Reference original not found: {ref_original}")
    if not ref_cnn_json.exists():
        raise FileNotFoundError(f"Reference CNN JSON not found: {ref_cnn_json}")
    if not ref_gradcam.exists():
        raise FileNotFoundError(f"Reference Grad-CAM not found: {ref_gradcam}")

    # Output files (9 files per case)
    original_dest = output_case_dir / "1_original.jpeg"
    cnn_json = output_case_dir / "2_cnn_diagnosis.json"
    gradcam_overlay = output_case_dir / "3_gradcam_overlay.png"
    stage1_json = output_case_dir / "4_stage1_vision.json"
    stage2_json = output_case_dir / "5_stage2_medical.json"
    stage3_json = output_case_dir / "6_stage3_translation.json"  # Not used
    final_report_en = output_case_dir / "7_final_report_en.txt"
    final_report_pt = output_case_dir / "8_final_report_pt.txt"  # Not used (no translation)
    metrics_json = output_case_dir / "9_pipeline_metrics.json"

    # 1. Copy original image from reference
    shutil.copy2(ref_original, original_dest)
    print(f"    ✓ Copied original image from reference")

    # 2. Copy CNN diagnosis from reference
    shutil.copy2(ref_cnn_json, cnn_json)
    with open(ref_cnn_json, "r") as f:
        cnn_data = json.load(f)
    print(
        f"    ✓ Reused CNN: {cnn_data['diagnosis']} "
        f"({cnn_data['confidence'] * 100:.1f}%) - {cnn_data['latency_s']:.3f}s"
    )

    # 3. Copy Grad-CAM from reference
    shutil.copy2(ref_gradcam, gradcam_overlay)
    print(f"    ✓ Reused Grad-CAM from reference")

    # 4. Initialize Medical Pipeline with llava-med
    pipeline = MedicalPipeline(
        use_rag=True,
        vision_model=VISION_MODEL,
        enable_translation=False,  # Disabled (2-stage)
    )

    # 5. Run Complete Pipeline
    print(f"    ⚙️  Running {VISION_MODEL} pipeline...")
    pipeline_start = time.time()

    try:
        result = pipeline.analyze_xray(
            image_path=str(SAMPLES_DIR / image_filename),
            cnn_diagnosis=cnn_data,
        )
    except Exception as e:
        print(f"    ❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e), "case_id": case_id, "vision_model": VISION_MODEL}

    pipeline_latency = time.time() - pipeline_start

    if not result.get("success"):
        print(f"    ❌ Pipeline returned error: {result.get('error')}")
        return result

    # 6. Save Stage Outputs

    # Stage 1: Vision
    stage1_data = {
        "model": result["stage1_vision"]["model"],
        "findings_en": result["stage1_vision"]["findings_en"],
        "rag_context": result["stage1_vision"].get("rag_context", ""),
        "latency_s": result["stage1_vision"]["latency_s"],
        **analyze_content_quality(result["stage1_vision"]["findings_en"]),
    }

    with open(stage1_json, "w", encoding="utf-8") as f:
        json.dump(stage1_data, f, indent=2, ensure_ascii=False)

    print(
        f"    ✓ Stage 1: {stage1_data['word_count']} words, "
        f"{stage1_data['medical_term_count']} medical terms - "
        f"{stage1_data['latency_s']:.1f}s"
    )

    # Stage 2: Medical
    stage2_data = {
        "model": result["stage2_medical"]["model"],
        "report_en": result["stage2_medical"]["report_en"],
        "rag_context": result["stage2_medical"].get("rag_context", ""),
        "latency_s": result["stage2_medical"]["latency_s"],
        **analyze_content_quality(result["stage2_medical"]["report_en"]),
    }

    with open(stage2_json, "w", encoding="utf-8") as f:
        json.dump(stage2_data, f, indent=2, ensure_ascii=False)

    print(
        f"    ✓ Stage 2: {stage2_data['word_count']} words, "
        f"{stage2_data['medical_term_count']} medical terms - "
        f"{stage2_data['latency_s']:.1f}s"
    )

    # 7. Generate Final Report (English only)

    report_en_text = f"""CHEST X-RAY ANALYSIS - 3-STAGE MEDICAL PIPELINE
{"=" * 70}

CNN CLASSIFICATION
Diagnosis: {cnn_data["diagnosis"]} (Confidence: {cnn_data["confidence"] * 100:.1f}%)
Ground Truth: {ground_truth}
{f"⚠️  CNN ERROR: {cnn_data['error_type']}" if cnn_data.get("error_type") else "✓ Correct Classification"}

{"=" * 70}

STAGE 1: VISION ANALYSIS ({VISION_MODEL})

{result["stage1_vision"]["findings_en"]}

{"=" * 70}

STAGE 2: MEDICAL ASSESSMENT (BioMistral-7B + RAG)

{result["stage2_medical"]["report_en"]}

{"=" * 70}

PERFORMANCE METRICS
- Stage 1 (Vision): {stage1_data["latency_s"]:.1f}s
- Stage 2 (Medical): {stage2_data["latency_s"]:.1f}s
- Total Pipeline: {result["total_latency_s"]:.1f}s

{"=" * 70}

⚠️ DISCLAIMER
This AI-assisted analysis is for educational and research purposes only.
Always consult qualified healthcare professionals for medical decisions.
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""

    with open(final_report_en, "w", encoding="utf-8") as f:
        f.write(report_en_text)

    # 8. Save Complete Metrics
    total_case_time = time.time() - case_start_time

    metrics = {
        "case_id": case_id,
        "image_filename": image_filename,
        "ground_truth": ground_truth,
        "vision_model": VISION_MODEL,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        # CNN metrics (reused)
        "cnn_diagnosis": cnn_data["diagnosis"],
        "cnn_confidence": float(cnn_data["confidence"]),
        "cnn_correct": cnn_data["correct"],
        "cnn_latency_s": float(cnn_data["latency_s"]),
        # Pipeline metrics
        "total_pipeline_latency_s": result["total_latency_s"],
        "stage1_latency_s": stage1_data["latency_s"],
        "stage2_latency_s": stage2_data["latency_s"],
        "stage3_latency_s": None,
        "gradcam_latency_s": None,  # Reused, not re-generated
        "total_case_time_s": float(total_case_time),
        # Content metrics
        "stage1_word_count": stage1_data["word_count"],
        "stage1_medical_terms": stage1_data["medical_term_count"],
        "stage1_medical_density": stage1_data["medical_density"],
        "stage2_word_count": stage2_data["word_count"],
        "stage2_medical_terms": stage2_data["medical_term_count"],
        "stage2_medical_density": stage2_data["medical_density"],
        # Quality indicators
        "stage1_structural_completeness": stage1_data["structural_completeness"],
        "stage2_structural_completeness": stage2_data["structural_completeness"],
        # RAG
        "used_rag": True,
        "rag_context_available": bool(stage1_data.get("rag_context")),
    }

    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"    ✅ Complete: {total_case_time:.1f}s total")

    return metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="LLaVA-Med Isolated Evaluation - Reuse CNN + Grad-CAM from reference"
    )
    parser.add_argument(
        "--only-case",
        type=str,
        choices=list(EVALUATION_CASES.keys()),
        help="Process only a specific case",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("LLAVA-MED ISOLATED EVALUATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  - Vision Model: {VISION_MODEL}")
    print(f"  - Reference Model: {REFERENCE_MODEL} (for CNN + Grad-CAM)")
    print(f"  - Stage 1: llava-med")
    print(f"  - Stage 2: BioMistral-7B + RAG")
    print(f"  - Stage 3: DISABLED (no translation)")
    print(f"  - Isolated venv: .venv_llava_med (protobuf 3.20.3)")

    # 1. Verify reference directory exists
    if not REFERENCE_DIR.exists():
        print(f"\n❌ Error: Reference directory not found: {REFERENCE_DIR}")
        print(f"   Run main evaluation first to generate reference data.")
        sys.exit(1)

    print(f"\n✓ Reference directory found: {REFERENCE_DIR}")

    # 2. Get test cases
    test_cases = EVALUATION_CASES
    if args.only_case:
        if args.only_case in test_cases:
            test_cases = {args.only_case: test_cases[args.only_case]}
            print(f"\n⚠️  Processing ONLY: {args.only_case}")
        else:
            print(f"\n❌ Error: Case '{args.only_case}' not found")
            sys.exit(1)

    print(f"\n📋 Test cases: {len(test_cases)}")
    for case_id, (filename, ground_truth) in test_cases.items():
        print(f"  - {case_id}: {filename} ({ground_truth})")

    # 3. Process all cases
    print(f"\n🔄 Starting evaluation...")
    print(f"  Total runs: {len(test_cases)}")
    estimated_time = len(test_cases) * 180  # ~3 min per case
    print(f"  Estimated time: ~{estimated_time / 60:.0f} minutes")
    print()

    all_results = []
    total_start = time.time()

    for case_id, (filename, ground_truth) in test_cases.items():
        case_dir = LLAVA_MED_DIR / case_id
        reference_case_dir = REFERENCE_DIR / case_id

        # Verify reference case exists
        if not reference_case_dir.exists():
            print(f"\n  ❌ Skipping {case_id}: Reference case not found")
            continue

        try:
            metrics = process_llava_med_case(
                case_id=case_id,
                image_filename=filename,
                ground_truth=ground_truth,
                output_case_dir=case_dir,
                reference_case_dir=reference_case_dir,
            )
            all_results.append(metrics)
        except Exception as e:
            print(f"    ❌ Error processing {case_id}: {e}")
            import traceback

            traceback.print_exc()

    total_time = time.time() - total_start

    # 4. Update Summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Output directory: {LLAVA_MED_DIR}")
    print(f"⏱️  Total time: {total_time / 60:.1f} minutes")
    print(f"\n📊 Generated:")
    print(f"  - {len(test_cases)} cases × 9 files = {len(test_cases) * 9} files")
    print(f"  - {len(all_results)} complete pipeline runs")

    # 5. Update aggregated summary
    summary_path = OUTPUT_DIR / "evaluation_summary.json"

    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary_data = json.load(f)

        # Remove old llava-med errors
        summary_data["results"] = [
            r for r in summary_data["results"] if r.get("vision_model") != VISION_MODEL
        ]

        # Add new llava-med results
        summary_data["results"].extend(all_results)

        # Update metadata
        if VISION_MODEL not in summary_data["configuration"]["vision_models_tested"]:
            summary_data["configuration"]["vision_models_tested"].append(VISION_MODEL)

        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        print(f"\n💾 Summary updated: {summary_path}")
    else:
        print(f"\n⚠️  Warning: Summary file not found, creating new one")
        summary_data = {
            "execution_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_runtime_minutes": total_time / 60,
            "configuration": {
                "vision_models_tested": [VISION_MODEL],
                "test_cases": list(test_cases.keys()),
                "translation_enabled": False,
                "rag_enabled": True,
            },
            "results": all_results,
        }

        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        print(f"\n💾 Summary created: {summary_path}")

    # 6. Quick stats
    if all_results:
        # Filter out error results
        successful_results = [r for r in all_results if "error" not in r]
        if successful_results:
            print(f"\n📈 Quick Statistics ({VISION_MODEL}):")
            avg_latency = sum(r["total_pipeline_latency_s"] for r in successful_results) / len(
                successful_results
            )
            avg_words_s1 = sum(r["stage1_word_count"] for r in successful_results) / len(
                successful_results
            )
            avg_medical_s2 = sum(r["stage2_medical_terms"] for r in successful_results) / len(
                successful_results
            )

            print(f"    - Avg latency: {avg_latency:.1f}s")
            print(f"    - Avg Stage 1 words: {avg_words_s1:.0f}")
            print(f"    - Avg Stage 2 medical terms: {avg_medical_s2:.0f}")
        else:
            print(f"\n⚠️  All runs failed - no statistics available")

    print("\n" + "=" * 80)
    print("✅ llava-med evaluation complete!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
