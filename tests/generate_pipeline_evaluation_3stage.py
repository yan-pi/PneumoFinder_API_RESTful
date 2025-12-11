#!/usr/bin/env python3
"""
3-Stage Medical Pipeline Evaluation - Complete Vision Model Comparison

This script evaluates 4 vision models in the complete 3-stage pipeline:
  Stage 1: Vision Analysis (VARIABLE: llava-med, llava-llama3, llava:7b, llava:13b)
  Stage 2: Medical Enhancement (FIXED: BioMistral-7B + RAG)
  Stage 3: Translation (OPTIONAL: Sabiá-7B) - DISABLED by default

Output: 9 files per case × 4 models × 5 cases = 180 files
Runtime: ~15 min per model × 4 models = ~60 min (2-stage)

Usage:
    python tests/generate_pipeline_evaluation_3stage.py
    python tests/generate_pipeline_evaluation_3stage.py --vision-model llava-med
    python tests/generate_pipeline_evaluation_3stage.py --only-case case_01_normal_clear
    python tests/generate_pipeline_evaluation_3stage.py --enable-translation
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Project imports
from src.core.diagnosis import diagnose_from_path, load_cnn_model
from src.core.medical_pipeline import MedicalPipeline
from src.core.visualization import (
    find_last_conv_layer,
    find_resnet_base,
    generate_complete_visualization,
)
from src.utils.file_utils import ensure_directory
from src.utils.image_utils import load_image, preprocess_image

import cv2
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

# Vision models to test
VISION_MODELS = [
    "llava-med",  # Microsoft LLaVA-Med v1.5 (medical specialist)
    "llava-llama3",  # LLaVA + Llama 3 (current production)
    "llava:7b",  # LLaVA 7B baseline (general)
    "llava:13b",  # LLaVA 13B (larger general)
]

# Test cases (reuse from old evaluation)
EVALUATION_CASES = {
    "case_01_normal_clear": ("6_normal3.jpeg", "NORMAL"),
    "case_02_pneumonia_severe": ("3_pneumonia1.jpeg", "PNEUMONIA"),
    "case_03_normal_challenging": ("1_normal1.jpeg", "NORMAL"),
    "case_04_pneumonia_moderate": ("5_pneumonia3.jpeg", "PNEUMONIA"),
    "case_05_false_negative": ("person72_bacteria_354.jpeg", "PNEUMONIA"),  # CNN ERROR
}

# Paths
CNN_MODEL_PATH = "models/pneumonia_model.keras"
SAMPLES_DIR = Path("data/samples")
OUTPUT_DIR = Path("tests/pipeline_evaluation_3stage")

# ============================================================================
# HELPER FUNCTIONS
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


def process_single_case(
    vision_model: str,
    case_id: str,
    image_filename: str,
    ground_truth: str,
    output_case_dir: Path,
    cnn_model,
    base_model,
    last_conv_layer,
    enable_translation: bool = False,
) -> dict:
    """
    Process a single case through the complete 3-stage pipeline.

    Pipeline:
    1. Copy original image
    2. Run CNN inference + Grad-CAM
    3. Run Stage 1 (Vision)
    4. Run Stage 2 (Medical + RAG)
    5. Run Stage 3 (Translation, optional)
    6. Generate reports and save metrics

    Returns:
        Dictionary with all metrics
    """
    print(f"\n  📄 Processing: {case_id}")
    case_start_time = time.time()

    # Setup paths
    ensure_directory(str(output_case_dir))
    source_image = SAMPLES_DIR / image_filename

    if not source_image.exists():
        raise FileNotFoundError(f"Image not found: {source_image}")

    # Output files (9 files per case)
    original_dest = output_case_dir / "1_original.jpeg"
    cnn_json = output_case_dir / "2_cnn_diagnosis.json"
    gradcam_overlay = output_case_dir / "3_gradcam_overlay.png"
    stage1_json = output_case_dir / "4_stage1_vision.json"
    stage2_json = output_case_dir / "5_stage2_medical.json"
    stage3_json = output_case_dir / "6_stage3_translation.json"
    final_report_en = output_case_dir / "7_final_report_en.txt"
    final_report_pt = output_case_dir / "8_final_report_pt.txt"
    metrics_json = output_case_dir / "9_pipeline_metrics.json"

    # 1. Copy original image
    shutil.copy2(source_image, original_dest)
    print(f"    ✓ Copied original image")

    # 2. CNN Inference
    start_cnn = time.time()
    diagnosis, confidence = diagnose_from_path(cnn_model, str(source_image))
    latency_cnn = time.time() - start_cnn

    cnn_data = {
        "diagnosis": diagnosis,
        "confidence": float(confidence),
        "latency_s": float(latency_cnn),
        "ground_truth": ground_truth,
        "correct": diagnosis == ground_truth,
        "error_type": (
            "FALSE_NEGATIVE"
            if (diagnosis == "NORMAL" and ground_truth == "PNEUMONIA")
            else "FALSE_POSITIVE"
            if (diagnosis == "PNEUMONIA" and ground_truth == "NORMAL")
            else None
        ),
    }

    with open(cnn_json, "w") as f:
        json.dump(cnn_data, f, indent=2)

    print(f"    ✓ CNN: {diagnosis} ({confidence * 100:.1f}%) - {latency_cnn:.3f}s")

    # 3. Grad-CAM Generation
    start_gradcam = time.time()
    pil_image = load_image(str(source_image))
    image_array = preprocess_image(str(source_image))
    image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    heatmap, overlay = generate_complete_visualization(
        base_model, last_conv_layer, image_array, image_bgr
    )
    cv2.imwrite(str(gradcam_overlay), overlay)

    latency_gradcam = time.time() - start_gradcam
    print(f"    ✓ Grad-CAM generated - {latency_gradcam:.3f}s")

    # 4. Initialize Medical Pipeline
    pipeline = MedicalPipeline(
        use_rag=True,
        vision_model=vision_model,
        enable_translation=enable_translation,
    )

    # 5. Run Complete Pipeline
    print(f"    ⚙️  Running {vision_model} pipeline...")
    pipeline_start = time.time()

    try:
        result = pipeline.analyze_xray(
            image_path=str(source_image),
            cnn_diagnosis=cnn_data,
        )
    except Exception as e:
        print(f"    ❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e), "case_id": case_id, "vision_model": vision_model}

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

    # Stage 3: Translation (if enabled)
    stage3_latency = 0
    if enable_translation and "stage3_translation" in result:
        stage3_data = {
            "model": result["stage3_translation"]["model"],
            "report_pt_br": result["stage3_translation"]["report_pt_br"],
            "latency_s": result["stage3_translation"]["latency_s"],
            **analyze_content_quality(result["stage3_translation"]["report_pt_br"]),
        }
        stage3_latency = stage3_data["latency_s"]

        with open(stage3_json, "w", encoding="utf-8") as f:
            json.dump(stage3_data, f, indent=2, ensure_ascii=False)

        print(f"    ✓ Stage 3: {stage3_data['word_count']} words - {stage3_latency:.1f}s")

    # 7. Generate Final Reports

    # English report
    report_en_text = f"""CHEST X-RAY ANALYSIS - 3-STAGE MEDICAL PIPELINE
{"=" * 70}

CNN CLASSIFICATION
Diagnosis: {diagnosis} (Confidence: {confidence * 100:.1f}%)
Ground Truth: {ground_truth}
{f"⚠️  CNN ERROR: {cnn_data['error_type']}" if cnn_data["error_type"] else "✓ Correct Classification"}

{"=" * 70}

STAGE 1: VISION ANALYSIS ({vision_model})

{result["stage1_vision"]["findings_en"]}

{"=" * 70}

STAGE 2: MEDICAL ASSESSMENT (BioMistral-7B + RAG)

{result["stage2_medical"]["report_en"]}

{"=" * 70}

PERFORMANCE METRICS
- Stage 1 (Vision): {stage1_data["latency_s"]:.1f}s
- Stage 2 (Medical): {stage2_data["latency_s"]:.1f}s
{f"- Stage 3 (Translation): {stage3_latency:.1f}s" if enable_translation else ""}
- Total Pipeline: {result["total_latency_s"]:.1f}s

{"=" * 70}

⚠️ DISCLAIMER
This AI-assisted analysis is for educational and research purposes only.
Always consult qualified healthcare professionals for medical decisions.
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""

    with open(final_report_en, "w", encoding="utf-8") as f:
        f.write(report_en_text)

    # Portuguese report (if translation enabled)
    if enable_translation and "stage3_translation" in result:
        report_pt_text = f"""ANÁLISE DE RAIO-X DE TÓRAX - PIPELINE MÉDICO 3 ESTÁGIOS
{"=" * 70}

CLASSIFICAÇÃO CNN
Diagnóstico: {diagnosis} (Confiança: {confidence * 100:.1f}%)
Verdade Fundamental: {ground_truth}
{f"⚠️  ERRO DA CNN: {cnn_data['error_type']}" if cnn_data["error_type"] else "✓ Classificação Correta"}

{"=" * 70}

ESTÁGIO 3: RELATÓRIO MÉDICO TRADUZIDO (Sabiá-7B)

{result["stage3_translation"]["report_pt_br"]}

{"=" * 70}

MÉTRICAS DE DESEMPENHO
- Estágio 1 (Visão): {stage1_data["latency_s"]:.1f}s
- Estágio 2 (Médico): {stage2_data["latency_s"]:.1f}s
- Estágio 3 (Tradução): {stage3_latency:.1f}s
- Pipeline Total: {result["total_latency_s"]:.1f}s

{"=" * 70}

⚠️ AVISO
Esta análise assistida por IA é apenas para fins educacionais e de pesquisa.
Sempre consulte profissionais de saúde qualificados para decisões médicas.
Gerado: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
        with open(final_report_pt, "w", encoding="utf-8") as f:
            f.write(report_pt_text)

    # 8. Save Complete Metrics
    total_case_time = time.time() - case_start_time

    metrics = {
        "case_id": case_id,
        "image_filename": image_filename,
        "ground_truth": ground_truth,
        "vision_model": vision_model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        # CNN metrics
        "cnn_diagnosis": diagnosis,
        "cnn_confidence": float(confidence),
        "cnn_correct": cnn_data["correct"],
        "cnn_latency_s": float(latency_cnn),
        # Pipeline metrics
        "total_pipeline_latency_s": result["total_latency_s"],
        "stage1_latency_s": stage1_data["latency_s"],
        "stage2_latency_s": stage2_data["latency_s"],
        "stage3_latency_s": stage3_latency if enable_translation else None,
        "gradcam_latency_s": float(latency_gradcam),
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
        description="3-Stage Medical Pipeline Evaluation - Vision Model Comparison"
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        choices=VISION_MODELS,
        help="Test only a specific vision model",
    )
    parser.add_argument(
        "--only-case",
        type=str,
        choices=list(EVALUATION_CASES.keys()),
        help="Process only a specific case",
    )
    parser.add_argument(
        "--enable-translation",
        action="store_true",
        help="Enable Stage 3 (PT-BR translation) - adds ~150s per case",
    )
    parser.add_argument(
        "--skip-download-check",
        action="store_true",
        help="Skip checking if models are downloaded",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("3-STAGE MEDICAL PIPELINE EVALUATION")
    print("=" * 80)
    print(f"\nPipeline Configuration:")
    print(f"  - Stage 1: Vision Analysis (VARIABLE)")
    print(f"  - Stage 2: BioMistral-7B + RAG (FIXED)")
    print(
        f"  - Stage 3: Sabiá-7B Translation ({'ENABLED' if args.enable_translation else 'DISABLED'})"
    )

    # 1. Determine which models to test
    if args.vision_model:
        models_to_test = [args.vision_model]
        print(f"\n⚠️  Testing ONLY: {args.vision_model}")
    else:
        models_to_test = VISION_MODELS
        print(f"\n📋 Testing ALL vision models: {len(models_to_test)}")

    for model in models_to_test:
        print(f"  - {model}")

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

    # 3. Load CNN model
    print(f"\n🤖 Loading CNN model: {CNN_MODEL_PATH}")
    cnn_model = load_cnn_model(CNN_MODEL_PATH)
    base_model = find_resnet_base(cnn_model)
    last_conv_layer = find_last_conv_layer(base_model)
    print("  ✓ CNN model loaded")

    # 4. Process all combinations
    print(f"\n🔄 Starting evaluation...")
    print(
        f"  Total runs: {len(models_to_test)} models × {len(test_cases)} cases = {len(models_to_test) * len(test_cases)}"
    )

    estimated_time = (
        len(models_to_test) * len(test_cases) * (240 if args.enable_translation else 180)
    )
    print(f"  Estimated time: ~{estimated_time / 60:.0f} minutes")
    print()

    all_results = []
    total_start = time.time()

    for model_name in models_to_test:
        print(f"\n{'=' * 80}")
        print(f"VISION MODEL: {model_name}")
        print(f"{'=' * 80}")

        model_dir = OUTPUT_DIR / f"vision_{model_name.replace(':', '_')}"
        model_start = time.time()

        for case_id, (filename, ground_truth) in test_cases.items():
            case_dir = model_dir / case_id

            try:
                metrics = process_single_case(
                    vision_model=model_name,
                    case_id=case_id,
                    image_filename=filename,
                    ground_truth=ground_truth,
                    output_case_dir=case_dir,
                    cnn_model=cnn_model,
                    base_model=base_model,
                    last_conv_layer=last_conv_layer,
                    enable_translation=args.enable_translation,
                )
                all_results.append(metrics)
            except Exception as e:
                print(f"    ❌ Error processing {case_id}: {e}")
                import traceback

                traceback.print_exc()

        model_time = time.time() - model_start
        print(f"\n  ✅ {model_name} complete in {model_time / 60:.1f} minutes")

    total_time = time.time() - total_start

    # 5. Generate Summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print(f"⏱️  Total time: {total_time / 60:.1f} minutes")
    print(f"\n📊 Generated:")
    print(
        f"  - {len(models_to_test)} models × {len(test_cases)} cases × 9 files = {len(models_to_test) * len(test_cases) * 9} files"
    )
    print(f"  - {len(all_results)} complete pipeline runs")

    # 6. Save aggregated results
    summary_path = OUTPUT_DIR / "evaluation_summary.json"
    summary_data = {
        "execution_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_minutes": total_time / 60,
        "configuration": {
            "vision_models_tested": models_to_test,
            "test_cases": list(test_cases.keys()),
            "translation_enabled": args.enable_translation,
            "rag_enabled": True,
        },
        "results": all_results,
    }

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n💾 Summary saved: {summary_path}")

    # 7. Quick stats
    if all_results:
        print(f"\n📈 Quick Statistics:")

        # Group by vision model
        by_model = {}
        for r in all_results:
            if "error" in r:
                continue
            model = r["vision_model"]
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(r)

        for model, results in by_model.items():
            if not results:
                continue
            avg_latency = sum(r["total_pipeline_latency_s"] for r in results) / len(results)
            avg_words_s1 = sum(r["stage1_word_count"] for r in results) / len(results)
            avg_medical_s2 = sum(r["stage2_medical_terms"] for r in results) / len(results)

            print(f"\n  {model}:")
            print(f"    - Avg latency: {avg_latency:.1f}s")
            print(f"    - Avg Stage 1 words: {avg_words_s1:.0f}")
            print(f"    - Avg Stage 2 medical terms: {avg_medical_s2:.0f}")

    print("\n" + "=" * 80)
    print("📋 Next steps:")
    print("  1. Review outputs in:", OUTPUT_DIR)
    print("  2. Run analysis: python tests/analyze_pipeline_results.py")
    print("  3. Complete evaluation form (if human review)")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
