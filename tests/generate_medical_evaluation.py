#!/usr/bin/env python3
"""
Generate medical evaluation dataset for LLaVA model comparison.

This script processes chest X-ray images through 3 LLaVA models and generates
comprehensive outputs for medical expert evaluation.

Usage:
    python tests/generate_medical_evaluation.py
    python tests/generate_medical_evaluation.py --seed 42
    python tests/generate_medical_evaluation.py --cleanup
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Project imports
from src.core.diagnosis import load_cnn_model, diagnose_from_path
from src.core.visualization import (
    find_last_conv_layer,
    find_resnet_base,
    generate_complete_visualization,
)
from src.core.clinical_description import (
    add_medical_disclaimer,
    call_ollama_api,
    encode_image_to_base64,
    load_medical_prompt_template,
)
from src.utils.file_utils import ensure_directory
from src.utils.image_utils import load_image, preprocess_image


# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS = ["llava:7b", "llava:13b", "llava-llama3"]
CNN_MODEL_PATH = "models/pneumonia_model.keras"
SAMPLES_DIR = Path("data/samples")
OUTPUT_DIR = Path("tests/medical_evaluation")
PROMPT_TEMPLATE_PATH = "prompts/medical_analysis.txt"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TEMP = 0.3
OLLAMA_MAX_TOKENS = 500

# Test cases (fixed for reproducibility - originally selected with seed=42)
# Case 05 added specifically to test LLM robustness when CNN fails
EVALUATION_CASES = {
    "case_01_normal_clear": ("6_normal3.jpeg", "NORMAL"),
    "case_02_pneumonia_severe": ("3_pneumonia1.jpeg", "PNEUMONIA"),
    "case_03_normal_challenging": ("1_normal1.jpeg", "NORMAL"),
    "case_04_pneumonia_moderate": ("5_pneumonia3.jpeg", "PNEUMONIA"),
    "case_05_false_negative": ("person72_bacteria_354.jpeg", "PNEUMONIA"),  # CNN ERROR
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def check_ollama_model_exists(model_name: str) -> bool:
    """Check if Ollama model is installed."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
        return model_name in result.stdout
    except Exception as e:
        print(f"⚠️ Error checking Ollama models: {e}")
        return False


def download_ollama_model(model_name: str) -> None:
    """Download Ollama model if not present."""
    print(f"\n📥 Downloading {model_name}...")
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"✅ {model_name} downloaded successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download {model_name}: {e}")
        sys.exit(1)


def translate_to_portuguese(text_en: str, model_name: str) -> tuple[str, float]:
    """
    Translate English text to Portuguese using LLM.

    Args:
        text_en: English text to translate
        model_name: Ollama model to use

    Returns:
        Tuple of (translated_text, latency_ms)
    """
    prompt = f"""Translate the following medical description from English to Brazilian Portuguese.
Keep all medical terminology accurate and maintain professional tone.

English text:
{text_en}

Brazilian Portuguese translation:"""

    start_time = time.time()

    try:
        result = subprocess.run(
            ["ollama", "run", model_name, prompt],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        translated = result.stdout.strip()
        latency_ms = (time.time() - start_time) * 1000
        return translated, latency_ms
    except Exception as e:
        print(f"⚠️ Translation failed: {e}")
        latency_ms = (time.time() - start_time) * 1000
        return f"[Translation unavailable]\n\n{text_en}", latency_ms


def process_single_case(
    model_name: str,
    case_id: str,
    image_filename: str,
    ground_truth: str,
    output_case_dir: Path,
    cnn_model,
    base_model,
    last_conv_layer,
    prompt_template: str,
) -> None:
    """
    Process a single case through the complete pipeline.

    Pipeline:
    1. Copy original image
    2. Run CNN inference
    3. Generate Grad-CAM heatmap + overlay
    4. Generate LLM description (English)
    5. Translate to Portuguese
    6. Save all outputs
    """
    print(f"\n  📄 Processing: {case_id}")

    # Setup paths
    ensure_directory(str(output_case_dir))
    source_image = SAMPLES_DIR / image_filename
    original_dest = output_case_dir / "1_original.jpeg"
    heatmap_path = output_case_dir / "2_gradcam_heatmap.png"
    overlay_path = output_case_dir / "3_gradcam_overlay.png"
    cnn_json = output_case_dir / "4_cnn_diagnosis.json"
    desc_en_path = output_case_dir / "5_llm_description_en.txt"
    desc_pt_path = output_case_dir / "6_llm_description_pt.txt"
    metadata_path = output_case_dir / "7_llm_metadata.json"

    # 1. Copy original image
    shutil.copy2(source_image, original_dest)
    print(f"    ✓ Copied original image")

    # 2. CNN Inference
    start_cnn = time.time()
    diagnosis, confidence = diagnose_from_path(cnn_model, str(source_image))
    latency_cnn = (time.time() - start_cnn) * 1000

    cnn_data = {
        "diagnosis": diagnosis,
        "confidence": float(confidence),
        "latency_ms": float(latency_cnn),
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

    print(f"    ✓ CNN: {diagnosis} ({confidence * 100:.1f}%) - {latency_cnn:.1f}ms")

    # 3. Grad-CAM Generation
    start_gradcam = time.time()

    # Load image for visualization
    pil_image = load_image(str(source_image))
    image_array = preprocess_image(str(source_image))
    image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Generate visualizations
    heatmap, overlay = generate_complete_visualization(
        base_model, last_conv_layer, image_array, image_bgr
    )

    # Save visualizations
    cv2.imwrite(str(heatmap_path), heatmap)
    cv2.imwrite(str(overlay_path), overlay)

    latency_gradcam = (time.time() - start_gradcam) * 1000
    print(f"    ✓ Grad-CAM generated - {latency_gradcam:.1f}ms")

    # 4. LLM Description (English)
    start_llm_en = time.time()

    prompt = prompt_template.format(class_name=diagnosis, confidence=f"{confidence * 100:.1f}")

    overlay_b64 = encode_image_to_base64(str(overlay_path))

    desc_en = call_ollama_api(
        prompt=prompt,
        image_base64=overlay_b64,
        model_name=model_name,
        ollama_host=OLLAMA_HOST,
        max_tokens=OLLAMA_MAX_TOKENS,
        temperature=OLLAMA_TEMP,
    )

    if not desc_en:
        desc_en = f"[LLM generation failed for {model_name}]"

    # Add disclaimer
    desc_en = add_medical_disclaimer(desc_en, confidence)

    latency_llm_en = (time.time() - start_llm_en) * 1000

    with open(desc_en_path, "w", encoding="utf-8") as f:
        f.write(desc_en)

    word_count_en = len(desc_en.split())
    print(f"    ✓ LLM (EN): {word_count_en} words - {latency_llm_en:.1f}ms")

    # 5. Translate to Portuguese
    desc_pt, latency_llm_pt = translate_to_portuguese(desc_en, model_name)

    with open(desc_pt_path, "w", encoding="utf-8") as f:
        f.write(desc_pt)

    word_count_pt = len(desc_pt.split())
    print(f"    ✓ LLM (PT): {word_count_pt} words - {latency_llm_pt:.1f}ms")

    # 6. Save metadata
    metadata = {
        "model": model_name,
        "case_id": case_id,
        "image_filename": image_filename,
        "latency_cnn_ms": float(latency_cnn),
        "latency_gradcam_ms": float(latency_gradcam),
        "latency_llm_en_ms": float(latency_llm_en),
        "latency_llm_pt_ms": float(latency_llm_pt),
        "total_latency_ms": float(latency_cnn + latency_gradcam + latency_llm_en),
        "word_count_en": word_count_en,
        "word_count_pt": word_count_pt,
        "char_count_en": len(desc_en),
        "char_count_pt": len(desc_pt),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    total_time = metadata["total_latency_ms"] + latency_llm_pt
    print(f"    ✓ Total: {total_time:.1f}ms")


def generate_readme() -> str:
    """Generate README.md content for medical evaluator."""
    return """# Avaliação Comparativa de Modelos LLaVA para Diagnóstico de Pneumonia

## 🎯 Objetivo

Avaliar empiricamente qual modelo LLaVA (7B, 13B, ou Llama3) gera descrições clínicas mais precisas e úteis para diagnóstico auxiliado por IA de pneumonia em radiografias de tórax.

## 📊 Estrutura do Estudo

### Modelos Testados
1. **LLaVA 7B** - Modelo base (4.7GB)
2. **LLaVA 13B** - Modelo grande (8.0GB)
3. **LLaVA-Llama3** - Modelo moderno baseado em Llama 3 (5.5GB)

### Casos Clínicos
Todos os 3 modelos analisaram os **MESMOS 4 casos**:
- **Caso 1:** Pulmão normal (caso claro)
- **Caso 2:** Pneumonia severa
- **Caso 3:** Pulmão normal (caso desafiador)
- **Caso 4:** Pneumonia moderada

## 📁 Arquivos por Caso

Cada caso contém 7 arquivos numerados:

1. **1_original.jpeg** - Radiografia de tórax original
2. **2_gradcam_heatmap.png** - Mapa de calor Grad-CAM (regiões relevantes para CNN)
3. **3_gradcam_overlay.png** - Sobreposição do mapa de calor no raio-X
4. **4_cnn_diagnosis.json** - Diagnóstico da CNN (classe, confiança, latência)
5. **5_llm_description_en.txt** - Descrição clínica em inglês
6. **6_llm_description_pt.txt** - Descrição clínica em português
7. **7_llm_metadata.json** - Metadados técnicos (latências, contagem de palavras)

## 📋 Como Avaliar

1. **Abra o formulário:** `evaluation_form.md`
2. **Para cada modelo** (llava_7b, llava_13b, llava_llama3):
   - Navegue até a pasta do modelo
   - Para cada caso (01 a 04):
     - Visualize a imagem original (`1_original.jpeg`)
     - Observe o Grad-CAM (`3_gradcam_overlay.png`) - áreas em vermelho são onde a CNN focou
     - Leia o diagnóstico da CNN (`4_cnn_diagnosis.json`)
     - **Leia a descrição em português** (`6_llm_description_pt.txt`)
     - Preencha o formulário de avaliação
3. **Compare os 3 modelos** e indique sua recomendação

## ⏱️ Tempo Estimado
- **15-20 minutos** (avaliação completa)

## 🔬 Critérios de Avaliação

- **Precisão anatômica:** Identificação correta de estruturas e regiões
- **Terminologia médica:** Uso apropriado de termos técnicos
- **Interpretação radiológica:** Correção dos achados descritos
- **Consistência com CNN:** Alinhamento com o diagnóstico automatizado
- **Utilidade clínica:** Valor agregado para tomada de decisão
- **Problemas:** Alucinações, omissões, contradições

## 📊 Dados Técnicos

Detalhes sobre latências, arquitetura da CNN, e parâmetros do LLM estão disponíveis em `evaluation_metadata.json`.

## 📧 Contato

Para dúvidas ou feedback, entre em contato com a equipe do projeto.

---

**Obrigado pela sua contribuição para este estudo!** 🙏
"""


def generate_evaluation_form() -> str:
    """Generate evaluation form (reduced version)."""
    form_template = """# Formulário de Avaliação - Modelos LLaVA

**Avaliador:** _________________________  
**Data:** ___/___/______  
**Especialidade/CRM:** _________________________

---

## Instruções

Você avaliará **3 modelos de IA** que geram descrições clínicas de raios-X de tórax.

- Cada modelo analisou os **MESMOS 4 casos** (2 normais + 2 pneumonia)
- Foque na descrição em português: `6_llm_description_pt.txt`
- Compare com: imagem original + overlay Grad-CAM + diagnóstico CNN
- **Tempo estimado:** 15-20 minutos

---
"""

    # For each model
    for model_idx, model_name in enumerate(["LLaVA 7B", "LLaVA 13B", "LLaVA-Llama3"], 1):
        form_template += f"\n## MODELO {model_idx}: {model_name}\n"

        # For each case
        case_names = [
            "Caso 01 - Normal (Claro)",
            "Caso 02 - Pneumonia (Severa)",
            "Caso 03 - Normal (Desafiador)",
            "Caso 04 - Pneumonia (Moderada)",
        ]

        for case_name in case_names:
            form_template += f"\n### {case_name}\n\n"
            form_template += "**1. Avaliação Geral (1-5):** ___\n\n"
            form_template += "**2. Problemas Identificados (marque todos aplicáveis):**\n"
            form_template += "- [ ] Anatomia incorreta ou imprecisa\n"
            form_template += "- [ ] Terminologia médica inapropriada\n"
            form_template += "- [ ] Interpretação radiológica incorreta\n"
            form_template += "- [ ] Inconsistente com diagnóstico CNN\n"
            form_template += "- [ ] Alucinações (informações não visíveis)\n"
            form_template += "- [ ] Omissão de achados importantes\n"
            form_template += "- [ ] Nenhum problema significativo\n\n"
            form_template += "**3. Comentários (opcional):**  \n"
            form_template += "_______________________________________________  \n"
            form_template += "_______________________________________________\n\n"
            form_template += "---\n"

    # Final comparison section
    form_template += """
## COMPARAÇÃO FINAL

### Ranking Geral
Ordene os modelos do **melhor (1) ao pior (3)**:

**Geral (considerando todos os 4 casos):**
1. ____________________
2. ____________________
3. ____________________

---

### Modelo Recomendado

Qual modelo você recomendaria para uso em sistema de apoio à decisão clínica?

- [ ] LLaVA 7B
- [ ] LLaVA 13B
- [ ] LLaVA-Llama3
- [ ] Nenhum (não estão prontos para uso clínico)

**Justificativa (1-2 frases):**  
_______________________________________________  
_______________________________________________  
_______________________________________________

---

### Comentários Adicionais

Observações gerais, sugestões de melhoria, ou outros comentários:

_______________________________________________  
_______________________________________________  
_______________________________________________  
_______________________________________________

---

## Escala de Avaliação Geral (1-5)

- **5 - Excelente:** Descrição precisa, detalhada, clinicamente útil
- **4 - Boa:** Descrição correta e adequada
- **3 - Adequada:** Descrição genérica mas não incorreta
- **2 - Ruim:** Descrição imprecisa ou confusa
- **1 - Inadequada:** Descrição incorreta ou não útil

---

**Obrigado pela sua avaliação!** 🙏
"""

    return form_template


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate medical evaluation dataset for LLaVA comparison"
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="42",
        help="Random seed (deprecated - cases are now fixed for reproducibility)",
    )
    parser.add_argument(
        "--only-case",
        type=str,
        choices=list(EVALUATION_CASES.keys()),
        help="Process only a specific case (e.g., case_05_false_negative)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove llava:7b and llava:13b models after generation",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("LLaVA Medical Evaluation - Dataset Generator")
    print("=" * 80)

    # 1. Get test cases
    print("\n📋 Test cases:")
    test_cases = EVALUATION_CASES

    # Filter if --only-case specified
    if args.only_case:
        if args.only_case in test_cases:
            test_cases = {args.only_case: test_cases[args.only_case]}
            print(f"  ⚠️  Processing ONLY: {args.only_case}")
        else:
            print(f"❌ Error: Case '{args.only_case}' not found in EVALUATION_CASES")
            sys.exit(1)

    for case_id, (filename, ground_truth) in test_cases.items():
        print(f"  - {case_id}: {filename} ({ground_truth})")

    # 2. Download models if needed
    print("\n📥 Checking Ollama models...")
    for model in MODELS:
        if not check_ollama_model_exists(model):
            download_ollama_model(model)
        else:
            print(f"  ✓ {model} already installed")

    # 3. Load CNN model
    print(f"\n🤖 Loading CNN model: {CNN_MODEL_PATH}")
    cnn_model = load_cnn_model(CNN_MODEL_PATH)
    base_model = find_resnet_base(cnn_model)
    last_conv_layer = find_last_conv_layer(base_model)
    print("  ✓ CNN model loaded successfully")

    # 4. Load prompt template
    prompt_template = load_medical_prompt_template(PROMPT_TEMPLATE_PATH)

    # 5. Process all combinations (3 models × 4 cases = 12 iterations)
    print("\n🔄 Processing cases...")
    total_start = time.time()

    for model_name in MODELS:
        print(f"\n{'=' * 80}")
        print(f"MODEL: {model_name}")
        print(f"{'=' * 80}")

        model_dir = OUTPUT_DIR / f"model_{model_name.replace(':', '_')}"

        for case_id, (filename, ground_truth) in test_cases.items():
            case_dir = model_dir / case_id

            try:
                process_single_case(
                    model_name=model_name,
                    case_id=case_id,
                    image_filename=filename,
                    ground_truth=ground_truth,
                    output_case_dir=case_dir,
                    cnn_model=cnn_model,
                    base_model=base_model,
                    last_conv_layer=last_conv_layer,
                    prompt_template=prompt_template,
                )
            except Exception as e:
                print(f"    ❌ Error processing {case_id}: {e}")
                import traceback

                traceback.print_exc()

    total_time = time.time() - total_start

    # 6. Generate documentation
    print("\n📝 Generating documentation...")

    readme_path = OUTPUT_DIR / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(generate_readme())
    print(f"  ✓ {readme_path}")

    form_path = OUTPUT_DIR / "evaluation_form.md"
    with open(form_path, "w", encoding="utf-8") as f:
        f.write(generate_evaluation_form())
    print(f"  ✓ {form_path}")

    # 7. Generate evaluation metadata
    metadata = {
        "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": 42,  # Historical reference (cases now fixed)
        "note": "Cases 01-04 originally selected with seed=42. Case 05 added to test CNN error handling.",
        "models_tested": MODELS,
        "test_cases": {
            case_id: {
                "filename": filename,
                "ground_truth": ground_truth,
                "cnn_error": case_id == "case_05_false_negative",
            }
            for case_id, (filename, ground_truth) in EVALUATION_CASES.items()
        },
        "cnn_model": CNN_MODEL_PATH,
        "cnn_architecture": "ResNet50",
        "prompt_template": PROMPT_TEMPLATE_PATH,
        "ollama_config": {
            "host": OLLAMA_HOST,
            "temperature": OLLAMA_TEMP,
            "max_tokens": OLLAMA_MAX_TOKENS,
        },
        "total_processing_time_seconds": float(total_time),
    }

    metadata_path = OUTPUT_DIR / "evaluation_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ {metadata_path}")

    # 8. Cleanup if requested
    if args.cleanup:
        print("\n🧹 Cleaning up models...")
        for model in ["llava:7b", "llava:13b"]:
            try:
                subprocess.run(["ollama", "rm", model], check=False)
                print(f"  ✓ Removed {model}")
            except Exception as e:
                print(f"  ⚠️ Could not remove {model}: {e}")

    # 9. Summary
    print("\n" + "=" * 80)
    print("✅ GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print(f"⏱️  Total time: {total_time / 60:.1f} minutes")
    print(f"\n📊 Generated:")
    print(
        f"  - {len(MODELS)} models × {len(test_cases)} cases × 7 files = {len(MODELS) * len(test_cases) * 7} files"
    )
    print(f"  - 3 documentation files (README, form, metadata)")
    print(f"\n📋 Next steps:")
    print(f"  1. Review: {OUTPUT_DIR}")
    print(f"  2. Share with medical evaluator")
    print(f"  3. Collect evaluation form responses")
    print(f"  4. Analyze results for TCC")
    print()


if __name__ == "__main__":
    main()
