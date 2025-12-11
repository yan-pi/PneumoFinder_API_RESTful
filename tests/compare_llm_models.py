"""
Script para comparar diferentes modelos LLaVA.
Testa qualidade, latência e detalhamento das descrições clínicas.

Modelos a comparar:
- llava:7b (4.7GB, Vicuna 7B base)
- llava:13b (8.0GB, Vicuna 13B base)
- llava-llama3 (5.5GB, Llama 3 8B base)

Usage:
    python tests/compare_llm_models.py
"""

import json
import time
from pathlib import Path

from src.core.clinical_description import generate_clinical_description


def test_model(
    model_name: str,
    diagnosis: str,
    confidence: float,
    image_path: str,
) -> dict:
    """
    Testa um modelo LLaVA específico.

    Args:
        model_name: Nome do modelo Ollama (ex: "llava:7b")
        diagnosis: Diagnóstico CNN
        confidence: Confiança do diagnóstico
        image_path: Caminho da imagem de teste

    Returns:
        dict com description, latency_ms, word_count, char_count
    """
    print(f"\n{'=' * 80}")
    print(f"Testing: {model_name}")
    print(f"{'=' * 80}")

    start_time = time.time()

    try:
        description = generate_clinical_description(
            diagnosis=diagnosis,
            confidence=confidence,
            original_image_path=image_path,
            ollama_model=model_name,
        )

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # Métricas da descrição
        word_count = len(description.split())
        char_count = len(description)

        print(f"\n✅ Success!")
        print(f"⏱️  Latency: {latency_ms:.0f}ms ({latency_ms / 1000:.1f}s)")
        print(f"📝 Words: {word_count}")
        print(f"📊 Characters: {char_count}")
        print(f"\n📄 Description Preview (first 300 chars):")
        print("-" * 80)
        print(description[:300] + "...")
        print("-" * 80)

        return {
            "model": model_name,
            "success": True,
            "description": description,
            "latency_ms": latency_ms,
            "word_count": word_count,
            "char_count": char_count,
        }

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return {
            "model": model_name,
            "success": False,
            "error": str(e),
            "latency_ms": None,
            "word_count": None,
            "char_count": None,
        }


def main():
    """Executa comparação entre modelos LLaVA."""

    print("\n" + "=" * 80)
    print("LLaVA Model Comparison Tool")
    print("=" * 80)

    # 1. Escolhe imagem de teste
    test_images = [
        "imgs/person75_bacteria_365.jpeg",
        "imgs/NORMAL2-IM-1431-0001.jpeg",
        "data/samples/pneumonia_case.jpg",
    ]

    # Encontra primeira imagem que existe
    test_image = None
    for img_path in test_images:
        if Path(img_path).exists():
            test_image = img_path
            break

    if not test_image:
        print("\n❌ Nenhuma imagem de teste encontrada.")
        print("   Coloque uma radiografia em um dos caminhos:")
        for path in test_images:
            print(f"   - {path}")
        return

    print(f"\n📁 Test Image: {test_image}")

    # 2. Define diagnóstico de teste
    # Você pode mudar estes valores conforme necessário
    diagnosis = "PNEUMONIA"
    confidence = 0.87

    print(f"🔬 Test Diagnosis: {diagnosis} ({confidence:.1%})")

    # 3. Testa cada modelo
    models_to_test = [
        "llava:7b",  # Baseline atual
        "llava:13b",  # Modelo maior
        "llava-llama3",  # Modelo mais recente
    ]

    results = []

    for model_name in models_to_test:
        result = test_model(
            model_name=model_name,
            diagnosis=diagnosis,
            confidence=confidence,
            image_path=test_image,
        )
        results.append(result)

        # Pausa entre modelos para evitar sobrecarregar Ollama
        if result["success"]:
            time.sleep(2)

    # 4. Comparação final
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    # Tabela de comparação
    print(
        "\n{:<20} {:<10} {:<12} {:<10} {:<10}".format(
            "Model", "Status", "Latency", "Words", "Chars"
        )
    )
    print("-" * 80)

    for r in results:
        if r["success"]:
            print(
                "{:<20} {:<10} {:<12} {:<10} {:<10}".format(
                    r["model"],
                    "✅ OK",
                    f"{r['latency_ms']:.0f}ms",
                    str(r["word_count"]),
                    str(r["char_count"]),
                )
            )
        else:
            print(
                "{:<20} {:<10} {:<12}".format(
                    r["model"],
                    "❌ FAIL",
                    r.get("error", "Unknown error")[:50],
                )
            )

    # 5. Salva resultados completos
    output_file = "tests/llm_comparison_results.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Full results saved to: {output_file}")

    # 6. Recomendação
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        print("\n❌ No models succeeded. Check Ollama server status:")
        print("   1. Make sure Ollama is running: ollama serve")
        print("   2. Pull missing models:")
        print("      - ollama pull llava:13b")
        print("      - ollama pull llava-llama3")
        return

    # Melhor por latência
    fastest = min(successful_results, key=lambda x: x["latency_ms"])
    print(f"\n⚡ Fastest: {fastest['model']} ({fastest['latency_ms']:.0f}ms)")

    # Mais detalhado
    most_detailed = max(successful_results, key=lambda x: x["word_count"])
    print(f"📝 Most Detailed: {most_detailed['model']} ({most_detailed['word_count']} words)")

    # Melhor trade-off (palavras por segundo)
    for r in successful_results:
        r["words_per_second"] = r["word_count"] / (r["latency_ms"] / 1000)

    best_tradeoff = max(successful_results, key=lambda x: x["words_per_second"])
    print(
        f"⚖️  Best Trade-off: {best_tradeoff['model']} ({best_tradeoff['words_per_second']:.1f} words/s)"
    )

    # Comparação de qualidade (heurística baseada em palavras e conteúdo)
    print("\n📊 Quality Indicators:")
    for r in successful_results:
        desc = r["description"]

        # Palavras técnicas médicas
        medical_terms = [
            "consolidação",
            "infiltrado",
            "opacidade",
            "broncograma",
            "campo pulmonar",
            "radiografia",
            "tórax",
            "bilateral",
        ]
        medical_count = sum(1 for term in medical_terms if term.lower() in desc.lower())

        # Score de qualidade (palavras * termos médicos / latência)
        quality_score = (r["word_count"] * (1 + medical_count * 0.5)) / (r["latency_ms"] / 1000)

        print(f"   {r['model']}: Quality Score = {quality_score:.1f}")
        print(f"      - Medical terms: {medical_count}")
        print(f"      - Words/second: {r['words_per_second']:.1f}")

    print("\n" + "=" * 80)
    print("💡 RECOMMENDATION FOR TCC:")
    print("=" * 80)

    # Recomendação baseada nos resultados
    if len(successful_results) >= 2:
        print("\n   Compare the full descriptions in llm_comparison_results.json")
        print("   Consider these factors:")
        print("   - Medical terminology richness")
        print("   - Clinical relevance of observations")
        print("   - Structured format (anatomy → findings → conclusion)")
        print("   - Latency vs. quality trade-off for your use case")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
