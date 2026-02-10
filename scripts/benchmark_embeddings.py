"""Benchmark different embedding models for medical RAG retrieval.

Compares:
1. all-MiniLM-L6-v2 (current - general purpose)
2. pritamdeka/S-PubMedBert-MS-MARCO (medical - PubMed trained)
3. sentence-transformers/all-mpnet-base-v2 (larger general purpose)

Run with: python scripts/benchmark_embeddings.py
"""

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Test queries - medical terms and diagnoses
TEST_QUERIES = [
    # Diagnosis queries
    ("pneumonia", ["Lobar Pneumonia", "Bronchopneumonia", "Interstitial Pneumonia"]),
    ("normal chest", ["Normal Chest X-ray", "Normal Variants"]),
    ("bilateral consolidation", ["Bilateral Pneumonia", "ARDS", "Pulmonary Edema"]),

    # Anatomical queries
    ("right lower lobe", ["Lobar Pneumonia", "Aspiration Pneumonia", "Lung Zones"]),
    ("pleural effusion", ["Pleural Effusion", "Parapneumonic Effusion"]),

    # Signs and patterns
    ("air bronchogram", ["Air Bronchogram Sign", "Lobar Pneumonia"]),
    ("ground glass opacity", ["Interstitial Pneumonia", "Viral Pneumonia", "ARDS"]),
    ("consolidation with fever", ["Community-Acquired Pneumonia", "Lobar Pneumonia"]),

    # Clinical scenarios
    ("immunocompromised patient lung infection", ["Hospital-Acquired Pneumonia", "Bronchopneumonia"]),
    ("aspiration risk patient", ["Aspiration Pneumonia"]),
]

# Sample documents from our knowledge base
SAMPLE_DOCUMENTS = [
    "Lobar pneumonia presents as homogeneous consolidation involving an entire lobe, with sharp demarcation at fissures. Air bronchograms are typically present.",
    "Bronchopneumonia shows multifocal patchy consolidations or ground-glass opacities, often bilateral and perihilar.",
    "Interstitial pneumonia presents with reticular or ground-glass opacities, predominantly in lower lobes.",
    "Normal chest X-ray: clear bilateral lung fields, sharp costophrenic angles, normal cardiac silhouette.",
    "Aspiration pneumonia typically affects gravity-dependent lung segments: posterior segments of upper lobes and superior segments of lower lobes.",
    "Hospital-acquired pneumonia develops ≥48h after hospital admission. Often multidrug-resistant organisms.",
    "Air bronchograms appear as dark, branching air-filled bronchi visible against opacified lung parenchyma.",
    "Pleural effusion presents as blunting of costophrenic angle (requires ~200mL on upright PA view).",
    "ARDS presents with bilateral, diffuse opacities within 7 days of clinical insult.",
    "Viral pneumonia typically shows bilateral, diffuse, ground-glass opacities with interstitial pattern.",
    "Community-Acquired Pneumonia (CAP) radiographic severity correlates with clinical outcomes. Multilobar involvement indicates severe CAP.",
    "On PA chest X-ray, the RIGHT side appears on the LEFT of the image. Right lung has three lobes.",
    "Cardiogenic pulmonary edema: bilateral, symmetric, perihilar (butterfly pattern), cardiomegaly.",
    "Parapneumonic effusion: loculated, >1cm on decubitus, pH <7.2, requires drainage.",
    "Common normal variants: azygos fissure (1%), pectus excavatum, scoliosis, prominent nipple shadows.",
]

# Models to benchmark
EMBEDDING_MODELS = [
    {
        "name": "all-MiniLM-L6-v2",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Current model - fast, general purpose (384 dims)",
    },
    {
        "name": "S-PubMedBert-MS-MARCO",
        "model_id": "pritamdeka/S-PubMedBert-MS-MARCO",
        "description": "Medical - trained on PubMed (768 dims)",
    },
    {
        "name": "all-mpnet-base-v2",
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "description": "Larger general purpose (768 dims)",
    },
]


def compute_similarity(model: SentenceTransformer, query: str, documents: list[str]) -> list[tuple[int, float]]:
    """Compute cosine similarity between query and documents.

    Returns list of (doc_index, similarity_score) sorted by score descending.
    """
    # Encode query and documents
    query_embedding = model.encode([query], convert_to_tensor=True)
    doc_embeddings = model.encode(documents, convert_to_tensor=True)

    # Compute cosine similarity
    from sentence_transformers import util
    similarities = util.cos_sim(query_embedding, doc_embeddings)[0]

    # Sort by similarity
    results = [(i, float(similarities[i])) for i in range(len(documents))]
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def evaluate_model(model_info: dict) -> dict:
    """Evaluate a single embedding model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {model_info['name']}")
    logger.info(f"Description: {model_info['description']}")
    logger.info(f"{'='*60}")

    # Load model
    start = time.time()
    try:
        model = SentenceTransformer(model_info["model_id"])
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {"name": model_info["name"], "error": str(e)}

    load_time = time.time() - start
    logger.info(f"Model loaded in {load_time:.2f}s")

    # Run queries
    results = {
        "name": model_info["name"],
        "model_id": model_info["model_id"],
        "description": model_info["description"],
        "load_time_s": load_time,
        "embedding_dim": model.get_sentence_embedding_dimension(),
        "queries": [],
        "avg_top1_score": 0,
        "avg_top3_score": 0,
        "recall_at_3": 0,  # How often expected docs appear in top 3
    }

    total_top1 = 0
    total_top3 = 0
    recall_hits = 0

    for query, expected_keywords in TEST_QUERIES:
        start = time.time()
        rankings = compute_similarity(model, query, SAMPLE_DOCUMENTS)
        query_time = time.time() - start

        # Get top 3 results
        top3 = rankings[:3]
        top3_docs = [SAMPLE_DOCUMENTS[idx] for idx, _ in top3]
        top3_scores = [score for _, score in top3]

        # Check if any expected keyword appears in top 3 docs
        hit = False
        for doc in top3_docs:
            for keyword in expected_keywords:
                if keyword.lower() in doc.lower():
                    hit = True
                    break
            if hit:
                break

        if hit:
            recall_hits += 1

        total_top1 += top3_scores[0]
        total_top3 += sum(top3_scores) / 3

        query_result = {
            "query": query,
            "expected": expected_keywords,
            "top3_docs": [doc[:80] + "..." for doc in top3_docs],
            "top3_scores": [round(s, 4) for s in top3_scores],
            "query_time_ms": round(query_time * 1000, 2),
            "hit": hit,
        }
        results["queries"].append(query_result)

        logger.info(f"Query: '{query}'")
        logger.info(f"  Top match ({top3_scores[0]:.3f}): {top3_docs[0][:60]}...")
        logger.info(f"  Hit: {'✓' if hit else '✗'}")

    # Calculate averages
    n_queries = len(TEST_QUERIES)
    results["avg_top1_score"] = round(total_top1 / n_queries, 4)
    results["avg_top3_score"] = round(total_top3 / n_queries, 4)
    results["recall_at_3"] = round(recall_hits / n_queries, 4)

    logger.info(f"\nSUMMARY for {model_info['name']}:")
    logger.info(f"  Avg Top-1 Score: {results['avg_top1_score']:.4f}")
    logger.info(f"  Avg Top-3 Score: {results['avg_top3_score']:.4f}")
    logger.info(f"  Recall@3: {results['recall_at_3']:.1%} ({recall_hits}/{n_queries})")

    return results


def main():
    """Run benchmark on all models."""
    logger.info("="*60)
    logger.info("MEDICAL EMBEDDING BENCHMARK")
    logger.info("="*60)
    logger.info(f"Test queries: {len(TEST_QUERIES)}")
    logger.info(f"Sample documents: {len(SAMPLE_DOCUMENTS)}")
    logger.info(f"Models to test: {len(EMBEDDING_MODELS)}")

    all_results = []

    for model_info in EMBEDDING_MODELS:
        result = evaluate_model(model_info)
        all_results.append(result)

    # Print comparison
    logger.info("\n" + "="*60)
    logger.info("FINAL COMPARISON")
    logger.info("="*60)

    print("\n" + "-"*80)
    print(f"{'Model':<30} {'Dims':<6} {'Load(s)':<8} {'Top1':<8} {'Top3':<8} {'Recall@3':<10}")
    print("-"*80)

    for r in all_results:
        if "error" in r:
            print(f"{r['name']:<30} ERROR: {r['error']}")
        else:
            print(f"{r['name']:<30} {r['embedding_dim']:<6} {r['load_time_s']:<8.2f} "
                  f"{r['avg_top1_score']:<8.4f} {r['avg_top3_score']:<8.4f} {r['recall_at_3']:<10.1%}")

    print("-"*80)

    # Find best model
    valid_results = [r for r in all_results if "error" not in r]
    if valid_results:
        best = max(valid_results, key=lambda x: x["recall_at_3"])
        print(f"\n✓ Best model by Recall@3: {best['name']} ({best['recall_at_3']:.1%})")

        # Save results
        output_path = project_root / "tests" / "embedding_benchmark_results.json"
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    main()
