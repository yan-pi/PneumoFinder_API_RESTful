"""Test RAG retrieval system with sample queries.

Run with: python scripts/test_rag_retrieval.py
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.prompts.medical_prompts import (
    build_medical_text_prompt,
    build_translation_prompt,
    build_vision_prompt,
)
from src.rag.retriever import MedicalRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_terminology_retrieval(retriever: MedicalRetriever) -> None:
    """Test medical terminology translation."""
    logger.info("\n=== TEST 1: Medical Terminology Translation ===")

    test_terms = [
        "right lung",
        "consolidation",
        "pleural effusion",
        "air bronchogram",
        "normal",
    ]

    for term in test_terms:
        translation = retriever.translate_term(term)
        logger.info(f"  {term:20s} → {translation}")


def test_guidelines_retrieval(retriever: MedicalRetriever) -> None:
    """Test guideline retrieval for different diagnoses."""
    logger.info("\n=== TEST 2: Guideline Retrieval ===")

    test_cases = [
        ("pneumonia", 2),
        ("normal chest x-ray", 2),
        ("pleural effusion", 1),
    ]

    for diagnosis, n_results in test_cases:
        logger.info(f"\nQuery: '{diagnosis}'")
        guidelines = retriever.get_relevant_guidelines(diagnosis, n_results=n_results)

        for i, guideline in enumerate(guidelines, 1):
            logger.info(f"  [{i}] {guideline['metadata'].get('title', 'Untitled')}")
            logger.info(f"      Distance: {guideline['distance']:.3f}")
            logger.info(f"      Text: {guideline['text'][:100]}...")


def test_report_retrieval(retriever: MedicalRetriever) -> None:
    """Test similar report retrieval."""
    logger.info("\n=== TEST 3: Similar Report Retrieval ===")

    test_findings = [
        "clear lung fields, normal cardiac silhouette",
        "right lower lobe consolidation with air bronchograms",
    ]

    for findings in test_findings:
        logger.info(f"\nFindings: '{findings}'")
        reports = retriever.get_similar_reports(findings, n_results=1)

        for i, report in enumerate(reports, 1):
            logger.info(f"  [{i}] {report['metadata'].get('title', 'Untitled')}")
            logger.info(f"      Category: {report['metadata'].get('category', 'Unknown')}")
            logger.info(f"      Distance: {report['distance']:.3f}")


def test_rag_context_building(retriever: MedicalRetriever) -> None:
    """Test full RAG context building for prompts."""
    logger.info("\n=== TEST 4: RAG Context Building ===")

    # Test Case 1: Normal X-ray
    logger.info("\nCase 1: Normal X-ray")
    context = retriever.build_rag_context(
        diagnosis="normal",
        findings="clear bilateral lung fields, normal heart size",
        include_guidelines=True,
        include_reports=True,
    )
    logger.info(f"Context length: {len(context)} chars")
    logger.info(f"Context preview:\n{context[:300]}...")

    # Test Case 2: Pneumonia
    logger.info("\nCase 2: Pneumonia")
    context = retriever.build_rag_context(
        diagnosis="pneumonia",
        findings="right lower lobe consolidation",
        include_guidelines=True,
        include_reports=True,
    )
    logger.info(f"Context length: {len(context)} chars")
    logger.info(f"Context preview:\n{context[:300]}...")


def test_prompt_generation(retriever: MedicalRetriever) -> None:
    """Test prompt generation with RAG context."""
    logger.info("\n=== TEST 5: Prompt Generation with RAG ===")

    # Build RAG context
    rag_context = retriever.build_rag_context(
        diagnosis="pneumonia", findings=None, include_guidelines=True, include_reports=False
    )

    # Test vision prompt
    vision_prompt = build_vision_prompt(use_rag=True, rag_context=rag_context)
    logger.info(f"\nVision Prompt Length: {len(vision_prompt)} chars")
    logger.info("Vision Prompt Preview:")
    logger.info("-" * 80)
    logger.info(vision_prompt[:400] + "...")

    # Test medical text prompt
    vision_desc = "Right lower lobe consolidation with air bronchograms visible."
    cnn_result = {"prediction": "PNEUMONIA", "confidence": 0.89}

    medical_prompt = build_medical_text_prompt(
        vision_description=vision_desc,
        cnn_diagnosis=cnn_result,
        use_rag=True,
        rag_context=rag_context,
    )
    logger.info(f"\nMedical Text Prompt Length: {len(medical_prompt)} chars")

    # Test translation prompt
    english_report = "Consolidation in the right lower lobe consistent with pneumonia."
    translation_prompt = build_translation_prompt(english_report=english_report)
    logger.info(f"\nTranslation Prompt Length: {len(translation_prompt)} chars")


def main() -> None:
    """Run all RAG retrieval tests."""
    logger.info("=== RAG Retrieval System Tests ===")

    # Initialize retriever
    logger.info("Initializing MedicalRetriever...")
    retriever = MedicalRetriever()

    # Run tests
    test_terminology_retrieval(retriever)
    test_guidelines_retrieval(retriever)
    test_report_retrieval(retriever)
    test_rag_context_building(retriever)
    test_prompt_generation(retriever)

    logger.info("\n=== All Tests Complete ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
