"""Populate ChromaDB vector store with medical knowledge base.

This script loads:
1. Medical terminology (EN→PT translations)
2. Radiology guidelines
3. Sample reports

Run with: python scripts/populate_knowledge_base.py
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.knowledge_base.guidelines.radiology_guidelines import (
    get_all_guidelines,
    get_report_examples,
)
from data.knowledge_base.terminology.medical_terms_en_pt import get_all_terms
from src.rag.vector_store import ChromaVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def populate_terminology(vector_store: ChromaVectorStore) -> None:
    """Populate medical terminology collection."""
    logger.info("Populating medical terminology...")

    terms = get_all_terms()

    # Prepare documents and metadata
    documents = []
    metadatas = []

    for en_term, pt_term in terms:
        # Store both EN and PT for bidirectional search
        documents.append(f"{en_term} | {pt_term}")
        metadatas.append(
            {"en_term": en_term, "pt_term": pt_term, "category": "medical_terminology"}
        )

    # Reset collection and add documents
    vector_store.reset_collection("medical_terminology")
    vector_store.add_documents(
        collection_name="medical_terminology", documents=documents, metadatas=metadatas
    )

    logger.info(f"Added {len(documents)} medical terms")


def populate_guidelines(vector_store: ChromaVectorStore) -> None:
    """Populate medical guidelines collection."""
    logger.info("Populating radiology guidelines...")

    guidelines = get_all_guidelines()

    documents = [g["text"] for g in guidelines]
    metadatas = [
        {
            "title": g["title"],
            "source": g["source"],
            "category": g["category"],
        }
        for g in guidelines
    ]

    vector_store.reset_collection("medical_guidelines")
    vector_store.add_documents(
        collection_name="medical_guidelines", documents=documents, metadatas=metadatas
    )

    logger.info(f"Added {len(documents)} guidelines")


def populate_reports(vector_store: ChromaVectorStore) -> None:
    """Populate sample reports collection."""
    logger.info("Populating sample reports...")

    reports = get_report_examples()

    documents = [r["text"] for r in reports]
    metadatas = [
        {
            "title": r["title"],
            "language": r["language"],
            "category": r["category"],
        }
        for r in reports
    ]

    vector_store.reset_collection("sample_reports")
    vector_store.add_documents(
        collection_name="sample_reports", documents=documents, metadatas=metadatas
    )

    logger.info(f"Added {len(documents)} sample reports")


def main() -> None:
    """Main function to populate all collections."""
    logger.info("=== Starting Knowledge Base Population ===")

    # Initialize vector store
    vector_store = ChromaVectorStore()

    # Populate all collections
    populate_terminology(vector_store)
    populate_guidelines(vector_store)
    populate_reports(vector_store)

    # Print statistics
    logger.info("\n=== Population Complete ===")
    for collection in ["medical_terminology", "medical_guidelines", "sample_reports"]:
        stats = vector_store.get_collection_stats(collection)
        logger.info(f"{collection}: {stats['count']} documents")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Failed to populate knowledge base: {e}", exc_info=True)
        sys.exit(1)
