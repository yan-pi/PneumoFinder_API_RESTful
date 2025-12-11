"""Medical knowledge retriever for RAG-enhanced generation."""

import logging
from typing import Any

from src.rag.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class MedicalRetriever:
    """Retriever for medical knowledge to ground LLM generation."""

    def __init__(self, vector_store: ChromaVectorStore | None = None) -> None:
        """Initialize medical retriever.

        Args:
            vector_store: ChromaDB vector store (creates new if None)
        """
        self.vector_store = vector_store or ChromaVectorStore()

    def get_relevant_guidelines(self, diagnosis: str, n_results: int = 3) -> list[dict[str, Any]]:
        """Retrieve relevant radiology guidelines.

        Args:
            diagnosis: Diagnosis or condition (e.g., 'pneumonia', 'normal')
            n_results: Number of guidelines to retrieve

        Returns:
            List of relevant guidelines with metadata
        """
        results = self.vector_store.query(
            collection_name="medical_guidelines", query_text=diagnosis, n_results=n_results
        )

        guidelines = []
        if results.get("documents"):
            for i, doc in enumerate(results["documents"][0]):
                guidelines.append(
                    {
                        "text": doc,
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                        "distance": results["distances"][0][i] if results.get("distances") else 0.0,
                    }
                )

        logger.info(f"Retrieved {len(guidelines)} guidelines for '{diagnosis}'")
        return guidelines

    def translate_term(self, english_term: str) -> str | None:
        """Translate medical term from English to Portuguese.

        Args:
            english_term: English medical term

        Returns:
            Portuguese translation or None if not found
        """
        results = self.vector_store.query(
            collection_name="medical_terminology",
            query_text=english_term,
            n_results=1,
        )

        if results.get("documents") and len(results["documents"][0]) > 0:
            # Extract PT translation from metadata
            metadata = results["metadatas"][0][0] if results.get("metadatas") else {}
            pt_term = metadata.get("pt_term", results["documents"][0][0])
            logger.info(f"Translated '{english_term}' → '{pt_term}'")
            return pt_term

        logger.warning(f"No translation found for '{english_term}'")
        return None

    def get_similar_reports(self, findings: str, n_results: int = 2) -> list[dict[str, Any]]:
        """Retrieve similar radiology reports for reference.

        Args:
            findings: Clinical findings description
            n_results: Number of similar reports to retrieve

        Returns:
            List of similar reports with metadata
        """
        results = self.vector_store.query(
            collection_name="sample_reports", query_text=findings, n_results=n_results
        )

        reports = []
        if results.get("documents"):
            for i, doc in enumerate(results["documents"][0]):
                reports.append(
                    {
                        "text": doc,
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                        "distance": results["distances"][0][i] if results.get("distances") else 0.0,
                    }
                )

        logger.info(f"Retrieved {len(reports)} similar reports")
        return reports

    def build_rag_context(
        self,
        diagnosis: str,
        findings: str | None = None,
        include_guidelines: bool = True,
        include_reports: bool = True,
    ) -> str:
        """Build RAG context for LLM prompts.

        Args:
            diagnosis: Primary diagnosis
            findings: Clinical findings (optional)
            include_guidelines: Whether to include guidelines
            include_reports: Whether to include similar reports

        Returns:
            Formatted RAG context string
        """
        context_parts = []

        if include_guidelines:
            guidelines = self.get_relevant_guidelines(diagnosis, n_results=2)
            if guidelines:
                context_parts.append("=== RELEVANT GUIDELINES ===")
                for guideline in guidelines:
                    context_parts.append(f"- {guideline['text']}")

        if include_reports and findings:
            reports = self.get_similar_reports(findings, n_results=1)
            if reports:
                context_parts.append("\n=== SIMILAR REPORTS ===")
                for report in reports:
                    context_parts.append(f"- {report['text']}")

        return "\n".join(context_parts) if context_parts else ""
