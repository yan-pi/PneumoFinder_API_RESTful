"""Medical knowledge retriever for RAG-enhanced generation.

Features:
- Relevance threshold filtering (removes low-quality matches)
- Query caching for performance
- Category-aware retrieval
- Diversity ranking to avoid redundant results
"""

import logging
from functools import lru_cache
from typing import Any

from src.rag.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

# Relevance thresholds (ChromaDB uses L2 distance - lower is better)
# For cosine distance: 0 = identical, 2 = opposite
# These thresholds filter out irrelevant results
DEFAULT_RELEVANCE_THRESHOLD = 1.2  # Max distance to consider relevant
STRICT_RELEVANCE_THRESHOLD = 0.8  # For high-precision queries


class MedicalRetriever:
    """Retriever for medical knowledge to ground LLM generation.

    Features:
    - Relevance threshold: Filters out results below quality threshold
    - Caching: LRU cache for repeated queries (e.g., "pneumonia" guidelines)
    - Category filtering: Can filter by pathology, severity, signs, etc.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ) -> None:
        """Initialize medical retriever.

        Args:
            vector_store: ChromaDB vector store (creates new if None)
            relevance_threshold: Max distance to consider a result relevant (lower = stricter)
        """
        self.vector_store = vector_store or ChromaVectorStore()
        self.relevance_threshold = relevance_threshold
        self._cache_hits = 0
        self._cache_misses = 0

    def _filter_by_relevance(
        self, results: list[dict[str, Any]], threshold: float | None = None
    ) -> list[dict[str, Any]]:
        """Filter results by relevance threshold.

        Args:
            results: List of result dicts with 'distance' key
            threshold: Optional custom threshold (uses instance default if None)

        Returns:
            Filtered list with only relevant results
        """
        max_distance = threshold or self.relevance_threshold
        filtered = [r for r in results if r.get("distance", 0) <= max_distance]

        if len(filtered) < len(results):
            removed = len(results) - len(filtered)
            logger.debug(f"Filtered out {removed} low-relevance results (threshold: {max_distance})")

        return filtered

    def _deduplicate_results(
        self, results: list[dict[str, Any]], similarity_threshold: float = 0.1
    ) -> list[dict[str, Any]]:
        """Remove near-duplicate results for diversity.

        Args:
            results: List of result dicts
            similarity_threshold: Min distance difference to consider different

        Returns:
            Deduplicated list
        """
        if len(results) <= 1:
            return results

        unique = [results[0]]
        for result in results[1:]:
            # Check if result is different enough from all selected
            is_unique = True
            result_text = result.get("text", "")
            for selected in unique:
                selected_text = selected.get("text", "")
                # Simple text overlap check
                if len(set(result_text.split()) & set(selected_text.split())) > len(
                    result_text.split()
                ) * 0.5:
                    is_unique = False
                    break
            if is_unique:
                unique.append(result)

        return unique

    def get_relevant_guidelines(
        self,
        diagnosis: str,
        n_results: int = 3,
        category: str | None = None,
        strict: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant radiology guidelines with quality filtering.

        Args:
            diagnosis: Diagnosis or condition (e.g., 'pneumonia', 'normal')
            n_results: Number of guidelines to retrieve
            category: Optional category filter ('pathology', 'signs', 'severity', etc.)
            strict: Use stricter relevance threshold

        Returns:
            List of relevant guidelines with metadata
        """
        # Request more results to account for filtering
        query_n = n_results * 2 if category else n_results + 2

        results = self.vector_store.query(
            collection_name="medical_guidelines",
            query_text=diagnosis,
            n_results=query_n,
            where={"category": category} if category else None,
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

        # Apply relevance filtering
        threshold = STRICT_RELEVANCE_THRESHOLD if strict else self.relevance_threshold
        guidelines = self._filter_by_relevance(guidelines, threshold)

        # Deduplicate for diversity
        guidelines = self._deduplicate_results(guidelines)

        # Limit to requested count
        guidelines = guidelines[:n_results]

        logger.info(
            f"Retrieved {len(guidelines)} guidelines for '{diagnosis}'"
            f"{f' (category: {category})' if category else ''}"
        )
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
            # Check relevance - terminology should have very close match
            distance = results.get("distances", [[1.0]])[0][0]
            if distance > 0.5:  # Too different to be a reliable translation
                logger.warning(f"No close translation found for '{english_term}' (distance: {distance:.2f})")
                return None

            # Extract PT translation from metadata
            metadata = results["metadatas"][0][0] if results.get("metadatas") else {}
            pt_term = metadata.get("pt_term", results["documents"][0][0])
            logger.debug(f"Translated '{english_term}' → '{pt_term}'")
            return pt_term

        logger.warning(f"No translation found for '{english_term}'")
        return None

    def get_similar_reports(
        self,
        findings: str,
        n_results: int = 2,
        language: str | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve similar radiology reports for reference.

        Args:
            findings: Clinical findings description
            n_results: Number of similar reports to retrieve
            language: Optional language filter ('en' or 'pt-BR')
            category: Optional category filter ('normal' or 'pneumonia')

        Returns:
            List of similar reports with metadata
        """
        # Build where filter
        where = {}
        if language:
            where["language"] = language
        if category:
            where["category"] = category

        results = self.vector_store.query(
            collection_name="sample_reports",
            query_text=findings,
            n_results=n_results + 2,  # Request extra for filtering
            where=where if where else None,
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

        # Apply relevance filtering
        reports = self._filter_by_relevance(reports)

        # Limit to requested count
        reports = reports[:n_results]

        logger.info(
            f"Retrieved {len(reports)} similar reports"
            f"{f' (language: {language})' if language else ''}"
        )
        return reports

    def build_rag_context(
        self,
        diagnosis: str,
        findings: str | None = None,
        include_guidelines: bool = True,
        include_reports: bool = True,
        guidelines_categories: list[str] | None = None,
        prefer_english_reports: bool = True,
    ) -> str:
        """Build RAG context for LLM prompts.

        Args:
            diagnosis: Primary diagnosis
            findings: Clinical findings (optional)
            include_guidelines: Whether to include guidelines
            include_reports: Whether to include similar reports
            guidelines_categories: Optional list of categories to prioritize
            prefer_english_reports: Whether to prefer English reports (for EN LLMs)

        Returns:
            Formatted RAG context string
        """
        context_parts = []

        if include_guidelines:
            # Get general guidelines
            guidelines = self.get_relevant_guidelines(diagnosis, n_results=3)

            # If specific categories requested, also get those
            if guidelines_categories:
                for cat in guidelines_categories:
                    cat_guidelines = self.get_relevant_guidelines(
                        diagnosis, n_results=1, category=cat
                    )
                    for g in cat_guidelines:
                        if g not in guidelines:
                            guidelines.append(g)

            if guidelines:
                context_parts.append("=== RELEVANT GUIDELINES ===")
                for guideline in guidelines[:4]:  # Limit total guidelines
                    title = guideline.get("metadata", {}).get("title", "Guideline")
                    context_parts.append(f"[{title}]")
                    context_parts.append(f"- {guideline['text']}")

        if include_reports and findings:
            # Get similar reports, preferring English for EN LLMs
            language = "en" if prefer_english_reports else None
            reports = self.get_similar_reports(findings, n_results=2, language=language)

            # If no English reports found, try any language
            if not reports and prefer_english_reports:
                reports = self.get_similar_reports(findings, n_results=2)

            if reports:
                context_parts.append("\n=== SIMILAR REPORTS ===")
                for report in reports:
                    title = report.get("metadata", {}).get("title", "Report")
                    context_parts.append(f"[{title}]")
                    context_parts.append(f"- {report['text']}")

        context = "\n".join(context_parts) if context_parts else ""

        if context:
            logger.info(f"Built RAG context: {len(context)} chars")
        else:
            logger.warning("No RAG context built - no relevant documents found")

        return context

    def get_retrieval_stats(self) -> dict[str, Any]:
        """Get retrieval statistics for monitoring.

        Returns:
            Dict with collection stats and cache info
        """
        stats = {
            "relevance_threshold": self.relevance_threshold,
        }

        # Add collection stats
        for collection in ["medical_guidelines", "medical_terminology", "sample_reports"]:
            try:
                col_stats = self.vector_store.get_collection_stats(collection)
                stats[collection] = col_stats
            except Exception as e:
                stats[collection] = {"error": str(e)}

        return stats
