"""
Vector database for semantic search of clinical descriptions.
Uses ChromaDB for embedding storage and similarity search.
"""

import chromadb
from chromadb.config import Settings

from src.utils.config import config


class VectorStore:
    """Manages vector embeddings for clinical descriptions using ChromaDB."""

    def __init__(self):
        """Initialize ChromaDB client and collection."""
        self.client = chromadb.PersistentClient(
            path=config.vector_db_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=config.vector_collection,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

    def add_description(self, diagnosis_id: int, description: str, metadata: dict | None = None):
        """
        Add a clinical description to the vector store.

        Args:
            diagnosis_id: Unique ID from diagnoses table
            description: Clinical description text
            metadata: Additional metadata (diagnosis, confidence, etc.)
        """
        self.collection.add(
            documents=[description],
            ids=[str(diagnosis_id)],
            metadatas=[metadata or {}],
        )

    def search_similar(
        self, query: str, top_k: int = 5, filter_metadata: dict | None = None
    ) -> list[dict]:
        """
        Search for similar clinical descriptions.

        Args:
            query: Search query text
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"diagnosis": "PNEUMONIA"})

        Returns:
            List of similar descriptions with metadata and distances
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_metadata,
        )

        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append(
                {
                    "diagnosis_id": int(results["ids"][0][i]),
                    "description": results["documents"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                }
            )

        return formatted_results

    def delete_description(self, diagnosis_id: int):
        """Delete a description from the vector store."""
        try:
            self.collection.delete(ids=[str(diagnosis_id)])
        except Exception:
            pass  # ID might not exist, that's okay

    def get_collection_count(self) -> int:
        """Get total number of descriptions in the vector store."""
        return self.collection.count()


# Global vector store instance
_vector_store = None


def get_vector_store() -> VectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
