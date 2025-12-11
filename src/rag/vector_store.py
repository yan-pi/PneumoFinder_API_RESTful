"""ChromaDB vector store for medical knowledge base.

This module manages three collections:
1. medical_guidelines: Radiology guidelines (ACR, Fleischner Society, etc.)
2. medical_terminology: EN→PT medical term mappings
3. sample_reports: Example radiology reports for reference
"""

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Vector store for medical knowledge using ChromaDB."""

    def __init__(
        self,
        persist_directory: str = "./database/chroma",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        """Initialize ChromaDB vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data
            embedding_model: Sentence transformer model for embeddings
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Create collections
        self._init_collections()

    def _init_collections(self) -> None:
        """Initialize ChromaDB collections for medical knowledge."""
        collection_configs = [
            {
                "name": "medical_guidelines",
                "metadata": {"description": "Radiology guidelines and clinical protocols"},
            },
            {
                "name": "medical_terminology",
                "metadata": {"description": "EN→PT medical term translations"},
            },
            {
                "name": "sample_reports",
                "metadata": {"description": "Example radiology reports"},
            },
        ]

        for config in collection_configs:
            try:
                collection = self.client.get_or_create_collection(
                    name=config["name"], metadata=config["metadata"]
                )
                logger.info(f"Collection '{config['name']}' ready ({collection.count()} docs)")
            except Exception as e:
                logger.error(f"Failed to create collection {config['name']}: {e}")
                raise

    def add_documents(
        self,
        collection_name: str,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """Add documents to a collection.

        Args:
            collection_name: Name of the collection
            documents: List of document texts
            metadatas: Optional metadata for each document
            ids: Optional IDs for each document (auto-generated if None)
        """
        collection = self.client.get_collection(collection_name)

        # Generate embeddings
        embeddings = self.embedding_model.encode(documents, convert_to_tensor=False).tolist()

        # Auto-generate IDs if not provided
        if ids is None:
            current_count = collection.count()
            ids = [
                f"{collection_name}_{i}"
                for i in range(current_count, current_count + len(documents))
            ]

        # Add to collection
        collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)

        logger.info(f"Added {len(documents)} documents to '{collection_name}'")

    def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Query a collection using semantic search.

        Args:
            collection_name: Name of the collection to query
            query_text: Query text
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            Query results with documents, metadatas, and distances
        """
        collection = self.client.get_collection(collection_name)

        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query_text], convert_to_tensor=False
        ).tolist()

        # Query collection
        results = collection.query(
            query_embeddings=query_embedding, n_results=n_results, where=where
        )

        return results

    def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        """Get statistics for a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Collection statistics (count, metadata)
        """
        collection = self.client.get_collection(collection_name)
        return {
            "name": collection_name,
            "count": collection.count(),
            "metadata": collection.metadata,
        }

    def reset_collection(self, collection_name: str) -> None:
        """Delete and recreate a collection.

        Args:
            collection_name: Name of the collection to reset
        """
        self.client.delete_collection(collection_name)
        self.client.create_collection(collection_name)
        logger.info(f"Reset collection '{collection_name}'")
