"""
Data repository for diagnosis, visualizations, and clinical descriptions.
Provides CRUD operations for the database.
"""

import hashlib
import json
import sqlite3
from datetime import datetime
from typing import Any

from src.db.database import get_connection
from src.db.vector_store import get_vector_store
from src.utils.config import config


def compute_image_hash(image_bytes: bytes) -> str:
    """Compute SHA-256 hash of image bytes for deduplication."""
    return hashlib.sha256(image_bytes).hexdigest()


def save_diagnosis(
    image_hash: str,
    diagnosis: str,
    confidence: float,
    user_id: str | None = None,
    metadata: dict | None = None,
) -> int:
    """
    Save a diagnosis to the database.

    Args:
        image_hash: SHA-256 hash of the image
        diagnosis: NORMAL or PNEUMONIA
        confidence: Prediction confidence (0-1)
        user_id: Optional user/phone identifier
        metadata: Optional additional metadata

    Returns:
        diagnosis_id: ID of the created record
    """
    with get_connection() as conn:
        cursor = conn.cursor()

        # Check if diagnosis already exists
        cursor.execute(
            "SELECT id FROM diagnoses WHERE image_hash = ?",
            (image_hash,),
        )
        existing = cursor.fetchone()

        if existing:
            return existing["id"]

        # Insert new diagnosis
        cursor.execute(
            """
            INSERT INTO diagnoses (image_hash, diagnosis, confidence, model_version, user_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                image_hash,
                diagnosis,
                confidence,
                config.model_version,
                user_id,
                json.dumps(metadata) if metadata else None,
            ),
        )

        return cursor.lastrowid


def save_visualization(diagnosis_id: int, heatmap_bytes: bytes, overlay_bytes: bytes):
    """
    Save Grad-CAM visualizations to the database.

    Args:
        diagnosis_id: Foreign key to diagnoses table
        heatmap_bytes: PNG bytes of heatmap image
        overlay_bytes: PNG bytes of overlay image
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO visualizations (diagnosis_id, heatmap_blob, overlay_blob)
            VALUES (?, ?, ?)
            """,
            (diagnosis_id, heatmap_bytes, overlay_bytes),
        )


def save_clinical_description(
    diagnosis_id: int, description: str, diagnosis: str, confidence: float
):
    """
    Save LLM-generated clinical description to database and vector store.

    Args:
        diagnosis_id: Foreign key to diagnoses table
        description: Clinical description text
        diagnosis: Diagnosis class (for metadata filtering)
        confidence: Confidence score (for metadata filtering)
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO clinical_descriptions (diagnosis_id, description_text, llm_model)
            VALUES (?, ?, ?)
            """,
            (diagnosis_id, description, config.ollama_model),
        )

    # Add to vector store for semantic search
    vector_store = get_vector_store()
    vector_store.add_description(
        diagnosis_id=diagnosis_id,
        description=description,
        metadata={
            "diagnosis": diagnosis,
            "confidence": round(confidence, 2),
            "model": config.ollama_model,
        },
    )


def get_diagnosis(diagnosis_id: int) -> dict[str, Any] | None:
    """Get a diagnosis by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM diagnoses WHERE id = ?", (diagnosis_id,))
        row = cursor.fetchone()

        if row:
            return dict(row)
        return None


def get_visualization(diagnosis_id: int) -> dict[str, bytes] | None:
    """Get visualization blobs by diagnosis ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT heatmap_blob, overlay_blob FROM visualizations WHERE diagnosis_id = ?",
            (diagnosis_id,),
        )
        row = cursor.fetchone()

        if row:
            return {"heatmap": row["heatmap_blob"], "overlay": row["overlay_blob"]}
        return None


def get_clinical_description(diagnosis_id: int) -> str | None:
    """Get clinical description by diagnosis ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT description_text FROM clinical_descriptions WHERE diagnosis_id = ?",
            (diagnosis_id,),
        )
        row = cursor.fetchone()

        if row:
            return row["description_text"]
        return None


def get_recent_diagnoses(limit: int = 10) -> list[dict[str, Any]]:
    """Get most recent diagnoses."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM diagnoses ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]


def search_similar_cases(query: str, top_k: int = 5) -> list[dict]:
    """
    Search for similar past diagnoses using vector similarity.

    Args:
        query: Natural language query (e.g., "pneumonia in lower right lobe")
        top_k: Number of similar cases to return

    Returns:
        List of similar cases with full diagnosis info
    """
    vector_store = get_vector_store()
    similar_descriptions = vector_store.search_similar(query, top_k=top_k)

    # Fetch full diagnosis info for each result
    results = []
    with get_connection() as conn:
        cursor = conn.cursor()
        for desc in similar_descriptions:
            cursor.execute(
                "SELECT * FROM diagnoses WHERE id = ?",
                (desc["diagnosis_id"],),
            )
            row = cursor.fetchone()
            if row:
                results.append(
                    {
                        **dict(row),
                        "description": desc["description"],
                        "similarity_score": 1 - desc["distance"] if desc["distance"] else None,
                    }
                )

    return results
