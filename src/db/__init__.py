"""
Database layer for PneumoFinder API.
Provides SQLite storage + ChromaDB vector search.
"""

from src.db.database import get_connection, init_database
from src.db.repositories import (
    get_clinical_description,
    get_diagnosis,
    get_recent_diagnoses,
    get_visualization,
    save_clinical_description,
    save_diagnosis,
    save_visualization,
    search_similar_cases,
)
from src.db.vector_store import get_vector_store

__all__ = [
    # Database functions
    "init_database",
    "get_connection",
    # Repository functions
    "save_diagnosis",
    "save_visualization",
    "save_clinical_description",
    "get_diagnosis",
    "get_visualization",
    "get_clinical_description",
    "get_recent_diagnoses",
    "search_similar_cases",
    # Vector store
    "get_vector_store",
]
