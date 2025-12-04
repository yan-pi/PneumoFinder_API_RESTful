"""
Database connection and session management.
Uses SQLite for structured data.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from src.utils.config import config


def get_db_path() -> Path:
    """Get the database file path."""
    db_path = Path(config.database_dir) / "pneumofinder.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def init_database():
    """Initialize the database schema."""
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    # Create diagnoses table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS diagnoses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_hash TEXT NOT NULL UNIQUE,
            diagnosis TEXT NOT NULL,
            confidence REAL NOT NULL,
            model_version TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_id TEXT,
            metadata TEXT
        )
    """)

    # Create visualizations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS visualizations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            diagnosis_id INTEGER NOT NULL,
            heatmap_blob BLOB,
            overlay_blob BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (diagnosis_id) REFERENCES diagnoses(id) ON DELETE CASCADE
        )
    """)

    # Create clinical_descriptions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clinical_descriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            diagnosis_id INTEGER NOT NULL,
            description_text TEXT NOT NULL,
            llm_model TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (diagnosis_id) REFERENCES diagnoses(id) ON DELETE CASCADE
        )
    """)

    # Create indexes for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_diagnoses_created_at 
        ON diagnoses(created_at DESC)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_diagnoses_hash 
        ON diagnoses(image_hash)
    """)

    conn.commit()
    conn.close()

    print(f"✓ Database initialized at {get_db_path()}")


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Get a database connection with context management."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row  # Enable column access by name
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_cursor(conn: sqlite3.Connection) -> sqlite3.Cursor:
    """Get a cursor from a connection."""
    return conn.cursor()
