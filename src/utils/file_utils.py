"""File handling utilities."""

import os
import uuid
from pathlib import Path
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


def save_uploaded_file(file: FileStorage, upload_dir: str = "temp") -> str:
    """
    Save uploaded file to directory with unique name.

    Args:
        file: Uploaded file from Flask request
        upload_dir: Directory to save file to

    Returns:
        Path to saved file
    """
    os.makedirs(upload_dir, exist_ok=True)

    # Generate unique filename to avoid collisions
    original_filename = secure_filename(file.filename or "upload.jpg")
    name, ext = os.path.splitext(original_filename)
    unique_filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"

    filepath = os.path.join(upload_dir, unique_filename)
    file.save(filepath)

    return filepath


def cleanup_file(filepath: str) -> None:
    """
    Remove file if it exists.

    Args:
        filepath: Path to file to remove
    """
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Warning: Could not remove file {filepath}: {e}")


def cleanup_files(*filepaths: str) -> None:
    """
    Remove multiple files if they exist.

    Args:
        filepaths: Paths to files to remove
    """
    for filepath in filepaths:
        cleanup_file(filepath)


def ensure_directory(directory: str) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        directory: Path to directory
    """
    os.makedirs(directory, exist_ok=True)


def get_filename_without_extension(filepath: str) -> str:
    """
    Get filename without extension from path.

    Args:
        filepath: Path to file

    Returns:
        Filename without extension
    """
    return os.path.splitext(os.path.basename(filepath))[0]


def generate_output_paths(input_path: str, output_dir: str, suffixes: list[str]) -> dict[str, str]:
    """
    Generate output file paths based on input filename.

    Args:
        input_path: Path to input file
        output_dir: Output directory
        suffixes: List of suffixes to append (e.g., ['_heatmap', '_overlay'])

    Returns:
        Dictionary mapping suffix to full path
    """
    ensure_directory(output_dir)
    base_name = get_filename_without_extension(input_path)

    return {suffix: os.path.join(output_dir, f"{base_name}{suffix}.png") for suffix in suffixes}
