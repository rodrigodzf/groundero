#!/usr/bin/env python3
"""
Platform-aware configuration for Groundero.

This module provides cross-platform directory paths using platformdirs,
ensuring data is stored in appropriate system locations.
"""

from pathlib import Path
from platformdirs import user_data_dir, user_cache_dir


# Application info
APP_NAME = "groundero"
APP_AUTHOR = "groundero"

# Platform-aware directories
DATA_DIR = Path(user_data_dir(APP_NAME, APP_AUTHOR))
CACHE_DIR = Path(user_cache_dir(APP_NAME, APP_AUTHOR))

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Database paths
VECTOR_DB_PATH = DATA_DIR / "vector_db"
DOCUMENT_STORE_PATH = DATA_DIR / "documents.db"
ZOTERO_METADATA_PATH = DATA_DIR / "zotero_metadata.json"

# Cache paths
PDF_CACHE_PATH = CACHE_DIR / "pdfs"

# Log paths
LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PROCESSING_LOG_PATH = LOG_DIR / "processing.log"

# Environment file path
ENV_FILE_PATH = DATA_DIR / ".env"


def get_vector_db_path() -> str:
    """Get the vector database path."""
    return str(VECTOR_DB_PATH)


def get_document_store_path() -> str:
    """Get the document store database path."""
    return str(DOCUMENT_STORE_PATH)


def get_zotero_metadata_path() -> str:
    """Get the Zotero metadata file path."""
    return str(ZOTERO_METADATA_PATH)


def get_pdf_cache_path() -> str:
    """Get the PDF cache directory path."""
    PDF_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    return str(PDF_CACHE_PATH)


def get_log_dir_path() -> str:
    """Get the log directory path."""
    return str(LOG_DIR)


def get_processing_log_path() -> str:
    """Get the processing log file path."""
    return str(PROCESSING_LOG_PATH)


def get_env_file_path() -> str:
    """Get the environment file path."""
    return str(ENV_FILE_PATH)


def get_data_info() -> dict:
    """Get information about data directories."""
    return {
        "data_dir": str(DATA_DIR),
        "cache_dir": str(CACHE_DIR),
        "vector_db_path": str(VECTOR_DB_PATH),
        "document_store_path": str(DOCUMENT_STORE_PATH),
        "pdf_cache_path": str(PDF_CACHE_PATH),
        "log_dir": str(LOG_DIR),
        "processing_log": str(PROCESSING_LOG_PATH),
        "env_file": str(ENV_FILE_PATH),
    }
