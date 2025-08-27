"""Tests for configuration module."""

from pathlib import Path
from groundero.config import (
    get_vector_db_path,
    get_document_store_path,
    get_pdf_cache_path,
    get_log_dir_path,
    get_data_info,
)


def test_get_vector_db_path():
    """Test vector database path generation."""
    path = get_vector_db_path()
    assert isinstance(path, str)
    assert len(path) > 0


def test_get_document_store_path():
    """Test document store path generation."""
    path = get_document_store_path()
    assert isinstance(path, str)
    assert path.endswith("documents.db")


def test_get_pdf_cache_path():
    """Test PDF cache path generation."""
    path = get_pdf_cache_path()
    assert isinstance(path, str)
    assert Path(path).exists()  # Should be created by the function


def test_get_log_dir_path():
    """Test log directory path generation."""
    path = get_log_dir_path()
    assert isinstance(path, str)
    assert Path(path).exists()  # Should exist from config


def test_get_data_info():
    """Test data info dictionary."""
    info = get_data_info()
    assert isinstance(info, dict)

    required_keys = [
        "data_dir",
        "cache_dir",
        "vector_db_path",
        "document_store_path",
        "pdf_cache_path",
        "log_dir",
        "processing_log",
    ]

    for key in required_keys:
        assert key in info
        assert isinstance(info[key], str)
        assert len(info[key]) > 0
