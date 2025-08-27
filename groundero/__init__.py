"""
Groundero - Visual grounding search for academic papers with Zotero integration.

This package provides tools for processing PDFs with visual grounding support
and building vector search indexes using Docling and LangChain.
"""

__version__ = "0.1.0"

# Import main components for easy access
from .zotero_client import ZoteroClient

# Define exports
__all__ = [
    "ZoteroClient",
]


def main():
    """Entry point for the groundero CLI."""
    from .cli import main as cli_main

    cli_main()
