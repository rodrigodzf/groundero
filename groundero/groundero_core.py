#!/usr/bin/env python3
"""
Core library functions for Groundero - Visual grounding search system.

This module consolidates the database building and querying functionality
into reusable library functions instead of separate scripts.
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import the existing functionality
from .build import (
    get_pdfs_from_zotero,
    setup_converter,
    setup_sql,
    setup_embeddings,
    setup_chunker,
    setup_vectorstore,
    setup_zotero_client,
    process_documents_incrementally,
    check_missing_documents,
    display_database_documents,
)
from .document_store import DocumentStore
from .zotero_client import ZoteroConnectionError
from .query import (
    load_vectorstore,
    load_document_store,
    create_rag_chain,
    display_answer,
    display_sources_table,
    display_visual_grounding_info,
)
from .config import get_vector_db_path

console = Console()


def cleanup_orphaned_documents(current_zotero_pdfs: List[Dict[str, Any]]) -> int:
    """
    Remove documents from database that are no longer in the current Zotero library.

    Args:
        current_zotero_pdfs: List of current PDF metadata from Zotero

    Returns:
        Number of orphaned documents removed
    """
    from .document_store import DocumentStore

    doc_store_engine = setup_sql()
    doc_store = DocumentStore(doc_store_engine)
    all_docs = doc_store.get_all_documents()

    # Get set of current attachment keys from Zotero
    current_attachment_keys = set()
    current_pdf_paths = set()

    for pdf_meta in current_zotero_pdfs:
        if pdf_meta.get("attachment_key"):
            current_attachment_keys.add(pdf_meta["attachment_key"])
        if pdf_meta.get("pdf_path"):
            current_pdf_paths.add(str(pdf_meta["pdf_path"]))

    orphaned_docs = []

    for doc in all_docs:
        # Check if document has attachment key and it's not in current Zotero
        if doc.attachment_key:
            if doc.attachment_key not in current_attachment_keys:
                orphaned_docs.append(doc)
        # For documents without attachment keys, check if PDF path still exists in Zotero
        elif doc.pdf_path:
            if doc.pdf_path not in current_pdf_paths:
                # Also check if the PDF file itself still exists
                if not Path(doc.pdf_path).exists():
                    orphaned_docs.append(doc)

    # Remove orphaned documents
    removed_count = 0
    for doc in orphaned_docs:
        if doc_store.delete_document(doc.binary_hash):
            removed_count += 1

    return removed_count


class GrounderoError(Exception):
    """Base exception for Groundero operations."""

    pass


class DatabaseNotFoundError(GrounderoError):
    """Raised when database is not found."""

    pass


class APIKeyNotSetError(GrounderoError):
    """Raised when Google API key is not set."""

    pass


def check_api_key():
    """Check if Google API key is set."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise APIKeyNotSetError("GOOGLE_API_KEY environment variable is not set")


def get_database_status() -> Dict[str, Any]:
    """Get current database status information."""
    from .config import get_vector_db_path, get_document_store_path

    db_file = Path(get_vector_db_path()) / "docling.db"
    doc_store_file = Path(get_document_store_path())

    status = {
        "vector_db_exists": db_file.exists(),
        "vector_db_path": str(db_file),
        "vector_db_size_mb": 0,
        "doc_store_exists": doc_store_file.exists(),
        "doc_store_path": str(doc_store_file),
        "doc_store_size_mb": 0,
        "document_count": 0,
        "api_key_set": bool(os.getenv("GOOGLE_API_KEY")),
    }

    if db_file.exists():
        status["vector_db_size_mb"] = db_file.stat().st_size / (1024 * 1024)

    if doc_store_file.exists():
        status["doc_store_size_mb"] = doc_store_file.stat().st_size / (1024 * 1024)

        # Get document count
        try:
            from .document_store import DocumentStore

            doc_store_engine = setup_sql()
            doc_store = DocumentStore(doc_store_engine)
            status["document_count"] = doc_store.count_documents()
        except Exception:
            status["document_count"] = 0

    return status


def build_database(
    from_zotero: bool = True,
    pdf_path: Optional[str] = None,
    limit: Optional[int] = None,
    force_rebuild: bool = False,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build or update the vector database.

    Args:
        from_zotero: Whether to get PDFs from Zotero
        pdf_path: Path to single PDF (if not using Zotero)
        limit: Maximum number of PDFs from Zotero (None for all)
        force_rebuild: Force complete rebuild
        db_path: Database path (uses platform default if None)

    Returns:
        Dictionary with build results
    """
    # Use platform-aware path if not specified
    if db_path is None:
        db_path = get_vector_db_path()

    try:
        # Get sources first
        if from_zotero:
            try:
                zotero_client = setup_zotero_client()
            except ZoteroConnectionError as e:
                return {
                    "success": False,
                    "error": str(e),
                    "action": "failed",
                    "error_type": "zotero_connection",
                }

            pdf_metadata, zotero_metadata = get_pdfs_from_zotero(
                zotero_client, limit=limit
            )
            sources = [str(item["pdf_path"]) for item in pdf_metadata]

            # Clean up orphaned documents that are no longer in Zotero
            orphaned_count = cleanup_orphaned_documents(pdf_metadata)
            if orphaned_count > 0:
                console.print(
                    f" Removed {orphaned_count} orphaned documents no longer in Zotero"
                )
        else:
            if not pdf_path or not Path(pdf_path).exists():
                raise GrounderoError(f"PDF file not found: {pdf_path}")
            sources = [pdf_path]
            zotero_metadata = None

        # Handle cleanup if force rebuild requested
        if force_rebuild:
            # Just handle cleanup, don't check database contents
            missing_sources, rebuild_vector = check_missing_documents(
                sources, force_rebuild, pdf_metadata if from_zotero else None
            )

        # Setup all components in main thread AFTER cleanup
        converter = setup_converter()
        doc_store_engine = setup_sql()
        doc_store = DocumentStore(doc_store_engine)
        embeddings = setup_embeddings()
        chunker = setup_chunker()
        vectorstore, milvus_uri = setup_vectorstore(embeddings, db_path)

        # For non-force rebuilds, check what actually needs processing using the real DocumentStore
        if not force_rebuild:
            missing_sources, rebuild_vector = check_missing_documents(
                sources,
                force_rebuild,
                pdf_metadata if from_zotero else None,
                doc_store,
            )

        if not missing_sources and not rebuild_vector:
            return {
                "success": True,
                "action": "no_changes",
                "message": "Database is already up to date",
                "documents_processed": 0,
                "total_documents": len(sources),
            }

        # Process documents using incremental processing
        if missing_sources:
            total_chunks, doc_store = process_documents_incrementally(
                sources=missing_sources,
                converter=converter,
                doc_store_engine=doc_store_engine,
                embeddings=embeddings,
                chunker=chunker,
                vectorstore=vectorstore,
                zotero_metadata=zotero_metadata,
            )
            action = "updated"
            db_uri = str(Path(db_path) / "docling.db")
        elif rebuild_vector:
            # Complete rebuild using incremental processing
            total_chunks, doc_store = process_documents_incrementally(
                sources=sources,
                converter=converter,
                doc_store_engine=doc_store_engine,
                embeddings=embeddings,
                chunker=chunker,
                vectorstore=vectorstore,
                zotero_metadata=zotero_metadata,
            )
            action = "rebuilt"
            db_uri = str(Path(db_path) / "docling.db")
        else:
            total_chunks = 0
            action = "no_changes"
            db_uri = str(Path(db_path) / "docling.db")

        return {
            "success": True,
            "action": action,
            "documents_processed": len(missing_sources)
            if missing_sources
            else len(sources)
            if rebuild_vector
            else 0,
            "total_documents": len(sources),
            "chunks_processed": total_chunks,
            "database_path": db_uri,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "action": "failed",
        }


def run_query(
    vectorstore,
    doc_store,
    question: str,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Run a single query against the database.

    Args:
        vectorstore: Pre-loaded vector store
        doc_store: Pre-loaded document store
        question: Question to ask
        top_k: Number of top sources to retrieve

    Returns:
        Dictionary with query results
    """
    try:
        check_api_key()

        # Create RAG chain
        rag_chain = create_rag_chain(vectorstore, top_k)

        # Run query
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Processing query...", total=None)
            resp_dict = rag_chain.invoke({"input": question})

        return {
            "success": True,
            "question": question,
            "answer": resp_dict["answer"],
            "context": resp_dict["context"],
            "sources_count": len(resp_dict["context"]),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "question": question,
        }


def run_interactive_query(
    default_top_k: int = 5,
    db_path: Optional[str] = None,
):
    """
    Run interactive query mode.

    Args:
        default_top_k: Default number of sources to retrieve
        db_path: Database path (uses platform default if None)
    """
    try:
        check_api_key()

        # Use platform-aware path if not specified
        if db_path is None:
            db_path = get_vector_db_path()

        # Load databases once
        vectorstore = load_vectorstore(db_path)
        if not vectorstore:
            raise DatabaseNotFoundError("Vector database could not be loaded")

        doc_store = load_document_store()
        if not doc_store:
            raise DatabaseNotFoundError("Document store could not be loaded")

        console.print(Panel("Interactive Query Mode", border_style="cyan"))
        console.print("💡 Type 'quit' or 'exit' to stop, 'help' for commands")

        while True:
            try:
                # Get user input
                question = console.input(
                    "\n[bold blue]Your question:[/bold blue] "
                ).strip()

                if not question:
                    continue

                if question.lower() in ["quit", "exit", "q"]:
                    console.print("👋 Goodbye!")
                    break

                if question.lower() == "help":
                    console.print(
                        Panel(
                            "Available commands:\n"
                            "• Type any question to search\n"
                            "• 'quit' or 'exit' to stop\n"
                            "• 'help' to show this message",
                            title="Help",
                            border_style="green",
                        )
                    )
                    continue

                # Run query using the library function
                result = run_query(vectorstore, doc_store, question, default_top_k)

                if result["success"]:
                    # Display results
                    display_answer(question, result["answer"])
                    display_sources_table(result["context"], doc_store)
                    display_visual_grounding_info(result["context"], doc_store)
                else:
                    console.print(f"Query failed: {result['error']}", style="red")

            except KeyboardInterrupt:
                console.print("\n Interactive mode interrupted")
                break
            except Exception as e:
                console.print(f"Error: {e}", style="red")

    except Exception as e:
        console.print(f"Failed to start interactive mode: {e}", style="red")


def list_documents():
    """List all documents in the database."""
    try:
        display_database_documents()
    except Exception as e:
        console.print(f"Failed to list documents: {e}", style="red")


def display_database_status():
    """Display database status information."""
    status = get_database_status()

    # Vector database status
    if status["vector_db_exists"]:
        console.print(f"Vector database size: ({status['vector_db_size_mb']:.1f} MB)")
    else:
        console.print(f"Vector database not found: {escape(status['vector_db_path'])}")

    # Document store status
    if status["doc_store_exists"]:
        console.print(f"Document store size: ({status['doc_store_size_mb']:.1f} MB)")
        console.print(f"Documents indexed: {status['document_count']}")
    else:
        console.print(f"Document store not found: {escape(status['doc_store_path'])}")

    if not status["api_key_set"]:
        console.print("GOOGLE_API_KEY not set, won't use llm")

    return status["vector_db_exists"] and status["doc_store_exists"]
