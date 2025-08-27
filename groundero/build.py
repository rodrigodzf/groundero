#!/usr/bin/env python3
"""
Database builder for visual grounding with Docling and LangChain.

This script builds a persistent vector database from PDFs with visual grounding support.
The database can then be used by query.py for fast querying.

Usage:
    python build.py [--pdf PDF_PATH] [--db-path DB_PATH] [--force]
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

from .document_store import DocumentStore
from .zotero_client import ZoteroClient
from .config import get_processing_log_path, get_vector_db_path, get_document_store_path

console = Console()
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")


def setup_processing_logger():
    """Set up logging for document processing."""
    log_path = get_processing_log_path()

    # Create logger
    logger = logging.getLogger("groundero_processing")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler with rotation
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings/errors to console

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def _safe_parse_year(year_str):
    """Safely parse year string to integer, handling 'Unknown' and invalid values."""
    if not year_str or year_str == "Unknown":
        return None
    try:
        return int(year_str)
    except (ValueError, TypeError):
        return None


def get_pdfs_from_zotero(
    client: ZoteroClient,
    limit: int | None = None,
):
    """Get PDFs from Zotero with metadata."""

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Connecting to Zotero...", total=None)

            progress.update(task, description="Finding local PDFs...")
            # Convert limit to max_file_size for the client call
            # The ZoteroClient now uses file size filtering instead of count limits
            max_size_mb = float(os.getenv("GROUNDERO_MAX_PDF_SIZE_MB", "5.0"))
            pdf_metadata = client.find_local_pdfs(max_file_size_mb=max_size_mb)

            # Apply document count limit after file size filtering if specified
            if limit is not None and limit > 0:
                pdf_metadata = pdf_metadata[:limit]

        console.print(f" Found {len(pdf_metadata)} PDFs in Zotero")

        # Create a mapping from PDF path to metadata for processing
        metadata_map = {}
        for item in pdf_metadata:
            pdf_path_str = str(item["pdf_path"])
            metadata_map[pdf_path_str] = {
                "title": item["title"],
                "authors": item["authors"],
                "year": item["year"],
                "attachment_key": item["attachment_key"],
                "citation_key": item["citation_key"],
            }

        console.print(f" Prepared metadata for {len(metadata_map)} documents")

        return pdf_metadata, metadata_map

    except Exception as e:
        console.print(f" Error finding PDFs from Zotero: {e}", style="red")
        raise


def setup_converter():
    """Set up Docling document converter with visual grounding support."""
    console.print(" Setting up document converter...")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False
    pipeline_options.table_structure_options.do_cell_matching = False
    pipeline_options.generate_page_images = False  # Enable for visual grounding
    pipeline_options.images_scale = 2.0

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )
    return converter


def setup_sql():
    """Set up SQLModel engine and create tables."""
    from sqlmodel import create_engine, SQLModel
    from .config import get_document_store_path
    from pathlib import Path

    # Create database path
    db_path = get_document_store_path()
    db_path_obj = Path(db_path)
    db_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Create engine
    engine = create_engine(f"sqlite:///{db_path}")

    # Create tables
    SQLModel.metadata.create_all(engine)

    return engine


def setup_embeddings():
    """Set up HuggingFace embeddings model."""
    logger = logging.getLogger("groundero_processing")
    logger.debug("Initializing HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)
    logger.debug("HuggingFace embeddings initialized successfully")
    return embeddings


def setup_chunker():
    """Set up document chunker."""
    logger = logging.getLogger("groundero_processing")
    logger.debug("Initializing HybridChunker...")
    chunker = HybridChunker()
    logger.debug("HybridChunker initialized successfully")
    return chunker


def setup_zotero_client():
    """Set up Zotero client."""
    return ZoteroClient()


def setup_vectorstore(embeddings, db_path: str):
    """Set up or load existing Milvus vectorstore."""
    logger = logging.getLogger("groundero_processing")
    logger.debug("Setting up Milvus vectorstore...")

    db_path_obj = Path(db_path)
    db_path_obj.mkdir(parents=True, exist_ok=True)
    milvus_uri = str(db_path_obj / "docling.db")
    logger.debug(f"Milvus database path: {milvus_uri}")

    # Check if vector database exists
    vectorstore = None
    if Path(milvus_uri).exists():
        try:
            logger.debug("Loading existing vector database")
            vectorstore = Milvus(
                embedding_function=embeddings,
                collection_name="docling_visual_grounding",
                connection_args={"uri": milvus_uri},
                auto_id=True,
            )
            logger.debug("Existing vector database loaded successfully")
        except Exception as e:
            logger.debug(f"Could not load existing vector database: {e}")
            vectorstore = None

    # If no existing vectorstore, create an empty one ready to receive documents
    if vectorstore is None:
        logger.debug("Creating empty vector database")
        try:
            # Create a simple empty vectorstore connection
            vectorstore = Milvus(
                embedding_function=embeddings,
                collection_name="docling_visual_grounding",
                connection_args={"uri": milvus_uri},
            )
            logger.debug("Empty vector database created successfully")
        except Exception as e:
            logger.debug(f"Could not create empty vector database: {e}")
            # This shouldn't happen, but if it does, we'll create it on first document
            vectorstore = None

    return vectorstore, milvus_uri


def process_documents(
    sources: list[str],
    converter: DocumentConverter,
    zotero_metadata=None,
):
    """Process and load documents with visual grounding metadata."""
    console.print(Panel("Processing Documents", border_style="blue"))

    # Initialize SQLModel document store
    doc_store_engine = setup_sql()
    doc_store = DocumentStore(doc_store_engine)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Converting documents...", total=len(sources))

        for i, source in enumerate(sources):
            progress.update(task, description=f"Converting {Path(source).name}...")

            # Convert document
            dl_doc = converter.convert(source=source).document

            # Get Zotero metadata for this document if available
            metadata = None
            if zotero_metadata:
                source_str = str(source)
                metadata = zotero_metadata.get(source_str)

            # Add to SQLModel store
            authors_list = None
            if metadata and metadata.get("authors"):
                # Convert authors string to list by splitting on comma
                authors_str = metadata.get("authors")
                if isinstance(authors_str, str):
                    authors_list = [author.strip() for author in authors_str.split(",")]
                else:
                    authors_list = authors_str

            doc_store.add_document(
                docling_doc=dl_doc,
                pdf_path=str(source),
                title=metadata.get("title") if metadata else None,
                authors=authors_list,
                year=_safe_parse_year(metadata.get("year")) if metadata else None,
                citation_key=metadata.get("citation_key") if metadata else None,
                attachment_key=metadata.get("attachment_key") if metadata else None,
                unique_id=metadata.get("unique_id") if metadata else None,
            )

            progress.advance(task)

        progress.update(task, description="Loading and chunking documents...")

        # Use a simple tokenizer name instead of the model object
        chunker = HybridChunker()

        loader = DoclingLoader(
            file_path=sources,
            converter=converter,
            export_type=ExportType.DOC_CHUNKS,
            chunker=chunker,
        )

        docs = loader.load()

    console.print(f" Processed {len(docs)} document chunks")
    console.print(" Document store saved to SQLModel database")

    return docs, doc_store


def process_documents_incrementally(
    sources: list[str],
    converter: DocumentConverter,
    doc_store_engine,
    embeddings,
    chunker,
    vectorstore,
    zotero_metadata=None,
    progress_callback: Optional[Callable] = None,
):
    """
    Process documents one by one with immediate database updates.

    Args:
        sources: List of PDF paths to process
        converter: Docling document converter
        doc_store_engine: SQLAlchemy engine for DocumentStore (required)
        embeddings: HuggingFace embeddings model for vector storage
        chunker: Pre-initialized document chunker
        vectorstore: Pre-loaded Milvus vectorstore (required)
        zotero_metadata: Metadata mapping from Zotero
        progress_callback: Optional callback for progress updates

    Returns:
        tuple: (total_chunks_processed, doc_store)
    """
    logger = setup_processing_logger()
    logger.info(f"Starting incremental processing of {len(sources)} documents")
    logger.debug("Setting up document store and embeddings")

    # Initialize SQLModel document store
    logger.debug("Initializing DocumentStore...")
    doc_store = DocumentStore(doc_store_engine)
    logger.debug("DocumentStore initialized successfully")

    # Use pre-loaded components
    logger.debug("Using pre-loaded embeddings and vectorstore...")

    # Vectorstore is required and should already be loaded
    logger.debug(f"Using required vectorstore: {type(vectorstore).__name__}")
    logger.debug(f"Using pre-loaded chunker: {type(chunker).__name__}")

    total_chunks = 0

    # Process documents with Rich Progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing documents...", total=len(sources))

        for i, source in enumerate(sources):
            start_time = datetime.now()
            doc_name = Path(source).name

            # Update progress description
            progress.update(task, description=f"Processing {doc_name}")

            try:
                logger.info(f"Processing document {i + 1}/{len(sources)}: {doc_name}")

                # Convert document
                dl_doc = converter.convert(source=source).document

                logger.debug(f"Converted {doc_name} to Docling document")
                # Get Zotero metadata for this document if available
                metadata = None
                if zotero_metadata:
                    source_str = str(source)
                    metadata = zotero_metadata.get(source_str)

                # Add to SQLModel store
                authors_list = None
                if metadata and metadata.get("authors"):
                    authors_str = metadata.get("authors")
                    if isinstance(authors_str, str):
                        authors_list = [
                            author.strip() for author in authors_str.split(",")
                        ]
                    else:
                        authors_list = authors_str

                doc_store.add_document(
                    docling_doc=dl_doc,
                    pdf_path=str(source),
                    title=metadata.get("title") if metadata else None,
                    authors=authors_list,
                    year=_safe_parse_year(metadata.get("year")) if metadata else None,
                    citation_key=metadata.get("citation_key") if metadata else None,
                    attachment_key=metadata.get("attachment_key") if metadata else None,
                    unique_id=metadata.get("unique_id") if metadata else None,
                )

                # Chunk document immediately
                logger.debug(f"Chunking {doc_name}...")
                loader = DoclingLoader(
                    file_path=[str(source)],
                    converter=converter,
                    export_type=ExportType.DOC_CHUNKS,
                    chunker=chunker,
                )
                docs = loader.load()

                # Add to vector database immediately
                if docs:
                    logger.debug(f"Adding {doc_name} to vector DB...")
                    # Add to existing vectorstore
                    vectorstore.add_documents(docs)
                    total_chunks += len(docs)

                processing_time = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"Completed {doc_name}: {len(docs)} chunks in {processing_time:.1f}s"
                )

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(
                        {
                            "current": i + 1,
                            "total": len(sources),
                            "document": doc_name,
                            "chunks_added": len(docs),
                            "total_chunks": total_chunks,
                            "processing_time": processing_time,
                        }
                    )

                logger.debug(
                    f"Successfully processed {doc_name}: {len(docs)} chunks added"
                )

                # Advance progress
                progress.advance(task)

            except Exception as e:
                logger.error(f"Failed to process {doc_name}: {e}", exc_info=True)
                # Still advance progress even on error
                progress.advance(task)
                continue

    logger.info(
        f"Completed incremental processing: {total_chunks} total chunks across {len(sources)} documents"
    )

    return total_chunks, doc_store


def build_vectorstore(docs, db_path: str):
    """Build and save Milvus vector store."""
    console.print(Panel("Building Vector Search Database", border_style="green"))

    db_path_obj = Path(db_path)
    db_path_obj.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating embeddings...", total=None)

        embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

        # Use the provided database path
        milvus_uri = str(db_path_obj / "docling.db")
        vectorstore = Milvus.from_documents(
            documents=docs,
            embedding=embedding,
            collection_name="docling_visual_grounding",
            connection_args={"uri": milvus_uri},
            index_params={"index_type": "FLAT"},
            drop_old=True,
        )

        progress.update(task, description="Saving database...")

    console.print(f" Vector database saved to: {milvus_uri}")
    return vectorstore, milvus_uri


def add_to_vectorstore(docs, db_path: str):
    """Add documents to existing Milvus vector store."""
    console.print(Panel("Adding Documents to Vector Database", border_style="green"))

    db_path_obj = Path(db_path)
    milvus_uri = str(db_path_obj / "docling.db")

    if not Path(milvus_uri).exists():
        console.print(
            " Vector database doesn't exist. Use build_vectorstore instead.",
            style="red",
        )
        return None, milvus_uri

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Adding new documents...", total=None)

        embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

        # Load existing vectorstore
        vectorstore = Milvus(
            embedding_function=embedding,
            collection_name="docling_visual_grounding",
            connection_args={"uri": milvus_uri},
        )

        # Add new documents to existing collection
        vectorstore.add_documents(docs)

        progress.update(task, description="Saving new documents...")

    console.print(f" Added {len(docs)} new document chunks to vector database")
    return vectorstore, milvus_uri


def display_database_documents():
    """Display all documents in the database as a table that adjusts to terminal size."""

    try:
        from .document_store import DocumentStore

        doc_store_engine = setup_sql()
        doc_store = DocumentStore(doc_store_engine)
        documents = doc_store.get_all_documents()

        if not documents:
            console.print("No documents found in database", style="yellow")
            return

        # Get terminal width and calculate column widths
        terminal_width = console.size.width

        # Reserve space for borders, padding, and separators
        available_width = terminal_width - 10  # Reserve for table borders and padding

        # Define minimum widths and priorities
        min_widths = {
            "num": 3,
            "citation": 25,
            "title": 35,
            "authors": 10,
            "year": 6,
            "added": 10,
        }

        # Calculate remaining width after minimum allocations
        used_width = sum(min_widths.values())
        remaining_width = max(0, available_width - used_width)

        # Distribute remaining width with priorities (title gets most, then citation, then authors)
        title_extra = min(remaining_width // 2, 40)  # Max 40 extra for title
        remaining_width -= title_extra

        citation_extra = min(remaining_width // 2, 15)  # Max 15 extra for citation
        remaining_width -= citation_extra

        authors_extra = remaining_width  # Rest goes to authors

        # Final column widths
        widths = {
            "num": min_widths["num"],
            "citation": min_widths["citation"] + citation_extra,
            "title": min_widths["title"] + title_extra,
            "authors": min_widths["authors"] + authors_extra,
            "year": min_widths["year"],
            "added": min_widths["added"],
        }

        # Create table with dynamic widths
        table = Table(
            title=f"Document Database ({len(documents)} total)",
            show_header=True,
            header_style="bold cyan",
            expand=False,
        )
        table.add_column("#", style="dim", width=widths["num"])
        table.add_column("Citation Key", style="green", width=widths["citation"])
        table.add_column("Title", style="white", width=widths["title"])
        table.add_column("Authors", style="blue", width=widths["authors"])
        table.add_column("Year", style="yellow", width=widths["year"])
        table.add_column("Added", style="dim", width=widths["added"])

        # Sort by citation key for consistent display
        documents.sort(key=lambda x: x.citation_key or "zzz")

        for i, doc in enumerate(documents, 1):
            # Format authors
            authors = doc.get_authors_list()
            if authors and len(authors) > 0:
                # Take first two authors
                authors_str = ", ".join(authors[:2])
                if len(authors) > 2:
                    authors_str += " et al."
            else:
                authors_str = "N/A"

            # Truncate based on available width
            title = doc.title or "N/A"
            if len(title) > widths["title"] - 3:
                title = title[: widths["title"] - 6] + "..."

            # Truncate citation key if needed
            citation_key = doc.citation_key or "N/A"
            if len(citation_key) > widths["citation"] - 3:
                citation_key = citation_key[: widths["citation"] - 6] + "..."

            # Truncate authors if needed
            if len(authors_str) > widths["authors"] - 3:
                authors_str = authors_str[: widths["authors"] - 6] + "..."

            # Format date
            date_str = doc.created_at.strftime("%Y-%m-%d") if doc.created_at else "N/A"

            table.add_row(
                str(i),
                citation_key,
                title,
                authors_str,
                str(doc.year) if doc.year else "N/A",
                date_str,
            )

        console.print(table)

    except Exception as e:
        console.print(f" Error loading database: {e}", style="red")


def validate_pdf(pdf_path: str) -> bool:
    """Validate that PDF file exists."""
    if not Path(pdf_path).exists():
        console.print(f" PDF file not found: {pdf_path}", style="red")
        return False

    if not pdf_path.lower().endswith(".pdf"):
        console.print(f" File is not a PDF: {pdf_path}", style="red")
        return False

    console.print(f" PDF validated: {pdf_path}")
    return True


def check_missing_documents(
    sources: list[str],
    force: bool,
    pdf_metadata: list | None = None,
    doc_store=None,
) -> tuple[list[str], bool]:
    """Check which documents are missing from the database and determine if rebuild is needed."""
    db_path_configured = get_vector_db_path()
    db_file = Path(db_path_configured) / "docling.db"
    doc_store_db = Path(get_document_store_path())
    old_doc_store_path = Path("data/doc_store")  # Legacy path for cleanup only

    # If force rebuild requested, clean everything
    if force:
        console.print("Force rebuild requested", style="yellow")
        if doc_store_db.exists():
            doc_store_db.unlink()
            console.print(" Cleaned document store database")

        if old_doc_store_path.exists():
            import shutil

            shutil.rmtree(old_doc_store_path)
            console.print(" Cleaned legacy document store directory")

        return sources, True  # Process all documents, rebuild vector DB

    # Check if databases exist
    if not doc_store_db.exists():
        console.print("📦 No document store database found - will create new one")
        return (
            sources,
            not db_file.exists(),
        )  # Rebuild vector DB if it also doesn't exist

    # Load existing document store to check what's already there
    try:
        if doc_store is None:
            # Fallback: create DocumentStore if not provided (for backward compatibility)
            from .document_store import DocumentStore

            doc_store_engine = setup_sql()
            doc_store = DocumentStore(doc_store_engine)

        existing_docs = doc_store.get_all_documents()
        existing_paths = {doc.pdf_path for doc in existing_docs if doc.pdf_path}

        console.print(f" Found {len(existing_docs)} documents in existing database")

        # Find missing documents using more robust matching
        missing_sources = []

        # Create mapping of existing documents by unique_id, attachment_key and binary_hash
        existing_attachment_keys = {
            doc.attachment_key for doc in existing_docs if doc.attachment_key
        }
        existing_unique_ids = {doc.unique_id for doc in existing_docs if doc.unique_id}
        {doc.binary_hash for doc in existing_docs}

        # Create mapping from pdf_path to metadata for reliable lookup
        path_to_metadata = {}
        if pdf_metadata:
            for metadata in pdf_metadata:
                pdf_path = str(metadata.get("pdf_path", ""))
                if pdf_path:
                    path_to_metadata[pdf_path] = metadata

        for source in sources:
            source_str = str(source)
            is_missing = True

            # First check if we have metadata to use efficient unique_id matching
            if source_str in path_to_metadata:
                metadata = path_to_metadata[source_str]
                unique_id = metadata.get("unique_id")
                attachment_key = metadata.get("attachment_key")

                # Most efficient check: unique_id (item_key + attachment_size)
                if unique_id and unique_id in existing_unique_ids:
                    is_missing = False
                    console.print(
                        f" Skipping duplicate PDF (smart): {Path(source_str).name}"
                    )
                # Fallback: attachment_key matching
                elif attachment_key and attachment_key in existing_attachment_keys:
                    is_missing = False

            # Fallback to path matching if no metadata match
            if is_missing and source_str in existing_paths:
                is_missing = False

            if is_missing:
                missing_sources.append(source)

        if missing_sources:
            console.print(f" Found {len(missing_sources)} new documents to add:")
            for src in missing_sources[:3]:  # Show first 3
                console.print(f"   • {Path(src).name}")
            if len(missing_sources) > 3:
                console.print(f"   • ... and {len(missing_sources) - 3} more")
        else:
            console.print(" All documents already exist in database")

        # Only rebuild vector DB if it doesn't exist, otherwise we'll add incrementally
        rebuild_vector = not db_file.exists()

        return missing_sources, rebuild_vector

    except Exception as e:
        console.print(f"Error checking existing database: {e}", style="yellow")
        console.print("Will process all documents to be safe")
        return sources, True
