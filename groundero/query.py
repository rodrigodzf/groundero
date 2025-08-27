#!/usr/bin/env python3
"""
Query app for visual grounding with pre-built database.

This script queries a database built by build.py and provides
RAG-based answers with visual grounding information and source tables.

Usage:
    python query.py [--question "Your question"] [--top-k 3] [--db-path DB_PATH]
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from docling.chunking import DocMeta

from .document_store import DocumentStore
from .build import setup_sql

console = Console()

# Configuration
os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")

# Note: API key check moved to functions that actually need it
# This allows imports without requiring the API key to be set

EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")

# Default prompt template for RAG
DEFAULT_PROMPT_TEMPLATE = "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {input}\nAnswer:\n"

PROMPT = PromptTemplate.from_template(
    os.getenv("GROUNDERO_PROMPT_TEMPLATE", DEFAULT_PROMPT_TEMPLATE)
)


def load_vectorstore(db_path: str):
    """Load existing Milvus vector store."""

    db_file = Path(db_path) / "docling.db"
    if not db_file.exists():
        console.print(f"Database not found: {db_file}", style="red")
        console.print("Run [green]groundero update[/green] to add documents")
        return None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading database...", total=None)

        embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

        vectorstore = Milvus(
            embedding_function=embedding,
            collection_name="docling_visual_grounding",
            connection_args={"uri": str(db_file)},
        )

        progress.update(task, description="Validating database...")

        # Quick test to ensure database is accessible
        try:
            vectorstore.similarity_search("test", k=1)
        except Exception as e:
            console.print(f"Database validation failed: {e}", style="red")
            return None

    return vectorstore


def load_document_store():
    """Load SQLModel document store."""

    try:
        engine = setup_sql()
        doc_store = DocumentStore(engine)
        doc_count = doc_store.count_documents()

        if doc_count == 0:
            console.print("No documents found in document store", style="yellow")
            console.print("Run [green]groundero update[/green] to add documents")
        else:
            console.print(f" Loaded document store: {doc_count} documents")

        return doc_store

    except Exception as e:
        console.print(f"Error loading document store: {e}", style="red")
        return None


def create_rag_chain(vectorstore, top_k: int):
    """Create RAG chain."""
    # Check API key here when actually needed
    if not os.getenv("GOOGLE_API_KEY"):
        console.print("GOOGLE_API_KEY environment variable is not set.", style="red")
        console.print(" Run 'python groundero.py setup' to configure your API key")
        return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    question_answer_chain = create_stuff_documents_chain(llm, PROMPT)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


def clip_text(text: str, threshold: int = 100) -> str:
    """Clip text to specified length."""
    return f"{text[:threshold]}..." if len(text) > threshold else text


def display_answer(question: str, answer: str):
    """Display the RAG answer."""
    console.print(
        Panel(
            f"[bold blue]Question:[/bold blue]\n{question}\n\n[bold green]Answer:[/bold green]\n{answer}",
            title=" RAG Response",
            border_style="cyan",
        )
    )


def display_sources_table(context_docs, doc_store):
    """Display sources in a Rich table with visual grounding info that adjusts to terminal size."""
    # Get terminal width and calculate column widths
    terminal_width = console.size.width

    # Reserve space for borders, padding, and separators
    available_width = terminal_width - 10  # Reserve for table borders and padding

    # Define minimum widths and priorities
    min_widths = {
        "source": 8,
        "content": 30,
        "document": 25,  # Increased for citation keys
        "pages": 8,
        "visual": 12,
    }

    # Calculate remaining width after minimum allocations
    used_width = sum(min_widths.values())
    remaining_width = max(0, available_width - used_width)

    # Distribute remaining width with priorities (document gets priority for citation keys, then content)
    document_extra = min(remaining_width // 3, 30)  # Max 30 extra for document
    remaining_width -= document_extra

    content_extra = min(remaining_width // 2, 40)  # Max 40 extra for content
    remaining_width -= content_extra

    pages_extra = remaining_width  # Rest goes to pages

    # Final column widths
    widths = {
        "source": min_widths["source"],
        "content": min_widths["content"] + content_extra,
        "document": min_widths["document"] + document_extra,
        "pages": min_widths["pages"] + pages_extra,
        "visual": min_widths["visual"],
    }

    table = Table(
        title=" Retrieved Sources",
        show_header=True,
        header_style="bold magenta",
        expand=False,
    )
    table.add_column("Source #", style="dim", width=widths["source"])
    table.add_column("Content Preview", style="white", width=widths["content"])
    table.add_column("Document", style="cyan", width=widths["document"])
    table.add_column("Pages", style="green", width=widths["pages"])
    table.add_column("Visual Elements", style="yellow", width=widths["visual"])

    for i, doc in enumerate(context_docs):
        try:
            meta = DocMeta.model_validate(doc.metadata["dl_meta"])

            # Get document info from SQLModel store
            doc_name = "Unknown"
            doc_record = doc_store.get_document_by_binary_hash(
                str(meta.origin.binary_hash)
            )

            if doc_record:
                if doc_record.citation_key:
                    doc_name = doc_record.citation_key
                elif doc_record.title and doc_record.year:
                    doc_name = f"{doc_record.title} ({doc_record.year})"
                elif doc_record.title:
                    doc_name = doc_record.title
                elif doc_record.filename:
                    doc_name = doc_record.filename

            # Fallback to filename from metadata
            if doc_name == "Unknown":
                doc_name = meta.origin.filename or "Unknown"

            content_preview = clip_text(
                doc.page_content, threshold=widths["content"] - 3
            )

            # Truncate document name if needed
            if len(doc_name) > widths["document"] - 3:
                doc_name = doc_name[: widths["document"] - 6] + "..."

            # Get page numbers with line ranges and visual elements
            page_line_info = {}
            visual_elements = 0

            for doc_item in meta.doc_items:
                if doc_item.prov:
                    prov = doc_item.prov[0]
                    page_no = prov.page_no

                    # Get line information if available
                    if (
                        hasattr(prov, "bbox")
                        and hasattr(prov.bbox, "l")
                        and hasattr(prov.bbox, "b")
                    ):
                        # Use bbox coordinates as line approximation
                        line_start = int(prov.bbox.l)
                        line_end = int(prov.bbox.b)

                        if page_no not in page_line_info:
                            page_line_info[page_no] = set()
                        page_line_info[page_no].add((line_start, line_end))
                    else:
                        # Fallback to just page number if no line info
                        if page_no not in page_line_info:
                            page_line_info[page_no] = set()

                    visual_elements += 1

            # Format page and line information
            if page_line_info:
                page_strs = []
                for page_no in sorted(page_line_info.keys()):
                    lines = page_line_info[page_no]
                    if lines:
                        # Group consecutive line ranges
                        line_ranges = sorted(lines)
                        if len(line_ranges) == 1:
                            line_start, line_end = line_ranges[0]
                            if line_start == line_end:
                                page_strs.append(f"p.{page_no},ln.{line_start}")
                            else:
                                page_strs.append(
                                    f"p.{page_no},ln.{line_start}-{line_end}"
                                )
                        else:
                            # Multiple ranges on same page
                            range_strs = []
                            for line_start, line_end in line_ranges:
                                if line_start == line_end:
                                    range_strs.append(str(line_start))
                                else:
                                    range_strs.append(f"{line_start}-{line_end}")
                            page_strs.append(f"p.{page_no},ln.{','.join(range_strs)}")
                    else:
                        page_strs.append(f"p.{page_no}")
                pages_str = ", ".join(page_strs)
            else:
                pages_str = "N/A"

            # Truncate pages string if needed
            if len(pages_str) > widths["pages"] - 3:
                pages_str = pages_str[: widths["pages"] - 6] + "..."

            visual_str = f"{visual_elements} regions" if visual_elements > 0 else "None"

            # Truncate visual string if needed
            if len(visual_str) > widths["visual"] - 3:
                visual_str = visual_str[: widths["visual"] - 6] + "..."

            table.add_row(f"{i + 1}", content_preview, doc_name, pages_str, visual_str)

        except Exception as e:
            # Fallback for documents without proper metadata
            error_msg = f"Error: {str(e)[:20]}..."
            if len(error_msg) > widths["visual"] - 3:
                error_msg = error_msg[: widths["visual"] - 6] + "..."

            table.add_row(
                f"{i + 1}",
                clip_text(doc.page_content, threshold=widths["content"] - 3),
                "Unknown",
                "N/A",
                error_msg,
            )

    console.print(table)


def display_visual_grounding_info(
    context_docs,
    doc_store,
):
    """Display visual grounding information."""

    total_regions = 0
    total_pages = set()

    for i, doc in enumerate(context_docs):
        try:
            meta = DocMeta.model_validate(doc.metadata["dl_meta"])

            # Get document info from SQLModel store
            filename = f"Document {i + 1}"
            doc_record = doc_store.get_document_by_binary_hash(
                str(meta.origin.binary_hash)
            )

            if doc_record:
                if doc_record.citation_key:
                    filename = doc_record.citation_key
                elif doc_record.title and doc_record.year:
                    filename = f"{doc_record.title} ({doc_record.year})"
                elif doc_record.title:
                    filename = doc_record.title
                elif doc_record.filename:
                    filename = doc_record.filename

            # Fallback to filename from metadata
            if filename == f"Document {i + 1}":
                filename = meta.origin.filename or f"Document {i + 1}"

            doc_regions = 0
            doc_pages = set()

            for doc_item in meta.doc_items:
                if doc_item.prov:
                    prov = doc_item.prov[0]
                    doc_pages.add(prov.page_no)
                    doc_regions += 1
                    total_regions += 1

            total_pages.update(doc_pages)

            if not doc_pages:
                console.print("No grounding data available")

        except Exception as e:
            console.print(f"Error processing data: {str(e)}")

    console.print(
        f"\n[bold green]Summary:[/bold green] {total_regions} total regions across {len(total_pages)} pages"
    )


def run_query(vectorstore, doc_store, question: str, top_k: int):
    """Run a single query against the database."""
    console.print(Panel("Running Query", border_style="magenta"))

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

    # Display results
    display_answer(question, resp_dict["answer"])
    display_sources_table(resp_dict["context"], doc_store)
    display_visual_grounding_info(resp_dict["context"], doc_store)

    return resp_dict


def run_interactive_mode(vectorstore, doc_store, default_top_k: int):
    """Run interactive query mode."""
    console.print(Panel("Interactive Query Mode", border_style="cyan"))
    console.print(" Type 'quit' or 'exit' to stop, 'help' for commands")

    while True:
        try:
            # Get user input
            question = console.input("\n[bold blue]Your question:[/bold blue] ").strip()

            if not question:
                continue

            if question.lower() in ["quit", "exit", "q"]:
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

            # Run query
            run_query(vectorstore, doc_store, question, default_top_k)

        except KeyboardInterrupt:
            console.print("\n Interactive mode interrupted")
            break
        except Exception as e:
            console.print(f"Error: {e}", style="red")
