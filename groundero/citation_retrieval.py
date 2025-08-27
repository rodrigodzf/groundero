#!/usr/bin/env python3
"""
Citation key-based document retrieval utility.

This script provides convenient functions to retrieve documents by citation key
and other metadata from the SQLModel document store.
"""

import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .document_store import DocumentStore

console = Console()


def list_documents(doc_store: DocumentStore, limit: int = 20):
    """List all documents with their metadata."""
    documents = doc_store.get_all_documents()

    if not documents:
        console.print("No documents found in the database.", style="yellow")
        return

    table = Table(
        title=f"Document Database ({len(documents)} total, showing {min(limit, len(documents))})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Citation Key", style="green", width=20)
    table.add_column("Title", style="white", width=40)
    table.add_column("Authors", style="blue", width=25)
    table.add_column("Year", style="yellow", width=6)
    table.add_column("Binary Hash", style="dim", width=12)

    for doc in documents[:limit]:
        authors_str = ", ".join(doc.get_authors_list()[:2])  # First 2 authors
        if len(doc.get_authors_list()) > 2:
            authors_str += " et al."

        table.add_row(
            doc.citation_key or "N/A",
            (doc.title[:37] + "...")
            if doc.title and len(doc.title) > 40
            else (doc.title or "N/A"),
            authors_str or "N/A",
            str(doc.year) if doc.year else "N/A",
            doc.binary_hash[:8] + "...",
        )

    console.print(table)

    if len(documents) > limit:
        console.print(
            f"\n... and {len(documents) - limit} more documents. Use --limit to see more."
        )


def search_by_citation_key(doc_store: DocumentStore, citation_key: str):
    """Search for a document by citation key."""
    doc = doc_store.get_document_by_citation_key(citation_key)

    if not doc:
        console.print(
            f" No document found with citation key: {citation_key}", style="red"
        )
        return None

    console.print(
        Panel(
            f"[bold green]Citation Key:[/bold green] {doc.citation_key}\n"
            f"[bold blue]Title:[/bold blue] {doc.title or 'N/A'}\n"
            f"[bold cyan]Authors:[/bold cyan] {', '.join(doc.get_authors_list()) or 'N/A'}\n"
            f"[bold yellow]Year:[/bold yellow] {doc.year or 'N/A'}\n"
            f"[bold magenta]PDF Path:[/bold magenta] {doc.pdf_path or 'N/A'}\n"
            f"[bold dim]Binary Hash:[/bold dim] {doc.binary_hash}\n"
            f"[bold dim]Added:[/bold dim] {doc.created_at}",
            title=f" Document: {citation_key}",
            border_style="green",
        )
    )

    return doc


def search_by_author(doc_store: DocumentStore, author_name: str):
    """Search for documents by author name."""
    documents = doc_store.get_documents_by_author(author_name)

    if not documents:
        console.print(f" No documents found by author: {author_name}", style="red")
        return []

    console.print(f" Found {len(documents)} documents by '{author_name}':")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Citation Key", style="green")
    table.add_column("Title", style="white", width=50)
    table.add_column("Year", style="yellow")

    for doc in documents:
        table.add_row(
            doc.citation_key or "N/A",
            (doc.title[:47] + "...")
            if doc.title and len(doc.title) > 50
            else (doc.title or "N/A"),
            str(doc.year) if doc.year else "N/A",
        )

    console.print(table)
    return documents


def search_by_year(doc_store: DocumentStore, year: int):
    """Search for documents by publication year."""
    documents = doc_store.get_documents_by_year(year)

    if not documents:
        console.print(f" No documents found from year: {year}", style="red")
        return []

    console.print(f"Found {len(documents)} documents from {year}:")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Citation Key", style="green")
    table.add_column("Title", style="white", width=50)
    table.add_column("Authors", style="blue", width=30)

    for doc in documents:
        authors_str = ", ".join(doc.get_authors_list()[:2])
        if len(doc.get_authors_list()) > 2:
            authors_str += " et al."

        table.add_row(
            doc.citation_key or "N/A",
            (doc.title[:47] + "...")
            if doc.title and len(doc.title) > 50
            else (doc.title or "N/A"),
            authors_str or "N/A",
        )

    console.print(table)
    return documents


def get_document_content(doc_store: DocumentStore, citation_key: str):
    """Get and display the full Docling document content."""
    doc_record = doc_store.get_document_by_citation_key(citation_key)

    if not doc_record:
        console.print(
            f" No document found with citation key: {citation_key}", style="red"
        )
        return None

    try:
        docling_doc = doc_record.get_docling_document()
        console.print(f" Retrieved Docling document for: {citation_key}")
        console.print(f"Document has {len(docling_doc.texts)} text elements")

        # Display first few text elements as preview
        console.print(Panel("Content Preview", border_style="blue"))
        for i, text_element in enumerate(docling_doc.texts[:3]):
            console.print(f"[bold]Element {i + 1}:[/bold] {text_element.text[:200]}...")

        if len(docling_doc.texts) > 3:
            console.print(f"... and {len(docling_doc.texts) - 3} more text elements")

        return docling_doc

    except Exception as e:
        console.print(f" Error loading document content: {e}", style="red")
        return None


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Citation Key Document Retrieval")
    parser.add_argument("--list", action="store_true", help="List all documents")
    parser.add_argument(
        "--citation-key", type=str, help="Find document by citation key"
    )
    parser.add_argument("--author", type=str, help="Find documents by author name")
    parser.add_argument("--year", type=int, help="Find documents by publication year")
    parser.add_argument("--content", type=str, help="Get full content for citation key")
    parser.add_argument(
        "--limit", type=int, default=20, help="Limit results (default: 20)"
    )

    args = parser.parse_args()

    if not any([args.list, args.citation_key, args.author, args.year, args.content]):
        args.list = True  # Default to listing documents

    console.print(
        Panel(
            "Citation Key Document Retrieval",
            subtitle="SQLModel Document Store",
            border_style="bold blue",
        )
    )

    try:
        # Initialize document store
        from .build import setup_sql

        doc_store_engine = setup_sql()
        doc_store = DocumentStore(doc_store_engine)

        if args.list:
            list_documents(doc_store, args.limit)

        if args.citation_key:
            search_by_citation_key(doc_store, args.citation_key)

        if args.author:
            search_by_author(doc_store, args.author)

        if args.year:
            search_by_year(doc_store, args.year)

        if args.content:
            get_document_content(doc_store, args.content)

        return 0

    except Exception as e:
        console.print(f" Error: {e}", style="red")
        return 1


if __name__ == "__main__":
    exit(main())
