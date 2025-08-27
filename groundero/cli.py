#!/usr/bin/env python3
"""
Groundero - Visual grounding search for papers with Zotero integration.

Commands:
    groundero                   # Interactive query mode (default)
    groundero setup             # Setup environment and build database
    groundero update            # Update database from Zotero
    groundero query "question"  # Single query mode
    groundero list              # List documents in database
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv, set_key
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from .groundero_core import (
    build_database,
    run_query,
    run_interactive_query,
    list_documents,
    display_database_status,
    get_database_status,
)

from .install import setup_env_file
from .config import get_env_file_path

console = Console()


# Configuration - Setup warnings and environment
def setup_environment():
    """Setup warnings and environment variables."""
    # Suppress warnings from library cleanup
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="multiprocessing.resource_tracker"
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="milvus_lite",
        message=".*pkg_resources.*",
    )
    warnings.filterwarnings("ignore", message=".*resource_tracker.*leaked semaphore.*")

    os.environ["GRPC_VERBOSITY"] = os.getenv("GRPC_VERBOSITY", "error")
    os.environ["PYTHONWARNINGS"] = (
        "ignore::UserWarning:multiprocessing.resource_tracker"
    )
    os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")


ENV_FILE = get_env_file_path()


def check_first_run():
    """Check if this is the first run by looking for database files."""
    from .config import get_vector_db_path, get_document_store_path

    vector_db_file = Path(get_vector_db_path()) / "docling.db"
    doc_store_file = Path(get_document_store_path())

    # If neither database file exists, this is likely the first run
    return not (vector_db_file.exists() or doc_store_file.exists())


def offer_first_run_setup():
    """Run automatic setup if this appears to be a first run."""
    if check_first_run():
        console.print(
            Panel(
                "First run detected - running initial setup...\n"
                "This will configure your API key and build the initial database.",
                title="First Run Setup",
                border_style="blue",
            )
        )
        console.print()  # Add spacing

        try:
            # First, set up .env file from .env.example if available
            setup_env_file()

            # Then run the full interactive setup command
            setup_command()
            return True
        except Exception as e:
            console.print(f"❌ Setup failed: {e}", style="red")
            console.print("You can run 'groundero setup' to try again.")
            return False
    return True  # Not first run, continue normally


# Database status checking is now handled by groundero_core.display_database_status


def setup_command():
    """Setup environment variables and build database."""
    console.print(Panel("Groundero Setup", border_style="green"))

    # Check for .env file
    env_path = Path(ENV_FILE)

    if env_path.exists():
        console.print(f"Found existing {ENV_FILE}")
        # Reload environment to pick up any existing values
        load_env_file()
    else:
        console.print(f"Creating {ENV_FILE}")
        # Try to copy from .env.example if available
        setup_env_file()
        if env_path.exists():
            # Successfully copied, reload environment
            load_env_file()

    # Get Google API key
    current_key = os.getenv("GOOGLE_API_KEY")
    api_key_changed = False

    if current_key:
        console.print("GOOGLE_API_KEY already set in environment")
        if Confirm.ask("Do you want to update it?"):
            api_key = Prompt.ask("Enter your Google API Key", password=True)
            api_key_changed = True
        else:
            api_key = current_key
    else:
        console.print("Google API Key is required for the LLM queries")
        console.print("Get one at: https://aistudio.google.com/app/apikey")
        api_key = Prompt.ask("Enter your Google API Key", password=True)
        api_key_changed = True

    # Get maximum PDF file size setting
    current_max_size = os.getenv("GROUNDERO_MAX_PDF_SIZE_MB", "5.0")
    console.print(f"Current maximum PDF file size: {current_max_size}MB")
    max_size_changed = False

    if Confirm.ask("Do you want to change the maximum PDF file size?"):
        max_size = Prompt.ask("Enter maximum PDF size in MB", default=current_max_size)
        try:
            float(max_size)  # Validate it's a number
            if max_size != current_max_size:
                set_key(ENV_FILE, "GROUNDERO_MAX_PDF_SIZE_MB", max_size)
                os.environ["GROUNDERO_MAX_PDF_SIZE_MB"] = max_size
                max_size_changed = True
        except ValueError:
            console.print("Invalid size, using default 5.0MB", style="red")
            if "5.0" != current_max_size:
                set_key(ENV_FILE, "GROUNDERO_MAX_PDF_SIZE_MB", "5.0")
                os.environ["GROUNDERO_MAX_PDF_SIZE_MB"] = "5.0"
                max_size_changed = True

    # Only update API key if it changed
    if api_key_changed:
        set_key(ENV_FILE, "GOOGLE_API_KEY", api_key)
        os.environ["GOOGLE_API_KEY"] = api_key

    # Show summary of changes
    changes_made = []
    if api_key_changed:
        changes_made.append("GOOGLE_API_KEY")
    if max_size_changed:
        changes_made.append("GROUNDERO_MAX_PDF_SIZE_MB")

    if changes_made:
        console.print(f"Updated: {', '.join(changes_made)}")
        console.print(f"Environment variables saved to {ENV_FILE}")
    else:
        console.print("No changes made to environment variables")

    # Check if database exists
    status = get_database_status()
    if status["vector_db_exists"] and status["doc_store_exists"]:
        if not Confirm.ask("Database already exists. Rebuild it?"):
            console.print(" Setup complete!")
            return

    # Build database using library function
    console.print("\nStarting database build from Zotero...")
    console.print(
        "This may take a while. While it's running, you can (in a separate terminal):"
    )
    console.print("  • Check progress: [bold cyan]groundero list[/bold cyan]")
    console.print(
        '  • Run queries: [bold cyan]groundero query "your question"[/bold cyan]'
    )
    try:
        result = build_database(from_zotero=True, force_rebuild=False)
        if result["success"]:
            if result["action"] in ["updated", "rebuilt"]:
                console.print("Background processing started!")
                console.print(f"Processing {result['documents_processed']} documents")
            else:
                console.print("Database built successfully!")
                if "documents_processed" in result:
                    console.print(
                        f"Processed {result['documents_processed']} documents"
                    )
        else:
            console.print(f"Database build failed: {result['error']}", style="red")
            return
    except Exception as e:
        console.print(f"Database build failed: {e}", style="red")
        return


def update_command(limit=None, reset=False):
    """Update database from Zotero."""

    if limit:
        console.print(f"Debug mode: Processing maximum {limit} items from Zotero")

    if reset:
        console.print(
            "Reset mode: Will delete existing database and rebuild from scratch"
        )

    try:
        result = build_database(
            from_zotero=True,
            force_rebuild=reset,
            limit=limit,
        )

        if result["success"]:
            if result["action"] == "no_changes":
                console.print("Database is already up to date!")
            elif result["action"] in ["updated", "rebuilt"]:
                console.print(
                    f"Processing {result['documents_processed']} new documents"
                )
            else:
                console.print(f"Database {result['action']} successfully!")
                if "documents_processed" in result:
                    console.print(
                        f"Processed {result['documents_processed']} new documents"
                    )
        else:
            console.print(f"Database update failed: {result['error']}", style="red")
        os._exit(0)
    except Exception as e:
        console.print(f"Database update failed: {e}", style="red")
        os._exit(1)


def query_command(question=None):
    """Run query mode."""
    # Display database status first
    db_ready = display_database_status()
    if not db_ready:
        console.print(" Database not found. Run 'groundero setup' first.", style="red")
        return

    console.print()  # Add spacing after database status

    # Check API key only when actually querying
    if not os.getenv("GOOGLE_API_KEY"):
        console.print(
            " To run queries, you need to set up your Google API key:", style="yellow"
        )
        console.print("   Run 'groundero setup' to configure the API key")
        console.print("\nOther available commands:")
        console.print("   • groundero update  - Update database from Zotero")
        console.print("   • groundero list    - List documents in database")
        console.print("   • groundero help    - Show help")
        return

    try:
        if question:
            # Single query mode
            result = run_query(question)
            if result["success"]:
                from .groundero_core import (
                    display_answer,
                    display_sources_table,
                    display_visual_grounding_info,
                    load_document_store,
                )

                doc_store = load_document_store()
                display_answer(question, result["answer"])
                display_sources_table(result["context"], doc_store)
                display_visual_grounding_info(result["context"], doc_store)
            else:
                console.print(f" Query failed: {result['error']}", style="red")
            os._exit(0)
        else:
            # Interactive mode
            run_interactive_query()
    except Exception as e:
        console.print(f" Query failed: {e}", style="red")
        os._exit(1)


def list_command():
    """List documents in database."""
    try:
        list_documents()
        os._exit(0)
    except Exception as e:
        console.print(f"Failed to list documents: {e}", style="red")
        os._exit(1)


def load_env_file():
    """Load environment variables from .env file."""
    load_dotenv(ENV_FILE)


def main():
    """Main CLI function."""
    # Setup environment first
    setup_environment()

    # Handle help early (before env loading to avoid API key check)
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h", "help"]:
        # Load .env file for help display
        load_env_file()
    else:
        # Load .env file first for normal operations
        load_env_file()

        # Check for first run and offer setup (skip for specific commands that don't need setup)
        skip_first_run_check = len(sys.argv) > 1 and sys.argv[1] in [
            "setup",
            "help",
            "list",
        ]
        if not skip_first_run_check:
            setup_success = offer_first_run_setup()
            if not setup_success:
                # User declined setup or it failed, continue anyway
                pass

    parser = argparse.ArgumentParser(
        description="Groundero - Visual grounding search for academic papers with Zotero integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  groundero                                   # Interactive query mode (default)
  groundero setup                             # Setup environment and build database
  groundero update                            # Update database from Zotero
  groundero query "PINNs training challenges" # Run single query
  groundero list                              # List indexed documents
  groundero help                              # Show this help message
  groundero-install                           # Run initial setup (directories and config)

For more information: https://github.com/rodrigodzf/groundero
        """,
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", metavar="COMMAND"
    )

    # Setup command
    subparsers.add_parser(
        "setup",
        help="Setup environment variables and build initial database",
        description="Configure Google API key (for queries) and build database from Zotero PDFs",
    )

    # Update command
    update_parser = subparsers.add_parser(
        "update",
        help="Update database with new papers from Zotero",
        description="Sync with Zotero to add new papers to the existing database (no API key needed)",
    )
    update_parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of items to process from Zotero (for debugging)",
        default=None,
    )
    update_parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing database and rebuild from scratch",
    )

    # Query command
    query_parser = subparsers.add_parser(
        "query",
        help="Run a single query against the database",
        description="Ask a question and get AI-powered answers with source citations",
    )
    query_parser.add_argument(
        "question", help="Question to ask (enclose in quotes if it contains spaces)"
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of source documents to retrieve (default: 3)",
    )

    # List command
    subparsers.add_parser(
        "list",
        help="List all documents in the database",
        description="Display a table of all indexed documents with metadata",
    )

    # Status command

    # Help command (alias for --help)
    subparsers.add_parser(
        "help",
        help="Show help message",
        description="Display help information about groundero commands",
    )

    args = parser.parse_args()

    try:
        if args.command == "setup":
            setup_command()
        elif args.command == "update":
            update_command(limit=args.limit, reset=args.reset)
        elif args.command == "query":
            query_command(args.question)
        elif args.command == "list":
            list_command()
        elif args.command == "status":
            console.print(
                "Status command removed. Background processing is no longer supported.",
                style="yellow",
            )
        elif args.command == "help":
            parser.print_help()
        else:
            # Default: show database status and run interactive query
            query_command()

    except Exception as e:
        console.print(f" Error: {e}", style="red")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # we get a sys.excepthook somewhere
        # probably because of Milvus, this is a temporary workaround
        os._exit(1)
