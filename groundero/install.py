#!/usr/bin/env python3
"""
Post-install setup for Groundero.

This module handles initial setup after package installation:
- Creates platform-specific directories
- Copies .env.example to .env if it doesn't exist
- Shows welcome message
"""

import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from .config import (
    get_vector_db_path,
    get_document_store_path,
    get_env_file_path,
    get_processing_log_path,
)

console = Console()


def create_directories():
    """Create all necessary platform-specific directories."""
    directories = [
        Path(get_vector_db_path()).parent,  # Gets the parent of the db file path
        Path(get_document_store_path()).parent,  # Gets the parent of the db file path
        Path(get_processing_log_path()).parent,  # Gets the parent of the log file path
    ]

    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            console.print(
                f"Could not create directory {directory}: {e}", style="yellow"
            )


def setup_env_file():
    """Copy .env.example to .env if .env doesn't exist."""
    cwd = Path.cwd()
    env_example = cwd / ".env.example"
    env_file = Path(get_env_file_path())

    # Check if we're in a development environment (has .env.example)
    if env_example.exists() and not env_file.exists():
        try:
            shutil.copy2(env_example, env_file)
            return True
        except Exception as e:
            console.print(f"Could not create .env file: {e}", style="yellow")
            return False
    elif env_file.exists():
        console.print("✓ .env file already exists", style="dim green")
        return True
    else:
        # In installed package, .env.example might not be available
        console.print(
            "No .env.example found (normal for installed packages)", style="dim blue"
        )
        return False


def show_welcome_message():
    """Show welcome message with next steps."""
    console.print(
        Panel(
            "Initial setup completed successfully.\n\n"
            "Next steps:\n"
            "1. Run 'groundero setup' to configure your API key and build database\n"
            "2. Use 'groundero' to start querying your documents\n"
            "3. Run 'groundero help' for more options\n\n"
            "Documentation: https://github.com/rodrigodzf/groundero",
            title="Groundero Setup Complete",
            border_style="bold green",
        )
    )


def post_install_setup():
    """Run post-install setup."""
    try:
        # Create directories
        create_directories()

        # Setup .env file
        env_setup = setup_env_file()

        # Show welcome message
        console.print()  # Add spacing
        show_welcome_message()

        if not env_setup:
            console.print(
                "\n💡 Tip: Create a .env file with your configuration variables",
                style="dim blue",
            )

        return True

    except Exception as e:
        console.print(f"❌ Post-install setup failed: {e}", style="red")
        return False


def main():
    """Entry point for post-install setup."""
    success = post_install_setup()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
