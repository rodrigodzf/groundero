"""
Zotero client for PDF retrieval and management.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from pyzotero import zotero
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class ZoteroConnectionError(Exception):
    """Raised when unable to connect to Zotero."""

    pass


class ZoteroClient:
    """Client for connecting to Zotero and retrieving PDF attachments."""

    def __init__(self):
        """Initialize Zotero client with environment variables."""
        self.library_id = os.getenv("ZOTERO_LIBRARY_ID")
        self.library_type = os.getenv("ZOTERO_LIBRARY_TYPE", "user")
        self.api_key = os.getenv("ZOTERO_API_KEY")
        self.local = os.getenv("ZOTERO_LOCAL", "").lower() in ["true", "yes", "1"]

        # For local API, default to user ID 0 if not specified
        if self.local and not self.library_id:
            self.library_id = "0"

        # For remote API, we need both library_id and api_key
        # For local API, we only need ZOTERO_LOCAL=true (api_key not required)
        if not self.local and not (self.library_id and self.api_key):
            raise ValueError(
                "Missing required environment variables. Please set ZOTERO_LIBRARY_ID and ZOTERO_API_KEY, "
                "or use ZOTERO_LOCAL=true for local Zotero instance."
            )

        self.client = zotero.Zotero(
            library_id=self.library_id,
            library_type=self.library_type,
            api_key=self.api_key,
            local=self.local,
        )

        # Test connectivity
        self._test_connection()

    def _test_connection(self):
        """Test connection to Zotero and raise exception if unreachable."""
        try:
            # Try to get the number of items to verify connection
            self.client.num_items()
            logger.info("Successfully connected to Zotero")
        except Exception as e:
            error_msg = f"Unable to connect to Zotero: {e}"
            if self.local:
                error_msg += (
                    "\n\nMake sure Zotero is running locally with the local API enabled.\n"
                    "Go to: Settings -> Advanced -> Config Editor -> "
                    "extensions.zotero.httpServer.localAPI.enabled = True"
                )
            else:
                error_msg += f"\n\nCheck your ZOTERO_LIBRARY_ID ({self.library_id}) and ZOTERO_API_KEY."
            logger.error(error_msg)
            raise ZoteroConnectionError(error_msg) from e

    def get_all_items(self, limit: int | None = None) -> List[Dict[str, Any]]:
        """Get all items from Zotero library."""
        try:
            # If no limit specified, get all items
            if limit is None:
                items = self.client.everything(self.client.items())
            else:
                items = self.client.items(limit=limit)
            logger.info(f"Retrieved {len(items)} items from Zotero")
            return items
        except Exception as e:
            logger.error(f"Error retrieving items: {e}")
            return []

    def get_pdf_attachments(self, item_key: str) -> List[Dict[str, Any]]:
        """Get PDF attachments for a specific item."""
        try:
            children = self.client.children(item_key)
            pdf_attachments = []

            for child in children:
                if (
                    child.get("data", {}).get("itemType") == "attachment"
                    and child.get("data", {}).get("contentType") == "application/pdf"
                ):
                    pdf_attachments.append(child)

            return pdf_attachments
        except Exception as e:
            logger.error(f"Error retrieving attachments for {item_key}: {e}")
            return []

    def download_pdf(self, attachment_key: str, download_dir: Path) -> Optional[Path]:
        """Download a PDF attachment to the specified directory."""
        try:
            # Get attachment info
            attachment = self.client.item(attachment_key)
            filename = attachment.get("data", {}).get(
                "filename", f"{attachment_key}.pdf"
            )

            # Ensure filename ends with .pdf
            if not filename.endswith(".pdf"):
                filename += ".pdf"

            # Create safe filename
            safe_filename = "".join(
                c for c in filename if c.isalnum() or c in "._- "
            ).strip()
            file_path = download_dir / safe_filename

            # Download the file using the correct dump parameters
            import tempfile
            import shutil

            with tempfile.TemporaryDirectory() as tmpdir:
                # Use dump with filename and path parameters like zotero-mcp
                self.client.dump(attachment_key, filename=safe_filename, path=tmpdir)

                # Check if file was created
                temp_file_path = Path(tmpdir) / safe_filename
                if temp_file_path.exists():
                    # Move to final destination
                    shutil.copy2(temp_file_path, file_path)
                    logger.info(f"Downloaded PDF: {file_path}")
                    return file_path
                else:
                    logger.error(f"File not created after dump: {temp_file_path}")
                    return None

        except Exception as e:
            logger.error(f"Error downloading PDF {attachment_key}: {e}")
            return None

    def find_and_download_pdfs(
        self, download_dir: Path, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Find all PDF attachments and download them."""
        download_dir.mkdir(parents=True, exist_ok=True)
        downloaded_pdfs = []

        # Get all items
        items = self.get_all_items(limit=limit)

        for item in items:
            item_data = item.get("data", {})
            item_key = item_data.get("key")

            if not item_key:
                continue

            # Get PDF attachments for this item
            pdf_attachments = self.get_pdf_attachments(item_key)

            for attachment in pdf_attachments:
                attachment_key = attachment.get("data", {}).get("key")
                if not attachment_key:
                    continue

                # Download the PDF
                pdf_path = self.download_pdf(attachment_key, download_dir)

                if pdf_path:
                    # Get attachment size from parent item links if available
                    attachment_size = None
                    links = item.get("links", {})
                    if "attachment" in links:
                        attachment_info = links["attachment"]
                        if attachment_info.get("attachmentType") == "application/pdf":
                            attachment_size = attachment_info.get("attachmentSize")

                    # Skip attachments with zero or missing size
                    if attachment_size is None or attachment_size == 0:
                        logger.info(
                            f"Skipping attachment {attachment_key}: zero or missing attachment size"
                        )
                        continue

                    # Create unique identifier for duplicate detection
                    item_key = item_data.get("key", "")
                    unique_id = f"{item_key}_{attachment_size}"

                    downloaded_pdfs.append(
                        {
                            "pdf_path": pdf_path,
                            "attachment_key": attachment_key,
                            "parent_item": item,
                            "title": item_data.get("title", "Unknown"),
                            "authors": self._format_creators(
                                item_data.get("creators", [])
                            ),
                            "year": self._extract_year(item_data.get("date", "")),
                            "citation_key": self._extract_citation_key(item_data),
                            "attachment_size": attachment_size,
                            "unique_id": unique_id,
                            "item_key": item_key,
                        }
                    )

        logger.info(f"Downloaded {len(downloaded_pdfs)} PDFs")
        return downloaded_pdfs

    def find_local_pdfs(
        self,
        max_file_size_mb: float | None = None,
    ) -> List[Dict[str, Any]]:
        """Find all PDF attachments in local Zotero storage without downloading.

        Args:
            max_file_size_mb: Maximum file size in MB to include (default from env or 5MB)
        """
        import os

        found_pdfs = []
        zotero_storage = Path.home() / "Zotero" / "storage"

        if not zotero_storage.exists():
            logger.error(f"Zotero storage directory not found: {zotero_storage}")
            return []

        # Get file size limit from environment or use default
        if max_file_size_mb is None:
            max_file_size_mb = float(os.getenv("GROUNDERO_MAX_PDF_SIZE_MB", "5.0"))

        max_file_size_bytes = max_file_size_mb * 1024 * 1024

        # Get all items (no limit)
        items = self.get_all_items(limit=None)

        for item in items:
            item_data = item.get("data", {})
            item_key = item_data.get("key")

            if not item_key:
                continue

            # Get PDF attachments for this item
            pdf_attachments = self.get_pdf_attachments(item_key)
            for attachment in pdf_attachments:
                attachment_data = attachment.get("data", {})
                attachment_key = attachment_data.get("key")
                attachment_data.get("filename", "")

                if not attachment_key:
                    continue

                # Look for PDF in Zotero storage
                storage_path = zotero_storage / attachment_key
                if storage_path.exists():
                    # Find PDF file in the storage directory
                    pdf_files = list(storage_path.glob("*.pdf"))
                    if pdf_files:
                        pdf_path = pdf_files[0]  # Take the first PDF found

                        # Check file size
                        try:
                            file_size = pdf_path.stat().st_size
                            if file_size > max_file_size_bytes:
                                logger.info(
                                    f"Skipping {pdf_path.name}: {file_size / (1024 * 1024):.1f}MB > {max_file_size_mb}MB limit"
                                )
                                continue
                        except OSError:
                            logger.warning(f"Could not get file size for {pdf_path}")
                            continue

                        # Get attachment size from parent item links if available
                        attachment_size = None
                        links = item.get("links", {})
                        if "attachment" in links:
                            attachment_info = links["attachment"]
                            if (
                                attachment_info.get("attachmentType")
                                == "application/pdf"
                            ):
                                attachment_size = attachment_info.get("attachmentSize")

                        # Skip attachments with zero or missing size
                        if attachment_size is None or attachment_size == 0:
                            logger.info(
                                f"Skipping {pdf_path.name}: zero or missing attachment size"
                            )
                            continue

                        # Create unique identifier for duplicate detection
                        item_key = item_data.get("key", "")
                        unique_id = f"{item_key}_{attachment_size}"

                        found_pdfs.append(
                            {
                                "pdf_path": pdf_path,
                                "attachment_key": attachment_key,
                                "parent_item": item,
                                "title": item_data.get("title", "Unknown"),
                                "authors": self._format_creators(
                                    item_data.get("creators", [])
                                ),
                                "year": self._extract_year(item_data.get("date", "")),
                                "citation_key": self._extract_citation_key(item_data),
                                "attachment_size": attachment_size,
                                "unique_id": unique_id,
                                "item_key": item_key,
                            }
                        )

        logger.info(
            f"Found {len(found_pdfs)} PDFs in local Zotero storage (max size: {max_file_size_mb}MB)"
        )
        return found_pdfs

    def _format_creators(self, creators: List[Dict[str, Any]]) -> str:
        """Format creators/authors list."""
        if not creators:
            return "Unknown"

        author_names = []
        for creator in creators:
            if creator.get("creatorType") in ["author", "editor"]:
                if "name" in creator:
                    author_names.append(creator["name"])
                else:
                    first = creator.get("firstName", "")
                    last = creator.get("lastName", "")
                    if first and last:
                        author_names.append(f"{first} {last}")
                    elif last:
                        author_names.append(last)

        return ", ".join(author_names) if author_names else "Unknown"

    def _extract_year(self, date_str: str) -> str:
        """Extract year from date string."""
        if not date_str:
            return "Unknown"

        # Try to extract 4-digit year
        import re

        year_match = re.search(r"\b(19|20)\d{2}\b", date_str)
        return year_match.group() if year_match else "Unknown"

    def _extract_citation_key(self, item_data: Dict[str, Any]) -> str:
        """Extract citation key from Zotero's extra field, with fallback generation."""
        import re

        # First, try to extract from extra field
        extra = item_data.get("extra", "")
        if extra:
            # Look for "Citation Key: xyz" pattern
            match = re.search(r"Citation Key:\s*([^\n\r]+)", extra, re.IGNORECASE)
            if match:
                citation_key = match.group(1).strip()
                if citation_key:
                    return citation_key

        # Fallback: generate citation key if not found in Zotero
        return self._generate_fallback_citation_key(item_data)

    def _generate_fallback_citation_key(self, item_data: Dict[str, Any]) -> str:
        """Generate a fallback citation key when Zotero doesn't have one.

        Format: lastname_firstwordoftitle_year
        Example: smith_deep_2023
        """
        import re

        # Get first author's last name
        creators = item_data.get("creators", [])
        first_author = "unknown"
        if creators:
            for creator in creators:
                if creator.get("creatorType") == "author":
                    if "lastName" in creator:
                        first_author = creator["lastName"].lower()
                        break
                    elif "name" in creator:
                        # Handle single name field
                        name_parts = creator["name"].split()
                        if name_parts:
                            first_author = name_parts[
                                -1
                            ].lower()  # Take last part as surname
                        break

        # Get year
        year = self._extract_year(item_data.get("date", ""))
        if year == "Unknown":
            year = "nd"  # "no date" abbreviated

        # Get first word of title (cleaned)
        title = item_data.get("title", "untitled")
        # Clean title and get first meaningful word
        title_words = re.sub(r"[^\w\s]", "", title.lower()).split()

        # Skip common articles and get first meaningful word
        skip_words = {
            "a",
            "an",
            "the",
            "on",
            "in",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        first_word = "untitled"
        for word in title_words:
            if word not in skip_words and len(word) > 2:
                first_word = word
                break

        # Clean up names - remove spaces and special characters
        first_author = re.sub(r"[^\w]", "", first_author)
        first_word = re.sub(r"[^\w]", "", first_word)

        return f"{first_author}_{first_word}_{year}"


def test_zotero_connection():
    """Test Zotero connection and basic functionality."""
    try:
        client = ZoteroClient()
        items = client.get_all_items(limit=5)
        print(f"Successfully connected to Zotero. Found {len(items)} items.")

        for item in items[:3]:
            data = item.get("data", {})
            print(f"- {data.get('title', 'No title')}")

        return True
    except Exception as e:
        print(f"Error connecting to Zotero: {e}")
        return False


if __name__ == "__main__":
    # Test the connection
    test_zotero_connection()
