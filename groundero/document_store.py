#!/usr/bin/env python3
"""
SQLModel-based document store for Docling documents with Zotero metadata.
"""

from datetime import datetime
from typing import Optional, List
import json

from sqlmodel import SQLModel, Field, Session, select
from docling.datamodel.document import DoclingDocument



class DocumentRecord(SQLModel, table=True):
    """Database model for Docling documents with Zotero metadata."""

    __tablename__ = "documents"

    # Primary key and identifiers
    id: Optional[int] = Field(default=None, primary_key=True)
    binary_hash: str = Field(unique=True, index=True)
    citation_key: Optional[str] = Field(default=None, index=True)
    attachment_key: Optional[str] = Field(default=None, index=True)
    unique_id: Optional[str] = Field(default=None, index=True)

    # Document file information
    filename: Optional[str] = None
    pdf_path: Optional[str] = None
    docling_json_path: Optional[str] = None

    # Zotero bibliographic metadata
    title: Optional[str] = None
    authors: Optional[str] = None  # JSON string of author list
    year: Optional[int] = None

    # Document content (Docling JSON serialized)
    docling_content: str  # JSON string of the full DoclingDocument

    # Processing metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def get_docling_document(self) -> DoclingDocument:
        """Deserialize and return the DoclingDocument."""
        doc_data = json.loads(self.docling_content)
        return DoclingDocument.model_validate(doc_data)

    def get_authors_list(self) -> List[str]:
        """Get authors as a list."""
        if not self.authors:
            return []
        try:
            return json.loads(self.authors)
        except json.JSONDecodeError:
            return [self.authors]  # Fallback for single author strings


class DocumentStore:
    """SQLModel-based document store manager."""

    def __init__(self, engine):
        """Initialize the document store with a pre-configured engine."""
        self.engine = engine

    def add_document(
        self,
        docling_doc: DoclingDocument,
        pdf_path: Optional[str] = None,
        docling_json_path: Optional[str] = None,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        year: Optional[int] = None,
        citation_key: Optional[str] = None,
        attachment_key: Optional[str] = None,
        unique_id: Optional[str] = None,
    ) -> DocumentRecord:
        """Add a document to the store."""

        # Validate document origin
        if not docling_doc.origin or not docling_doc.origin.binary_hash:
            raise ValueError("DoclingDocument must have origin with binary_hash")

        # Serialize the DoclingDocument
        docling_json = docling_doc.model_dump_json()

        # Serialize authors list
        authors_json = json.dumps(authors) if authors else None

        # Create document record
        doc_record = DocumentRecord(
            binary_hash=str(docling_doc.origin.binary_hash),
            citation_key=citation_key,
            attachment_key=attachment_key,
            unique_id=unique_id,
            filename=docling_doc.origin.filename,
            pdf_path=pdf_path,
            docling_json_path=docling_json_path,
            title=title,
            authors=authors_json,
            year=year,
            docling_content=docling_json,
        )

        with Session(self.engine) as session:
            # Check if document already exists
            existing = session.exec(
                select(DocumentRecord).where(
                    DocumentRecord.binary_hash == str(docling_doc.origin.binary_hash)
                )
            ).first()

            if existing:
                # Update existing record
                existing.title = doc_record.title or existing.title
                existing.authors = doc_record.authors or existing.authors
                existing.year = doc_record.year or existing.year
                existing.citation_key = doc_record.citation_key or existing.citation_key
                existing.attachment_key = (
                    doc_record.attachment_key or existing.attachment_key
                )
                existing.unique_id = doc_record.unique_id or existing.unique_id
                existing.pdf_path = doc_record.pdf_path or existing.pdf_path
                existing.docling_json_path = (
                    doc_record.docling_json_path or existing.docling_json_path
                )
                existing.docling_content = doc_record.docling_content
                existing.updated_at = datetime.utcnow()

                session.add(existing)
                session.commit()
                session.refresh(existing)
                return existing
            else:
                # Add new record
                session.add(doc_record)
                session.commit()
                session.refresh(doc_record)
                return doc_record

    def get_document_by_citation_key(
        self, citation_key: str
    ) -> Optional[DocumentRecord]:
        """Get a document by its citation key."""
        with Session(self.engine) as session:
            return session.exec(
                select(DocumentRecord).where(
                    DocumentRecord.citation_key == citation_key
                )
            ).first()

    def get_document_by_binary_hash(self, binary_hash: str) -> Optional[DocumentRecord]:
        """Get a document by its binary hash."""
        with Session(self.engine) as session:
            return session.exec(
                select(DocumentRecord).where(DocumentRecord.binary_hash == binary_hash)
            ).first()

    def get_document_by_attachment_key(
        self, attachment_key: str
    ) -> Optional[DocumentRecord]:
        """Get a document by its Zotero attachment key."""
        with Session(self.engine) as session:
            return session.exec(
                select(DocumentRecord).where(
                    DocumentRecord.attachment_key == attachment_key
                )
            ).first()

    def get_all_documents(self) -> List[DocumentRecord]:
        """Get all documents."""
        with Session(self.engine) as session:
            return session.exec(select(DocumentRecord)).all()

    def get_documents_by_author(self, author_name: str) -> List[DocumentRecord]:
        """Get documents by author name (partial match)."""
        with Session(self.engine) as session:
            return session.exec(
                select(DocumentRecord).where(
                    DocumentRecord.authors.contains(author_name)
                )
            ).all()

    def get_documents_by_year(self, year: int) -> List[DocumentRecord]:
        """Get documents by publication year."""
        with Session(self.engine) as session:
            return session.exec(
                select(DocumentRecord).where(DocumentRecord.year == year)
            ).all()

    def search_documents(self, query: str) -> List[DocumentRecord]:
        """Search documents by title or content."""
        with Session(self.engine) as session:
            return session.exec(
                select(DocumentRecord).where(
                    DocumentRecord.title.contains(query)
                    | DocumentRecord.authors.contains(query)
                    | DocumentRecord.docling_content.contains(query)
                )
            ).all()

    def delete_document(self, binary_hash: str) -> bool:
        """Delete a document by binary hash."""
        with Session(self.engine) as session:
            doc = session.exec(
                select(DocumentRecord).where(DocumentRecord.binary_hash == binary_hash)
            ).first()

            if doc:
                session.delete(doc)
                session.commit()
                return True
            return False

    def count_documents(self) -> int:
        """Get total number of documents."""
        with Session(self.engine) as session:
            return len(session.exec(select(DocumentRecord)).all())

    def create_document_mapping(self) -> dict:
        """Create a mapping compatible with the old doc_store format."""
        mapping = {}
        for doc in self.get_all_documents():
            mapping[doc.binary_hash] = {
                "docling_path": doc.docling_json_path,
                "title": doc.title,
                "authors": doc.get_authors_list(),
                "year": doc.year,
                "citation_key": doc.citation_key,
                "attachment_key": doc.attachment_key,
                "pdf_path": doc.pdf_path,
                "filename": doc.filename,
            }
        return mapping
