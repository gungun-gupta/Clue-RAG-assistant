import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks

from backend.config import settings
from backend.models.schemas import DocumentInfo
from backend.ingestion.parser import DocumentParser
from backend.ingestion.chunker import CodeAwareChunker
from backend.ingestion.metadata import MetadataManager
from backend.database.chroma import db_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)):
    """Uploads and indexes a technical document or code file."""
    if not DocumentParser.is_supported(file.filename):
        supported = ", ".join(sorted(DocumentParser.SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{Path(file.filename).suffix}'. Supported formats: {supported}"
        )

    # Save uploaded file to disk
    docs_dir = Path(settings.DOCUMENTS_DIRECTORY)
    docs_dir.mkdir(parents=True, exist_ok=True)
    saved_path = docs_dir / file.filename

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Calculate hash and doc_id
        file_hash = MetadataManager.calculate_file_hash(saved_path)
        doc_id = MetadataManager.generate_doc_id(file.filename, file_hash)

        # Parse document
        parsed_doc = DocumentParser.parse_file(str(saved_path))

        # Code-aware chunking
        chunks = CodeAwareChunker.chunk_document(parsed_doc, doc_id=doc_id)
        chunk_stats = MetadataManager.compute_chunk_statistics(chunks)

        doc_info = DocumentInfo(
            doc_id=doc_id,
            filename=file.filename,
            file_type=parsed_doc.file_type,
            file_size=parsed_doc.file_size,
            page_count=parsed_doc.page_count,
            chunk_count=len(chunks),
            chunk_type_counts=chunk_stats,
            upload_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            file_hash=file_hash
        )

        # Index into ChromaDB
        db_manager.add_chunks(doc_id=doc_id, doc_info=doc_info, chunks=chunks)
        return doc_info

    except Exception as e:
        logger.error(f"Failed to process and index document {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")

@router.get("", response_model=List[DocumentInfo])
async def list_documents():
    """Lists all indexed technical documents."""
    return db_manager.list_documents()

@router.get("/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: str):
    """Retrieves metadata for a specific indexed document."""
    doc = db_manager.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@router.post("/{doc_id}/index", response_model=DocumentInfo)
async def reindex_document(doc_id: str):
    """Re-indexes an existing document from its original file."""
    doc = db_manager.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = Path(settings.DOCUMENTS_DIRECTORY) / doc.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Source file missing from disk")

    try:
        parsed_doc = DocumentParser.parse_file(str(file_path))
        chunks = CodeAwareChunker.chunk_document(parsed_doc, doc_id=doc_id)
        chunk_stats = MetadataManager.compute_chunk_statistics(chunks)

        doc.chunk_count = len(chunks)
        doc.chunk_type_counts = chunk_stats
        doc.upload_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        db_manager.add_chunks(doc_id=doc_id, doc_info=doc, chunks=chunks)
        return doc
    except Exception as e:
        logger.error(f"Failed to re-index {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Re-indexing failed: {str(e)}")

@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """Deletes a document and its embeddings from the vector database."""
    doc = db_manager.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    db_manager.delete_document(doc_id)
    # Remove file from storage if present
    file_path = Path(settings.DOCUMENTS_DIRECTORY) / doc.filename
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception:
            pass

    return {"message": f"Document '{doc.filename}' ({doc_id}) successfully deleted"}

@router.delete("")
async def clear_all_documents():
    """Clears all documents and resets the vector database."""
    db_manager.clear_all()
    # Clean uploaded files
    docs_dir = Path(settings.DOCUMENTS_DIRECTORY)
    if docs_dir.exists():
        for item in docs_dir.iterdir():
            if item.is_file():
                try:
                    item.unlink()
                except Exception:
                    pass
    return {"message": "All documents and vectors cleared successfully"}
