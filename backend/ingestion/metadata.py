import hashlib
from pathlib import Path
from typing import List, Dict, Any
from backend.models.schemas import DocumentChunk, DocumentInfo

class MetadataManager:
    """Manages document hashing, deduplication IDs, and chunk statistics."""

    @staticmethod
    def calculate_file_hash(file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def generate_doc_id(filename: str, file_hash: str) -> str:
        clean_name = "".join(c if c.isalnum() else "_" for c in Path(filename).stem)
        return f"{clean_name}_{file_hash[:8]}"

    @staticmethod
    def compute_chunk_statistics(chunks: List[DocumentChunk]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for chunk in chunks:
            c_type = chunk.chunk_type.value if hasattr(chunk.chunk_type, "value") else str(chunk.chunk_type)
            counts[c_type] = counts.get(c_type, 0) + 1
        return counts
