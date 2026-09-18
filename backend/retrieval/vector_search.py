from typing import List, Tuple, Dict
from backend.models.schemas import DocumentChunk
from backend.database.chroma import db_manager
from backend.config import settings

class VectorSearch:
    """Executes multi-query vector searches in ChromaDB."""

    @classmethod
    def search(
        cls,
        queries: List[str],
        top_k: int = 15,
        filter_chunk_type: str = None
    ) -> List[Tuple[DocumentChunk, float]]:
        merged_results: Dict[str, Tuple[DocumentChunk, float]] = {}

        where_filter = {"chunk_type": filter_chunk_type} if filter_chunk_type else None

        for q in queries:
            results = db_manager.vector_search(
                query=q,
                top_k=top_k,
                where_filter=where_filter
            )
            for chunk, score in results:
                if chunk.id not in merged_results or score > merged_results[chunk.id][1]:
                    merged_results[chunk.id] = (chunk, score)

        sorted_items = sorted(merged_results.values(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]
