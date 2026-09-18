import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config import settings
from backend.models.schemas import DocumentChunk, DocumentInfo, ChunkType
from backend.database.embeddings import get_embedding_function

logger = logging.getLogger(__name__)

class ChromaDBManager:
    """Manages persistent ChromaDB vector storage and document registry."""

    COLLECTION_NAME = "technical_docs"

    def __init__(self):
        self.persist_dir = Path(settings.CHROMA_PERSIST_DIRECTORY)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.meta_file = self.persist_dir / "documents_meta.json"

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.embedding_fn = get_embedding_function()
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        self._init_meta()

    def _init_meta(self):
        if not self.meta_file.exists():
            self._save_meta({})

    def _load_meta(self) -> Dict[str, Any]:
        if not self.meta_file.exists():
            return {}
        try:
            return json.loads(self.meta_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_meta(self, meta: Dict[str, Any]):
        self.meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def add_chunks(self, doc_id: str, doc_info: DocumentInfo, chunks: List[DocumentChunk]):
        """Indexes chunks into ChromaDB and updates document metadata."""
        if not chunks:
            return

        # First, remove existing chunks if re-indexing
        self.delete_document(doc_id, update_meta=False)

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk.id)
            documents.append(chunk.content)
            # Flatten metadata ensuring values are primitives (str, int, float, bool)
            meta = {
                "doc_id": str(chunk.doc_id),
                "source": str(chunk.source),
                "page": int(chunk.page),
                "section": str(chunk.section or "General"),
                "parent_section": str(chunk.parent_section or ""),
                "chunk_type": str(chunk.chunk_type.value if hasattr(chunk.chunk_type, "value") else chunk.chunk_type),
                "language": str(chunk.language or ""),
                "code_size": int(chunk.code_size),
                "code_completeness_score": float(chunk.code_completeness_score),
                "has_imports": bool(chunk.has_imports),
                "has_functions": bool(chunk.has_functions),
                "has_initialization": bool(chunk.has_initialization),
                "implementation_id": str(chunk.implementation_id or ""),
                "chunk_index": int(chunk.chunk_index),
                "total_chunks": int(chunk.total_chunks),
                "chunk_sequence": int(chunk.metadata.get("chunk_sequence", 0)),
                "extracted_identifiers_str": str(chunk.metadata.get("extracted_identifiers_str", "")),
                "identifier_density": float(chunk.metadata.get("identifier_density", 0.0)),
            }
            metadatas.append(meta)

        # Batch insert to ChromaDB
        batch_size = 50
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end]
            )

        # Update metadata registry
        meta_dict = self._load_meta()
        meta_dict[doc_id] = doc_info.model_dump()
        self._save_meta(meta_dict)
        logger.info(f"Indexed {len(chunks)} chunks for document {doc_info.filename} ({doc_id})")

    def delete_document(self, doc_id: str, update_meta: bool = True):
        """Deletes all chunks of a document from ChromaDB and registry."""
        try:
            self.collection.delete(where={"doc_id": doc_id})
        except Exception as e:
            logger.warning(f"Error deleting chunks for doc_id {doc_id}: {e}")

        if update_meta:
            meta_dict = self._load_meta()
            if doc_id in meta_dict:
                del meta_dict[doc_id]
                self._save_meta(meta_dict)

    def list_documents(self) -> List[DocumentInfo]:
        """Lists all indexed documents."""
        meta_dict = self._load_meta()
        results = []
        for d in meta_dict.values():
            try:
                results.append(DocumentInfo(**d))
            except Exception:
                pass
        return results

    def get_document(self, doc_id: str) -> Optional[DocumentInfo]:
        meta_dict = self._load_meta()
        data = meta_dict.get(doc_id)
        return DocumentInfo(**data) if data else None

    def clear_all(self):
        """Clears all documents and vectors."""
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        self._save_meta({})

    def vector_search(self, query: str, top_k: int = 15, where_filter: Optional[dict] = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Executes semantic vector search.
        Returns list of (DocumentChunk, semantic_similarity_score [0.0 to 1.0]).
        """
        count = self.collection.count()
        if count == 0:
            return []

        query_k = min(top_k, count)
        kwargs = {
            "query_texts": [query],
            "n_results": query_k
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)

        items: List[Tuple[DocumentChunk, float]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(ids)

        for chunk_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
            # Chroma with cosine returns distance where distance = 1 - cosine_sim
            # Similarity = max(0.0, 1.0 - distance)
            sim = max(0.0, min(1.0, 1.0 - dist))

            chunk = DocumentChunk(
                id=chunk_id,
                doc_id=meta.get("doc_id", ""),
                source=meta.get("source", ""),
                page=int(meta.get("page", 1)),
                section=meta.get("section", "General"),
                parent_section=meta.get("parent_section", ""),
                chunk_type=ChunkType(meta.get("chunk_type", "prose")),
                language=meta.get("language") or None,
                code_size=int(meta.get("code_size", 0)),
                code_completeness_score=float(meta.get("code_completeness_score", 0.0)),
                has_imports=bool(meta.get("has_imports", False)),
                has_functions=bool(meta.get("has_functions", False)),
                has_initialization=bool(meta.get("has_initialization", False)),
                implementation_id=meta.get("implementation_id") or None,
                chunk_index=int(meta.get("chunk_index", 0)),
                total_chunks=int(meta.get("total_chunks", 1)),
                content=doc_text,
                metadata=meta
            )
            items.append((chunk, round(sim, 4)))

        return items

    def get_all_chunks(self) -> List[DocumentChunk]:
        """Retrieves all indexed chunks for keyword indexing/BM25."""
        count = self.collection.count()
        if count == 0:
            return []
        data = self.collection.get()
        if not data or not data["ids"]:
            return []

        chunks = []
        for chunk_id, doc_text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    doc_id=meta.get("doc_id", ""),
                    source=meta.get("source", ""),
                    page=int(meta.get("page", 1)),
                    section=meta.get("section", "General"),
                    parent_section=meta.get("parent_section", ""),
                    chunk_type=ChunkType(meta.get("chunk_type", "prose")),
                    language=meta.get("language") or None,
                    code_size=int(meta.get("code_size", 0)),
                    code_completeness_score=float(meta.get("code_completeness_score", 0.0)),
                    has_imports=bool(meta.get("has_imports", False)),
                    has_functions=bool(meta.get("has_functions", False)),
                    has_initialization=bool(meta.get("has_initialization", False)),
                    implementation_id=meta.get("implementation_id") or None,
                    chunk_index=int(meta.get("chunk_index", 0)),
                    total_chunks=int(meta.get("total_chunks", 1)),
                    content=doc_text,
                    metadata=meta
                )
            )
        return chunks

    def get_implementation_chunks(self, implementation_id: str) -> List[DocumentChunk]:
        """Retrieves and reconstructs all sibling chunks for a multi-part implementation."""
        if not implementation_id:
            return []
        data = self.collection.get(where={"implementation_id": implementation_id})
        if not data or not data["ids"]:
            return []

        chunks = []
        for chunk_id, doc_text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    doc_id=meta.get("doc_id", ""),
                    source=meta.get("source", ""),
                    page=int(meta.get("page", 1)),
                    section=meta.get("section", "General"),
                    parent_section=meta.get("parent_section", ""),
                    chunk_type=ChunkType(meta.get("chunk_type", "complete_code")),
                    language=meta.get("language") or None,
                    code_size=int(meta.get("code_size", 0)),
                    code_completeness_score=float(meta.get("code_completeness_score", 0.0)),
                    has_imports=bool(meta.get("has_imports", False)),
                    has_functions=bool(meta.get("has_functions", False)),
                    has_initialization=bool(meta.get("has_initialization", False)),
                    implementation_id=meta.get("implementation_id") or None,
                    chunk_index=int(meta.get("chunk_index", 0)),
                    total_chunks=int(meta.get("total_chunks", 1)),
                    content=doc_text,
                    metadata=meta
                )
            )

        # Sort chunks sequentially by chunk_index
        chunks.sort(key=lambda x: x.chunk_index)
        return chunks

    def get_section_sibling_chunks(
        self,
        doc_id: str,
        section: str,
        current_chunk_sequence: int,
        window: int = 1
    ) -> List[DocumentChunk]:
        """
        Retrieves the immediate preceding and succeeding sibling chunks
        from the same document and section to expand surrounding context.
        """
        if not doc_id:
            return []

        try:
            data = self.collection.get(where={"doc_id": doc_id})
            if not data or not data["ids"]:
                return []

            matched = []
            for chunk_id, doc_text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
                if meta.get("section") == section:
                    matched.append(
                        DocumentChunk(
                            id=chunk_id,
                            doc_id=meta.get("doc_id", ""),
                            source=meta.get("source", ""),
                            page=int(meta.get("page", 1)),
                            section=meta.get("section", "General"),
                            parent_section=meta.get("parent_section", ""),
                            chunk_type=ChunkType(meta.get("chunk_type", "prose")),
                            language=meta.get("language") or None,
                            code_size=int(meta.get("code_size", 0)),
                            code_completeness_score=float(meta.get("code_completeness_score", 0.0)),
                            has_imports=bool(meta.get("has_imports", False)),
                            has_functions=bool(meta.get("has_functions", False)),
                            has_initialization=bool(meta.get("has_initialization", False)),
                            implementation_id=meta.get("implementation_id") or None,
                            chunk_index=int(meta.get("chunk_index", 0)),
                            total_chunks=int(meta.get("total_chunks", 1)),
                            content=doc_text,
                            metadata=meta
                        )
                    )

            # Sort by chunk_sequence or page
            matched.sort(key=lambda c: int(c.metadata.get("chunk_sequence", c.page)))

            # Find position of current_chunk_sequence
            cur_idx = -1
            for i, c in enumerate(matched):
                if int(c.metadata.get("chunk_sequence", -1)) == current_chunk_sequence:
                    cur_idx = i
                    break

            if cur_idx == -1:
                return [c for c in matched if int(c.metadata.get("chunk_sequence", -1)) != current_chunk_sequence][:window * 2]

            siblings: List[DocumentChunk] = []
            # Preceding sibling
            if cur_idx - 1 >= 0:
                siblings.append(matched[cur_idx - 1])
            # Succeeding sibling
            if cur_idx + 1 < len(matched):
                siblings.append(matched[cur_idx + 1])

            return siblings
        except Exception as e:
            logger.warning(f"Error fetching sibling chunks: {e}")
            return []

db_manager = ChromaDBManager()
