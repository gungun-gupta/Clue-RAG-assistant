import re
from typing import List, Dict, Tuple, Optional
from backend.models.schemas import DocumentChunk, ChunkType, QueryClass, SearchResult
from backend.config import settings

class CodeAwareReranker:
    """
    Reranks candidate chunks by combining semantic vector scores, BM25 keyword scores,
    section relevance, and code completeness scores with query-specific boosts.
    """

    @classmethod
    def rerank(
        cls,
        chunks_with_semantic: List[Tuple[DocumentChunk, float]],
        keyword_scores: Dict[str, float],
        query: str,
        query_class: QueryClass,
        identifiers: List[str],
        min_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        candidates: List[SearchResult] = []
        query_lower = query.lower()

        w_sem = settings.WEIGHT_SEMANTIC
        w_kw = settings.WEIGHT_KEYWORD
        w_comp = settings.WEIGHT_COMPLETENESS
        w_sec = settings.WEIGHT_SECTION

        for chunk, sem_score in chunks_with_semantic:
            kw_score = keyword_scores.get(chunk.id, 0.0)

            # Section matching score
            section_full = f"{chunk.section} {chunk.parent_section}".lower()
            sec_score = 0.0
            if any(term in section_full for term in query_lower.split() if len(term) > 3):
                sec_score = 0.7
            if identifiers and any(ident.lower() in section_full for ident in identifiers):
                sec_score = 1.0

            # Exact Keyword / Identifier Deterministic Check
            chunk_text_lower = chunk.content.lower()
            meta_ids = [i.lower() for i in chunk.metadata.get("extracted_identifiers", [])]
            exact_id_matches = sum(
                1 for ident in identifiers
                if ident.lower() in chunk_text_lower or ident.lower() in section_full or ident.lower() in meta_ids
            ) if identifiers else 0

            # Overview, Architecture, Definition section boost for conceptual queries (only when query topic matches)
            if query_class == QueryClass.EXPLANATION and (exact_id_matches > 0 or kw_score > 0.0 or sem_score >= 0.50):
                if any(k in section_full for k in ["overview", "architecture", "concept", "introduction", "about", "definition", "purpose", "scope"]):
                    sec_score = max(sec_score, 0.9)

            # Code boost and completeness dynamics based on Query Class
            comp_score = chunk.code_completeness_score
            code_boost = 0.0

            if query_class == QueryClass.COMPLETE_CODE:
                if chunk.chunk_type == ChunkType.COMPLETE_CODE:
                    code_boost = settings.COMPLETE_CODE_BOOST  # +0.40
                    # Additional boost for rich complete code signals
                    if chunk.has_functions:
                        code_boost += 0.08
                    if chunk.has_initialization:
                        code_boost += 0.08
                    if chunk.has_imports:
                        code_boost += 0.05
                elif chunk.chunk_type == ChunkType.CONFIGURATION:
                    code_boost = 0.15
                elif chunk.chunk_type == ChunkType.MIXED:
                    code_boost = 0.05
                elif chunk.chunk_type == ChunkType.CODE_SNIPPET:
                    # HEAVILY PENALIZE tiny isolated snippets when user asks for COMPLETE code
                    code_boost = -0.30
                elif chunk.chunk_type == ChunkType.PROSE:
                    code_boost = -0.15

            elif query_class == QueryClass.CODE:
                if chunk.chunk_type == ChunkType.COMPLETE_CODE:
                    code_boost = 0.25
                elif chunk.chunk_type == ChunkType.CODE_SNIPPET:
                    code_boost = 0.20
                elif chunk.chunk_type == ChunkType.CONFIGURATION:
                    code_boost = 0.15
                elif chunk.chunk_type == ChunkType.MIXED:
                    code_boost = 0.10
                elif chunk.chunk_type == ChunkType.PROSE:
                    code_boost = -0.05

            elif query_class == QueryClass.CONFIGURATION:
                if chunk.chunk_type == ChunkType.CONFIGURATION:
                    code_boost = 0.35
                elif chunk.chunk_type == ChunkType.TABLE:
                    code_boost = 0.25
                elif chunk.chunk_type in [ChunkType.COMPLETE_CODE, ChunkType.CODE_SNIPPET]:
                    code_boost = 0.10

            elif query_class == QueryClass.EXPLANATION:
                if chunk.chunk_type == ChunkType.PROSE:
                    code_boost = 0.35
                elif chunk.chunk_type == ChunkType.MIXED:
                    code_boost = 0.20
                elif chunk.chunk_type in [ChunkType.CODE_SNIPPET, ChunkType.COMPLETE_CODE]:
                    # De-prioritize raw code without prose for pure conceptual questions
                    code_boost = -0.25

                # Extra boost if chunk explicitly defines or introduces the queried identifier
                if identifiers and (exact_id_matches > 0 or kw_score > 0.0 or sem_score >= 0.50):
                    for ident in identifiers:
                        il = ident.lower()
                        if any(p in chunk_text_lower for p in [
                            f"called {il}", f"called {il}s",
                            f"{il} is", f"{il}s are",
                            f"{il} represents", f"{il}s represent",
                            f"what is a {il}", f"what is an {il}",
                            f"definition of {il}", f"overview of {il}"
                        ]):
                            code_boost += 0.30
                            break

            # Determine topical relevance to query:
            # A chunk is topically relevant if it matches exact identifiers, keywords,
            # section headers (with matching query content), or exhibits high semantic similarity (>= 0.60).
            is_topically_relevant = (exact_id_matches > 0) or (kw_score > 0.0) or (sem_score >= 0.60)

            if not is_topically_relevant:
                # Out-of-scope / unrelated document:
                # Suppress code boost and completeness score so that irrelevant documents cannot
                # cross the minimum relevance threshold.
                raw_score = w_sem * sem_score * 0.5
            else:
                # Topically relevant document: compute full weighted score with completeness,
                # type boosts, and deterministic exact identifier boosts
                exact_boost = 0.25 * min(exact_id_matches, 2)
                raw_score = (
                    (w_sem * sem_score) +
                    (w_kw * kw_score) +
                    (w_comp * comp_score) +
                    (w_sec * sec_score) +
                    code_boost +
                    exact_boost
                )

            final_score = round(max(0.01, min(1.0, raw_score)), 4)

            candidates.append(
                (
                    SearchResult(
                        chunk_id=chunk.id,
                        doc_id=chunk.doc_id,
                        source=chunk.source,
                        page=chunk.page,
                        section=chunk.section,
                        parent_section=chunk.parent_section,
                        chunk_type=chunk.chunk_type.value if hasattr(chunk.chunk_type, "value") else str(chunk.chunk_type),
                        language=chunk.language,
                        content=chunk.content,
                        code_completeness_score=round(chunk.code_completeness_score, 2),
                        semantic_score=round(sem_score, 4),
                        keyword_score=round(kw_score, 4),
                        completeness_score=round(comp_score, 4),
                        section_score=round(sec_score, 4),
                        final_score=final_score,
                        implementation_id=chunk.implementation_id,
                        chunk_index=chunk.chunk_index,
                        total_chunks=chunk.total_chunks,
                        chunk_sequence=int(chunk.metadata.get("chunk_sequence", chunk.chunk_index))
                    ),
                    raw_score
                )
            )

        # Sort descending by raw_score first (breaks ties when final_score is clamped to 1.0),
        # then by semantic similarity and keyword score
        candidates.sort(key=lambda item: (item[1], item[0].semantic_score, item[0].keyword_score), reverse=True)
        results = [item[0] for item in candidates]

        if min_threshold is not None:
            results = [c for c in results if c.final_score >= min_threshold]

        return results

    @classmethod
    def filter_by_relevance(
        cls,
        results: List[SearchResult],
        min_threshold: float = settings.MIN_RELEVANCE_THRESHOLD
    ) -> List[SearchResult]:
        """Filters candidates to only those meeting or exceeding the minimum relevance threshold."""
        return [r for r in results if r.final_score >= min_threshold]

    @classmethod
    def has_sufficient_relevance(
        cls,
        results: List[SearchResult],
        min_threshold: float = settings.MIN_RELEVANCE_THRESHOLD
    ) -> bool:
        """Checks if at least one candidate meets or exceeds the minimum relevance threshold."""
        return any(r.final_score >= min_threshold for r in results)

