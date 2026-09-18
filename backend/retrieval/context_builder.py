from typing import List, Tuple, Dict, Set, Optional
from backend.models.schemas import SearchResult, SourceCitation, ChunkType
from backend.database.chroma import db_manager
from backend.config import settings

class ContextBuilder:
    """
    Assembles retrieved chunks into structured, logically ordered context.
    Reconstructs complete implementations across parent-child chunks.
    """

    @classmethod
    def assemble_context(
        cls,
        top_candidates: List[SearchResult],
        min_threshold: float = settings.MIN_RELEVANCE_THRESHOLD,
        queried_identifiers: Optional[List[str]] = None
    ) -> Tuple[str, List[SourceCitation]]:
        # Filter candidates by relevance threshold
        valid_candidates = [c for c in top_candidates if c.final_score >= min_threshold]

        if not valid_candidates:
            return "The requested information is not available in the provided technical documentation.", []

        seen_chunks: Set[str] = set()
        seen_implementations: Set[str] = set()

        sections_prose: List[str] = []
        complete_codes: List[str] = []
        configurations: List[str] = []
        procedures_tables: List[str] = []
        supporting_snippets: List[str] = []

        citations: List[SourceCitation] = []
        seen_citation_keys: Set[str] = set()

        total_estimated_chars = 0
        max_chars = settings.MAX_CONTEXT_TOKENS * 4  # ~4 chars per token

        for candidate in valid_candidates:
            if candidate.chunk_id in seen_chunks:
                continue

            # Record citation
            cit_key = f"{candidate.source}_{candidate.page}_{candidate.section}_{candidate.chunk_type}"
            if cit_key not in seen_citation_keys:
                seen_citation_keys.add(cit_key)
                snippet_preview = candidate.content[:160].replace("\n", " ") + "..."
                citations.append(
                    SourceCitation(
                        source=candidate.source,
                        page=candidate.page,
                        section=candidate.section,
                        chunk_type=candidate.chunk_type,
                        snippet=snippet_preview
                    )
                )

            # Reconstruct multi-part complete implementation if applicable
            content_to_add = candidate.content
            if candidate.implementation_id and candidate.total_chunks > 1:
                if candidate.implementation_id in seen_implementations:
                    continue
                seen_implementations.add(candidate.implementation_id)
                # Fetch all sibling chunks
                sibling_chunks = db_manager.get_implementation_chunks(candidate.implementation_id)
                if sibling_chunks:
                    reconstructed_lines = [s.content for s in sibling_chunks]
                    content_to_add = "\n\n// --- [Reconstructed Implementation Block] ---\n".join(reconstructed_lines)
                    for s in sibling_chunks:
                        seen_chunks.add(s.id)

            # Surrounding Context Window Expansion:
            # If candidate matches an exact queried identifier, pull immediate section siblings
            has_id_match = bool(
                queried_identifiers and any(
                    ident.lower() in candidate.content.lower() or
                    ident.lower() in candidate.section.lower()
                    for ident in queried_identifiers
                )
            )
            if has_id_match and candidate.doc_id:
                c_seq = getattr(candidate, "chunk_sequence", candidate.chunk_index)
                siblings = db_manager.get_section_sibling_chunks(
                    doc_id=candidate.doc_id,
                    section=candidate.section,
                    current_chunk_sequence=c_seq,
                    window=1
                )
                for sib in siblings:
                    if sib.id not in seen_chunks:
                        seen_chunks.add(sib.id)
                        content_to_add += (
                            f"\n\n// --- [Surrounding Context: Section: {sib.section} | Page: {sib.page}] ---\n"
                            f"{sib.content}"
                        )

            seen_chunks.add(candidate.chunk_id)

            # Header metadata for LLM grounding
            block_header = (
                f"### [DOCUMENT: {candidate.source} | PAGE: {candidate.page} | "
                f"SECTION: {candidate.section} | TYPE: {candidate.chunk_type.upper()}]\n"
            )
            formatted_block = f"{block_header}\n{content_to_add}\n"

            # Check character budget
            if total_estimated_chars + len(formatted_block) > max_chars and len(citations) >= 2:
                continue

            total_estimated_chars += len(formatted_block)

            # Categorize into ordered assembly buckets
            c_type = candidate.chunk_type
            if c_type in ["complete_code"]:
                complete_codes.append(formatted_block)
            elif c_type in ["prose", "mixed"]:
                sections_prose.append(formatted_block)
            elif c_type in ["configuration"]:
                configurations.append(formatted_block)
            elif c_type in ["procedure", "table"]:
                procedures_tables.append(formatted_block)
            else:
                supporting_snippets.append(formatted_block)

        # Logical assembly order:
        # 1. Prose / Explanations
        # 2. Complete Implementations
        # 3. Configurations
        # 4. Procedures / Tables
        # 5. Supporting snippets
        context_parts = []

        if sections_prose:
            context_parts.append("## OVERVIEW & EXPLANATIONS\n" + "\n---\n".join(sections_prose))
        if complete_codes:
            context_parts.append("## COMPLETE IMPLEMENTATIONS\n" + "\n---\n".join(complete_codes))
        if configurations:
            context_parts.append("## CONFIGURATION & SETTINGS\n" + "\n---\n".join(configurations))
        if procedures_tables:
            context_parts.append("## PROCEDURES & SPECIFICATIONS\n" + "\n---\n".join(procedures_tables))
        if supporting_snippets:
            context_parts.append("## SUPPORTING SNIPPETS\n" + "\n---\n".join(supporting_snippets))

        final_context = "\n\n========================================\n\n".join(context_parts)
        return final_context, citations
