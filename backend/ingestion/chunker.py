import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from backend.models.schemas import DocumentChunk, ChunkType
from backend.ingestion.parser import ParsedDocument, ParsedElement
from backend.ingestion.structure import StructureDetector
from backend.ingestion.code_detector import CodeDetector
from backend.config import settings

class CodeAwareChunker:
    """
    Code-aware, structurally-grounded chunking pipeline.
    Preserves hierarchical section boundaries, unifies explanations with code blocks/tables,
    maintains atomic code block integrity, and tracks extracted identifier density.
    """

    MAX_COMPOSITE_CHUNK_CHARS = 2400
    MAX_CODE_CHUNK_CHARS = 1800
    MAX_PROSE_CHUNK_CHARS = 1200
    PROSE_OVERLAP_CHARS = 150

    @classmethod
    def extract_chunk_identifiers(cls, content: str) -> List[str]:
        """Extracts technical identifiers: CamelCase, snake_case, functions, APIs, classes, variables."""
        identifiers: List[str] = []

        # 1. CamelCase / PascalCase
        camel = re.findall(r"\b[A-Za-z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b", content)
        identifiers.extend(camel)

        # 2. snake_case
        snake = re.findall(r"\b[a-z0-9]+_[a-z0-9_]+\b", content)
        identifiers.extend(snake)

        # 3. Variable, function, and class declarations
        decls = re.findall(r"\b(?:const|let|var|function|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", content)
        identifiers.extend(decls)

        # 4. API endpoints: /api/...
        apis = re.findall(r"(/api/[a-zA-Z0-9_\-\./]+)", content)
        identifiers.extend(apis)

        # 5. Component / HTML tags
        tags = re.findall(r"<([a-zA-Z0-9_\-]+)>", content)
        identifiers.extend(tags)

        # 6. Capitalized technical identifiers (len >= 4)
        caps = re.findall(r"\b[A-Z][a-zA-Z0-9_]{3,}\b", content)
        identifiers.extend(caps)

        # Stop words to exclude
        stop_words = {
            "this", "that", "with", "from", "have", "more", "some", "what",
            "when", "where", "which", "will", "true", "false", "null", "none",
            "undefined", "return", "function", "const", "class", "async", "await",
            "export", "import", "type", "interface", "public", "private", "then"
        }
        filtered = [ident for ident in identifiers if ident.lower() not in stop_words]
        return list(dict.fromkeys(filtered))

    @classmethod
    def chunk_document(cls, parsed_doc: ParsedDocument, doc_id: str) -> List[DocumentChunk]:
        """
        Chunks a parsed document adhering strictly to structural heading boundaries.
        Unifies heading, explanatory prose, code blocks, and parameter tables in composite chunks.
        """
        chunks: List[DocumentChunk] = []

        # 1. Group consecutive elements by section and page
        section_groups: List[List[ParsedElement]] = []
        current_group: List[ParsedElement] = []
        current_section = "General"
        parent_section = ""

        for element in parsed_doc.elements:
            if element.element_type == "heading":
                if current_group:
                    section_groups.append(current_group)
                    current_group = []
                parent_section = current_section
                current_section = element.content
                # Include heading in the new group to anchor context
                current_group.append(element)
                continue

            # Check if section changed
            elem_sec = element.section or current_section
            elem_psec = element.parent_section or parent_section

            if current_group:
                prev_elem = current_group[-1]
                prev_sec = prev_elem.section or current_section
                if elem_sec != prev_sec:
                    section_groups.append(current_group)
                    current_group = []

            element.section = elem_sec
            element.parent_section = elem_psec
            current_group.append(element)

        if current_group:
            section_groups.append(current_group)

        # 2. Process each section group
        chunk_sequence = 0

        for group in section_groups:
            if not group:
                continue

            sec = group[0].section or current_section
            p_sec = group[0].parent_section or parent_section

            # Filter out standalone heading element if it's the only one
            non_heading_elements = [e for e in group if e.element_type != "heading"]
            # Anchor page number to the primary body content (or heading if alone)
            page = non_heading_elements[0].page_number if non_heading_elements else group[0].page_number
            heading_text = ""
            for e in group:
                if e.element_type == "heading":
                    heading_text = f"## {e.content}\n\n"
                    break

            if not non_heading_elements:
                if heading_text:
                    # Single heading alone without body
                    chunk = cls._create_single_chunk(
                        content=heading_text.strip(),
                        chunk_type=ChunkType.PROSE,
                        page=page,
                        section=sec,
                        parent_section=p_sec,
                        doc_id=doc_id,
                        source=parsed_doc.filename,
                        chunk_sequence=chunk_sequence
                    )
                    chunks.append(chunk)
                    chunk_sequence += 1
                continue

            # Check if elements in this section can be unified into a single composite chunk
            combined_body = "\n\n".join(e.content.strip() for e in non_heading_elements if e.content.strip())
            full_composite_text = (heading_text + combined_body).strip()

            has_code = any(
                e.element_type == "code_block" or CodeDetector.is_code_block(e.content)
                for e in non_heading_elements
            )
            has_table = any(
                e.element_type == "table" or StructureDetector.is_table(e.content)
                for e in non_heading_elements
            )
            has_config = any(
                StructureDetector.is_configuration(e.content)
                for e in non_heading_elements
            )

            # Composite unification: if entire section content fits within budget, keep unified
            if len(full_composite_text) <= cls.MAX_COMPOSITE_CHUNK_CHARS:
                # Determine unified chunk type and completeness
                completeness_score, signals = (0.0, {})
                code_size = 0
                lang = None

                if has_code:
                    code_texts = [e.content for e in non_heading_elements if e.element_type == "code_block" or CodeDetector.is_code_block(e.content)]
                    joined_code = "\n".join(code_texts)
                    lang = CodeDetector.detect_language(joined_code)
                    completeness_score, signals = CodeDetector.analyze_completeness(joined_code, sec, p_sec)
                    code_size = len(joined_code.splitlines())

                chunk_type = CodeDetector.classify_chunk_type(
                    text=full_composite_text,
                    is_code=has_code,
                    completeness_score=completeness_score,
                    is_config=has_config
                )

                chunk = DocumentChunk(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    source=parsed_doc.filename,
                    page=page,
                    section=sec,
                    parent_section=p_sec,
                    chunk_type=chunk_type,
                    language=lang,
                    code_size=code_size,
                    code_completeness_score=completeness_score,
                    has_imports=signals.get("has_imports", False),
                    has_functions=signals.get("has_functions", False),
                    has_initialization=signals.get("has_initialization", False),
                    implementation_id=str(uuid.uuid4()) if chunk_type == ChunkType.COMPLETE_CODE else None,
                    chunk_index=0,
                    total_chunks=1,
                    content=full_composite_text,
                    metadata=cls._build_metadata(
                        content=full_composite_text,
                        signals=signals,
                        source=parsed_doc.filename,
                        page=page,
                        section=sec,
                        chunk_sequence=chunk_sequence
                    )
                )
                chunks.append(chunk)
                chunk_sequence += 1
            else:
                # Section is too large: process elements while preserving atomic code block boundaries
                group_chunks = cls._process_large_section_group(
                    elements=non_heading_elements,
                    heading_text=heading_text,
                    page=page,
                    section=sec,
                    parent_section=p_sec,
                    doc_id=doc_id,
                    source=parsed_doc.filename,
                    start_sequence=chunk_sequence
                )
                chunks.extend(group_chunks)
                chunk_sequence += len(group_chunks)

        return chunks

    @classmethod
    def _process_large_section_group(
        cls,
        elements: List[ParsedElement],
        heading_text: str,
        page: int,
        section: str,
        parent_section: str,
        doc_id: str,
        source: str,
        start_sequence: int
    ) -> List[DocumentChunk]:
        """Handles a section group that exceeds MAX_COMPOSITE_CHUNK_CHARS without tearing code blocks."""
        chunks: List[DocumentChunk] = []
        seq = start_sequence

        for idx, element in enumerate(elements):
            elem_page = element.page_number or page
            is_code = element.element_type == "code_block" or CodeDetector.is_code_block(element.content)
            is_table = element.element_type == "table" or StructureDetector.is_table(element.content)
            is_proc = element.element_type == "procedure" or StructureDetector.is_procedure(element.content)
            is_config = StructureDetector.is_configuration(element.content)

            # Prefix the first element with the section heading to prevent orphan context
            prefix = heading_text if idx == 0 and heading_text else ""
            content = (prefix + element.content).strip()

            if is_code:
                code_chunks = cls._process_code_element(
                    content=content,
                    page=elem_page,
                    section=section,
                    parent_section=parent_section,
                    doc_id=doc_id,
                    source=source,
                    start_sequence=seq,
                    specified_lang=element.metadata.get("language")
                )
                chunks.extend(code_chunks)
                seq += len(code_chunks)
            elif is_table:
                chunk = cls._create_single_chunk(
                    content=content,
                    chunk_type=ChunkType.TABLE,
                    page=elem_page,
                    section=section,
                    parent_section=parent_section,
                    doc_id=doc_id,
                    source=source,
                    chunk_sequence=seq
                )
                chunks.append(chunk)
                seq += 1
            elif is_proc:
                chunk = cls._create_single_chunk(
                    content=content,
                    chunk_type=ChunkType.PROCEDURE,
                    page=elem_page,
                    section=section,
                    parent_section=parent_section,
                    doc_id=doc_id,
                    source=source,
                    chunk_sequence=seq
                )
                chunks.append(chunk)
                seq += 1
            elif is_config:
                chunk = cls._create_single_chunk(
                    content=content,
                    chunk_type=ChunkType.CONFIGURATION,
                    page=elem_page,
                    section=section,
                    parent_section=parent_section,
                    doc_id=doc_id,
                    source=source,
                    chunk_sequence=seq
                )
                chunks.append(chunk)
                seq += 1
            else:
                prose_chunks = cls._process_prose_element(
                    content=content,
                    page=elem_page,
                    section=section,
                    parent_section=parent_section,
                    doc_id=doc_id,
                    source=source,
                    start_sequence=seq
                )
                chunks.extend(prose_chunks)
                seq += len(prose_chunks)

        return chunks

    @classmethod
    def _process_code_element(
        cls,
        content: str,
        page: int,
        section: str,
        parent_section: str,
        doc_id: str,
        source: str,
        start_sequence: int = 0,
        specified_lang: Optional[str] = None
    ) -> List[DocumentChunk]:
        lang = specified_lang or CodeDetector.detect_language(content)
        completeness_score, signals = CodeDetector.analyze_completeness(content, section, parent_section)
        chunk_type = CodeDetector.classify_chunk_type(
            text=content,
            is_code=True,
            completeness_score=completeness_score,
            is_config=StructureDetector.is_configuration(content)
        )
        code_size = len(content.splitlines())

        # Atomic integrity: if code fits within max code limit, do not slice it
        if len(content) <= cls.MAX_CODE_CHUNK_CHARS:
            impl_id = str(uuid.uuid4()) if chunk_type == ChunkType.COMPLETE_CODE else None
            return [
                DocumentChunk(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    source=source,
                    page=page,
                    section=section,
                    parent_section=parent_section,
                    chunk_type=chunk_type,
                    language=lang,
                    code_size=code_size,
                    code_completeness_score=completeness_score,
                    has_imports=signals.get("has_imports", False),
                    has_functions=signals.get("has_functions", False),
                    has_initialization=signals.get("has_initialization", False),
                    implementation_id=impl_id,
                    chunk_index=0,
                    total_chunks=1,
                    content=content,
                    metadata=cls._build_metadata(
                        content=content,
                        signals=signals,
                        source=source,
                        page=page,
                        section=section,
                        chunk_sequence=start_sequence
                    )
                )
            ]

        # Very large code: split along outer function/declaration boundaries with overlapping signature preservation
        return cls._split_large_code(
            content=content,
            page=page,
            section=section,
            parent_section=parent_section,
            doc_id=doc_id,
            source=source,
            lang=lang,
            base_completeness=completeness_score,
            signals=signals,
            start_sequence=start_sequence
        )

    @classmethod
    def _split_large_code(
        cls,
        content: str,
        page: int,
        section: str,
        parent_section: str,
        doc_id: str,
        source: str,
        lang: Optional[str],
        base_completeness: float,
        signals: Dict[str, Any],
        start_sequence: int = 0
    ) -> List[DocumentChunk]:
        """Splits oversized code blocks while strictly preserving function signatures and class definitions."""
        lines = content.splitlines()
        chunks: List[str] = []
        implementation_id = str(uuid.uuid4())

        current_lines: List[str] = []
        current_len = 0
        last_signature = ""

        for line in lines:
            stripped = line.strip()
            line_len = len(line) + 1

            # Track surrounding signature (function, class, export)
            if stripped.startswith(("function ", "def ", "class ", "export function ", "export class ", "const ", "let ")):
                if "(" in stripped or "{" in stripped or ":" in stripped:
                    last_signature = line

            # Only break at outer declaration boundaries
            is_boundary = (
                stripped.startswith(("function ", "def ", "class ", "export function ", "export class ")) or
                (stripped.startswith(("const ", "let ", "var ")) and not line.startswith(" "))
            ) and current_len > 500

            if (current_len + line_len > cls.MAX_CODE_CHUNK_CHARS or is_boundary) and current_lines:
                chunk_text = "\n".join(current_lines)
                chunks.append(chunk_text)
                # Overlapping signature preservation
                current_lines = [f"// [Context continued: {last_signature}]"] if last_signature and not is_boundary else []
                current_lines.append(line)
                current_len = sum(len(l) + 1 for l in current_lines)
            else:
                current_lines.append(line)
                current_len += line_len

        if current_lines:
            chunks.append("\n".join(current_lines))

        total_chunks = len(chunks)
        result_chunks = []

        for idx, chunk_text in enumerate(chunks):
            sub_score, sub_signals = CodeDetector.analyze_completeness(chunk_text, section, parent_section)
            final_completeness = max(sub_score, round(base_completeness * 0.9, 2))

            result_chunks.append(
                DocumentChunk(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    source=source,
                    page=page,
                    section=section,
                    parent_section=parent_section,
                    chunk_type=ChunkType.COMPLETE_CODE,
                    language=lang,
                    code_size=len(chunk_text.splitlines()),
                    code_completeness_score=final_completeness,
                    has_imports=sub_signals.get("has_imports", signals.get("has_imports", False)),
                    has_functions=sub_signals.get("has_functions", False),
                    has_initialization=sub_signals.get("has_initialization", False),
                    implementation_id=implementation_id,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    content=chunk_text,
                    metadata=cls._build_metadata(
                        content=chunk_text,
                        signals={**sub_signals, "is_part_of_complete": True},
                        source=source,
                        page=page,
                        section=section,
                        chunk_sequence=start_sequence + idx
                    )
                )
            )

        return result_chunks

    @classmethod
    def _process_prose_element(
        cls,
        content: str,
        page: int,
        section: str,
        parent_section: str,
        doc_id: str,
        source: str,
        start_sequence: int = 0
    ) -> List[DocumentChunk]:
        if len(content) <= cls.MAX_PROSE_CHUNK_CHARS:
            chunk_type = ChunkType.MIXED if "`" in content else ChunkType.PROSE
            return [
                cls._create_single_chunk(
                    content=content,
                    chunk_type=chunk_type,
                    page=page,
                    section=section,
                    parent_section=parent_section,
                    doc_id=doc_id,
                    source=source,
                    chunk_sequence=start_sequence
                )
            ]

        # Break long prose with paragraph or sentence boundaries
        chunks = []
        words = content.split(" ")
        current_words: List[str] = []
        current_len = 0
        seq = start_sequence

        for word in words:
            current_words.append(word)
            current_len += len(word) + 1
            if current_len >= cls.MAX_PROSE_CHUNK_CHARS:
                text_slice = " ".join(current_words)
                chunk_type = ChunkType.MIXED if "`" in text_slice else ChunkType.PROSE
                chunks.append(
                    cls._create_single_chunk(
                        content=text_slice,
                        chunk_type=chunk_type,
                        page=page,
                        section=section,
                        parent_section=parent_section,
                        doc_id=doc_id,
                        source=source,
                        chunk_sequence=seq
                    )
                )
                seq += 1
                overlap_words = current_words[-15:]
                current_words = list(overlap_words)
                current_len = sum(len(w) + 1 for w in current_words)

        if current_words:
            text_slice = " ".join(current_words)
            chunks.append(
                cls._create_single_chunk(
                    content=text_slice,
                    chunk_type=ChunkType.MIXED if "`" in text_slice else ChunkType.PROSE,
                    page=page,
                    section=section,
                    parent_section=parent_section,
                    doc_id=doc_id,
                    source=source,
                    chunk_sequence=seq
                )
            )

        return chunks

    @classmethod
    def _create_single_chunk(
        cls,
        content: str,
        chunk_type: ChunkType,
        page: int,
        section: str,
        parent_section: str,
        doc_id: str,
        source: str,
        chunk_sequence: int = 0
    ) -> DocumentChunk:
        return DocumentChunk(
            id=str(uuid.uuid4()),
            doc_id=doc_id,
            source=source,
            page=page,
            section=section,
            parent_section=parent_section,
            chunk_type=chunk_type,
            language=None,
            code_size=0,
            code_completeness_score=0.0,
            has_imports=False,
            has_functions=False,
            has_initialization=False,
            implementation_id=None,
            chunk_index=0,
            total_chunks=1,
            content=content,
            metadata=cls._build_metadata(
                content=content,
                signals={},
                source=source,
                page=page,
                section=section,
                chunk_sequence=chunk_sequence
            )
        )

    @classmethod
    def _build_metadata(
        cls,
        content: str,
        signals: Dict[str, Any],
        source: str,
        page: int,
        section: str,
        chunk_sequence: int
    ) -> Dict[str, Any]:
        """Constructs standardized metadata including tracked extracted identifiers and density."""
        identifiers = cls.extract_chunk_identifiers(content)
        words_count = max(1, len(content.split()))
        density = round(len(identifiers) / words_count, 4)

        return {
            **signals,
            "source": source,
            "page": page,
            "section": section,
            "chunk_sequence": chunk_sequence,
            "extracted_identifiers": identifiers,
            "extracted_identifiers_str": ", ".join(identifiers),
            "identifier_density": density
        }

