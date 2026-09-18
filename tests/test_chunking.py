import pytest
from backend.ingestion.code_detector import CodeDetector
from backend.ingestion.structure import StructureDetector
from backend.ingestion.chunker import CodeAwareChunker
from backend.ingestion.parser import ParsedDocument, ParsedElement
from backend.models.schemas import ChunkType

def test_language_detection():
    js_code = "const gridConfig = { dataSource: data }; $(container).dxDataGrid(gridConfig);"
    py_code = "def get_segments(self):\n    return self.db.query(Segment).all()"
    sql_code = "SELECT id, name FROM segments WHERE active = 1 ORDER BY id DESC;"

    assert CodeDetector.detect_language(js_code) == "javascript"
    assert CodeDetector.detect_language(py_code) == "python"
    assert CodeDetector.detect_language(sql_code) == "sql"

def test_code_completeness_snippet_vs_complete():
    # 1. Small isolated snippet from user prompt:
    snippet = "$(container).dxDataGrid(gridConfig);"
    snippet_score, snippet_signals = CodeDetector.analyze_completeness(
        snippet,
        section="Quick Snippet",
        parent_section="Tips"
    )

    # 2. Complete implementation from user prompt:
    complete_code = """
const gridConfig = {
    dataSource: data,
    columns: [
        { dataField: "id", caption: "ID" },
        { dataField: "segmentName", caption: "Segment Name" }
    ],
    paging: { pageSize: 10 },
    onInitialized: function(e) {
        console.log("Grid ready");
    }
};

function loadSegments() {
    return $.ajax({ url: "/api/segments", method: "GET" });
}

$(container).dxDataGrid(gridConfig);
    """.strip()

    complete_score, complete_signals = CodeDetector.analyze_completeness(
        complete_code,
        section="Segment Grid Implementation",
        parent_section="Complete Code"
    )

    print(f"Snippet score: {snippet_score}")
    print(f"Complete code score: {complete_score}")

    # The snippet must have a low completeness score (<= 0.35)
    assert snippet_score <= 0.35
    assert not snippet_signals["has_declarations"]
    assert not snippet_signals["has_functions"]

    # The complete implementation must have a high score (>= 0.80)
    assert complete_score >= 0.80
    assert complete_signals["has_declarations"]
    assert complete_signals["has_functions"]
    assert complete_signals["has_initialization"]

    # Significant gap between snippet and complete implementation
    assert complete_score - snippet_score > 0.45

def test_chunker_assigns_correct_types():
    parsed_doc = ParsedDocument(
        filename="AMS_spec.md",
        file_type="markdown",
        file_size=1024,
        page_count=2,
        elements=[
            ParsedElement(
                page_number=1,
                element_type="paragraph",
                content="A Segment represents a logical partition of data in AMS.",
                section="Overview"
            ),
            ParsedElement(
                page_number=2,
                element_type="code_block",
                content="$(container).dxDataGrid(gridConfig);",
                section="Quick Syntax",
                metadata={"language": "javascript"}
            ),
            ParsedElement(
                page_number=2,
                element_type="code_block",
                content="""
const gridConfig = {
    dataSource: "/api/segments",
    columns: ["id", "name"]
};
function init() {
    $(container).dxDataGrid(gridConfig);
}
init();
                """.strip(),
                section="Segment Grid Complete Implementation",
                metadata={"language": "javascript"}
            )
        ]
    )

    chunks = CodeAwareChunker.chunk_document(parsed_doc, doc_id="test_doc")

    assert len(chunks) == 3
    assert chunks[0].chunk_type == ChunkType.PROSE
    assert chunks[1].chunk_type == ChunkType.CODE_SNIPPET
    assert chunks[2].chunk_type == ChunkType.COMPLETE_CODE
    assert chunks[2].code_completeness_score >= 0.70

def test_composite_section_unification_and_atomic_code_integrity():
    # Verify that a heading, explanation, complete code block, and parameter table
    # within the same section remain unified in a single composite chunk without syntax tears
    complete_js = """
const gridConfig = {
    dataSource: new DevExpress.data.CustomStore({
        key: "segmentId",
        load: function() { return loadSegments(); }
    }),
    columns: ["segmentId", "segmentName", "zoneCode"]
};
function loadSegments() {
    return $.ajax({ url: "/api/ams/segments", method: "GET" });
}
$(container).dxDataGrid(gridConfig);
    """.strip()

    parsed_doc = ParsedDocument(
        filename="AMS_spec.md",
        file_type="markdown",
        file_size=2048,
        page_count=1,
        elements=[
            ParsedElement(
                page_number=1,
                element_type="heading",
                content="Complete Segment Grid Implementation",
                section="Complete Segment Grid Implementation"
            ),
            ParsedElement(
                page_number=1,
                element_type="paragraph",
                content="The following component initializes the Segment Grid with custom data loading and column definitions.",
                section="Complete Segment Grid Implementation"
            ),
            ParsedElement(
                page_number=1,
                element_type="code_block",
                content=complete_js,
                section="Complete Segment Grid Implementation",
                metadata={"language": "javascript"}
            ),
            ParsedElement(
                page_number=1,
                element_type="paragraph",
                content="Ensure that the container element exists in the DOM prior to executing the initialization script.",
                section="Complete Segment Grid Implementation"
            )
        ]
    )

    chunks = CodeAwareChunker.chunk_document(parsed_doc, doc_id="test_doc_composite")

    # All elements in this section MUST be unified in a single composite chunk
    assert len(chunks) == 1
    composite_chunk = chunks[0]

    # Check structural heading inclusion
    assert "## Complete Segment Grid Implementation" in composite_chunk.content

    # Check that explanatory prose is not orphaned
    assert "initializes the Segment Grid" in composite_chunk.content
    assert "Ensure that the container element exists" in composite_chunk.content

    # Check atomic code block integrity (complete function definitions intact)
    assert "const gridConfig" in composite_chunk.content
    assert "function loadSegments()" in composite_chunk.content
    assert "$(container).dxDataGrid(gridConfig);" in composite_chunk.content

    # Check classification and completeness
    assert composite_chunk.chunk_type == ChunkType.COMPLETE_CODE
    assert composite_chunk.code_completeness_score >= 0.80

    # Check extracted identifiers
    extracted = composite_chunk.metadata.get("extracted_identifiers", [])
    assert "dxDataGrid" in extracted
    assert "gridConfig" in extracted
    assert "loadSegments" in extracted
    assert "/api/ams/segments" in extracted
    assert composite_chunk.metadata.get("chunk_sequence") == 0

def test_extracted_identifier_tracking():
    code_text = """
class SegmentController {
    constructor() {
        this.apiEndpoint = "/api/v1/segments";
    }
    loadData() {
        const active_status = true;
        return fetch(this.apiEndpoint);
    }
}
const defaultOptions = { mode: "single" };
    """
    identifiers = CodeAwareChunker.extract_chunk_identifiers(code_text)

    assert "SegmentController" in identifiers
    assert "active_status" in identifiers
    assert "defaultOptions" in identifiers
    assert "/api/v1/segments" in identifiers

