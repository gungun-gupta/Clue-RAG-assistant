import os
import pytest
from pathlib import Path
from backend.ingestion.parser import DocumentParser
from backend.ingestion.chunker import CodeAwareChunker
from backend.ingestion.metadata import MetadataManager
from backend.database.chroma import db_manager
from backend.api.search import execute_hybrid_retrieval
from backend.models.schemas import DocumentInfo

SAMPLE_DOC_TEXT = """# AMS Technical Specification

## Overview of Segment
<!-- page 10 -->
A Segment represents a logical partition of inventory items within the AMS warehouse management system.
Segments define storage boundaries, velocity zones, and picking allocation groups.
Every material item must belong to at least one active Segment before it can be assigned to a location bin.

## Quick Syntax
<!-- page 16 -->
```javascript
$(container).dxDataGrid(gridConfig);
```

## Complete Segment Grid Implementation
<!-- page 17 -->
```javascript
// Complete Segment Grid Component
const gridConfig = {
    dataSource: new DevExpress.data.CustomStore({
        key: "segmentId",
        load: function() {
            return loadSegments();
        }
    }),
    columns: [
        { dataField: "segmentId", caption: "Segment ID", width: 100 },
        { dataField: "segmentName", caption: "Segment Name" },
        { dataField: "zoneCode", caption: "Zone Code" },
        { dataField: "activeStatus", caption: "Active Status", dataType: "boolean" }
    ],
    paging: { pageSize: 25 },
    selection: { mode: "single" },
    onInitialized: function(e) {
        console.log("Segment Grid initialized successfully.");
    }
};

function loadSegments() {
    return $.ajax({
        url: "/api/ams/segments",
        method: "GET",
        dataType: "json"
    });
}

// Render the complete segment grid into the designated container
$(container).dxDataGrid(gridConfig);
```
"""

@pytest.fixture(scope="module", autouse=True)
def setup_test_index():
    # Write sample fixture file
    test_file = Path("data/documents/AMS_draftprefinal.md")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(SAMPLE_DOC_TEXT, encoding="utf-8")

    # Clear and index fixture
    db_manager.clear_all()
    parsed_doc = DocumentParser.parse_file(str(test_file))
    doc_id = "test_ams_doc"
    chunks = CodeAwareChunker.chunk_document(parsed_doc, doc_id=doc_id)
    doc_info = DocumentInfo(
        doc_id=doc_id,
        filename=test_file.name,
        file_type="markdown",
        file_size=len(SAMPLE_DOC_TEXT),
        page_count=parsed_doc.page_count,
        chunk_count=len(chunks),
        chunk_type_counts=MetadataManager.compute_chunk_statistics(chunks),
        upload_timestamp="2026-09-18 12:00:00",
        file_hash="test_hash_123"
    )
    db_manager.add_chunks(doc_id=doc_id, doc_info=doc_info, chunks=chunks)
    yield
    # Cleanup fixture
    if test_file.exists():
        test_file.unlink()

def test_complete_code_query_prioritizes_complete_implementation():
    # User asks for complete code
    query = "Give me complete code for creating a Segment grid"
    response = execute_hybrid_retrieval(query, top_k=5)

    assert response.query_class == "COMPLETE_CODE"
    assert len(response.results) > 0

    top_result = response.results[0]
    print(f"\nTop result for '{query}':")
    print(f"Page: {top_result.page}, Section: {top_result.section}, Type: {top_result.chunk_type}, Score: {top_result.final_score}")

    # The #1 result MUST be the Page 17 complete implementation, NOT the Page 16 snippet!
    assert top_result.page == 17
    assert top_result.chunk_type == "complete_code"
    assert "const gridConfig" in top_result.content
    assert "loadSegments" in top_result.content

    # If the Page 16 snippet is in the results, it must have a lower score
    snippet_results = [r for r in response.results if r.page == 16]
    if snippet_results:
        assert top_result.final_score > snippet_results[0].final_score

def test_explanation_query_prioritizes_prose():
    # User asks for conceptual explanation
    query = "What is a Segment?"
    response = execute_hybrid_retrieval(query, top_k=5)

    assert response.query_class == "EXPLANATION"
    assert len(response.results) > 0

    top_result = response.results[0]
    print(f"\nTop result for '{query}':")
    print(f"Page: {top_result.page}, Section: {top_result.section}, Type: {top_result.chunk_type}, Score: {top_result.final_score}")

    # The #1 result MUST be the Page 10 explanation prose
    assert top_result.page == 10
    assert top_result.chunk_type in ["prose", "mixed"]
    assert "logical partition" in top_result.content.lower()

def test_initialization_code_query():
    query = "What is the Segment grid initialization code?"
    response = execute_hybrid_retrieval(query, top_k=5)

    assert len(response.results) > 0
    # Must retrieve code mentioning dxDataGrid and container
    found_init = any("dxDataGrid" in r.content for r in response.results)
    assert found_init

def test_multilingual_hindi_query_retrieves_english_chunk():
    query = "सेगमेंट ग्रिड बनाने का पूरा कोड दें"
    response = execute_hybrid_retrieval(query, top_k=5)

    assert response.detected_language == "hi"
    assert response.query_class == "COMPLETE_CODE"
    assert len(response.results) > 0

    top_result = response.results[0]
    assert top_result.page == 17
    assert top_result.chunk_type == "complete_code"
    assert "const gridConfig" in top_result.content
    assert "loadSegments" in top_result.content

def test_multilingual_spanish_query_retrieves_english_chunk():
    query = "Dame el código completo para crear un grid de Segment"
    response = execute_hybrid_retrieval(query, top_k=5)

    assert response.detected_language == "es"
    assert response.query_class == "COMPLETE_CODE"
    assert len(response.results) > 0

    top_result = response.results[0]
    assert top_result.page == 17
    assert top_result.chunk_type == "complete_code"
    assert "const gridConfig" in top_result.content

@pytest.mark.asyncio
async def test_chat_refusal_on_irrelevant_query():
    from backend.models.schemas import ChatRequest
    from backend.api.chat import chat_endpoint

    req = ChatRequest(message="How do I bake a chocolate cake at home?", stream=False)
    resp = await chat_endpoint(req)
    assert resp.answer == "The requested information is not available in the provided technical documentation."
    assert len(resp.sources) == 0

@pytest.mark.asyncio
async def test_chat_refusal_on_irrelevant_hindi_query():
    from backend.models.schemas import ChatRequest
    from backend.api.chat import chat_endpoint

    req = ChatRequest(message="घर पर चॉकलेट केक कैसे बनाएं?", stream=False)
    resp = await chat_endpoint(req)
    assert resp.answer == "अनुरोधित जानकारी प्रदान किए गए तकनीकी दस्तावेज़ में उपलब्ध नहीं है।"
    assert len(resp.sources) == 0

