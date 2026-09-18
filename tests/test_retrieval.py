import pytest
from backend.models.schemas import QueryClass, DocumentChunk, ChunkType
from backend.retrieval.query_classifier import QueryClassifier
from backend.retrieval.query_expander import QueryExpander
from backend.retrieval.reranker import CodeAwareReranker

def test_query_classification():
    assert QueryClassifier.classify("Give me complete code for creating a Segment grid") == QueryClass.COMPLETE_CODE
    assert QueryClassifier.classify("Give me the complete production-ready code for creating a Segment grid") == QueryClass.COMPLETE_CODE
    assert QueryClassifier.classify("Give me full implementation of Segment grid") == QueryClass.COMPLETE_CODE
    assert QueryClassifier.classify("What is a Segment?") == QueryClass.EXPLANATION
    assert QueryClassifier.classify("Explain how Segment works") == QueryClass.EXPLANATION
    assert QueryClassifier.classify("How do I configure a Segment grid?") == QueryClass.CONFIGURATION
    assert QueryClassifier.classify("Why is my Segment grid not loading?") == QueryClass.TROUBLESHOOTING
    assert QueryClassifier.classify("Give me code for creating a Segment grid") == QueryClass.CODE

def test_identifier_extraction():
    query = "Give me complete code for creating a Segment grid with dxDataGrid and gridConfig"
    idents = QueryClassifier.extract_identifiers(query)

    assert "dxDataGrid" in idents
    assert "gridConfig" in idents
    assert "Segment" in idents

def test_query_expansion():
    query = "Give me complete code for creating a Segment grid"
    idents = ["Segment", "grid"]
    expansions = QueryExpander.expand_query(query, QueryClass.COMPLETE_CODE, idents)

    assert len(expansions) >= 3
    assert any("complete implementation" in e.lower() for e in expansions)
    assert any("Segment grid" in e for e in expansions)

def test_reranker_prioritizes_complete_code_for_complete_queries():
    # Construct snippet chunk (Page 16)
    snippet_chunk = DocumentChunk(
        id="snippet_1",
        doc_id="doc_1",
        source="AMS.pdf",
        page=16,
        section="Quick Snippet",
        parent_section="",
        chunk_type=ChunkType.CODE_SNIPPET,
        language="javascript",
        code_size=1,
        code_completeness_score=0.20,
        content="$(container).dxDataGrid(gridConfig);"
    )

    # Construct complete code chunk (Page 17)
    complete_chunk = DocumentChunk(
        id="complete_1",
        doc_id="doc_1",
        source="AMS.pdf",
        page=17,
        section="Segment Grid Implementation",
        parent_section="Complete Code",
        chunk_type=ChunkType.COMPLETE_CODE,
        language="javascript",
        code_size=18,
        code_completeness_score=0.92,
        has_declarations=True,
        has_functions=True,
        has_initialization=True,
        content="""
const gridConfig = { dataSource: data, columns: ['id', 'name'] };
function loadSegments() { return fetch('/api/segments'); }
$(container).dxDataGrid(gridConfig);
        """.strip()
    )

    # Simulate vector search where snippet happens to have slightly higher cosine similarity
    chunks_with_semantic = [
        (snippet_chunk, 0.91),    # snippet got 0.91 semantic
        (complete_chunk, 0.88)    # complete code got 0.88 semantic
    ]

    keyword_scores = {
        "snippet_1": 0.85,
        "complete_1": 0.95
    }

    # Execute reranker with COMPLETE_CODE query
    reranked = CodeAwareReranker.rerank(
        chunks_with_semantic=chunks_with_semantic,
        keyword_scores=keyword_scores,
        query="Give me complete code for creating a Segment grid",
        query_class=QueryClass.COMPLETE_CODE,
        identifiers=["Segment", "grid", "dxDataGrid"]
    )

    # Complete implementation MUST outrank the snippet
    assert reranked[0].chunk_id == "complete_1"
    assert reranked[1].chunk_id == "snippet_1"
    assert reranked[0].final_score > reranked[1].final_score
    print(f"Complete code final: {reranked[0].final_score} vs Snippet final: {reranked[1].final_score}")

def test_language_detection():
    assert QueryClassifier.detect_language("Give me complete code for creating a Segment grid") == "en"
    assert QueryClassifier.detect_language("सेगमेंट ग्रिड बनाने का पूरा कोड दें") == "hi"
    assert QueryClassifier.detect_language("Dame el código completo para crear un grid de Segment") == "es"
    assert QueryClassifier.detect_language("Comment configurer le Segment grid dans AMS?") == "fr"
    assert QueryClassifier.detect_language("Wie konfiguriere ich das Segment Grid?") == "de"
    assert QueryClassifier.detect_language("セグメントグリッドの完全なコードをください") == "ja"

def test_multilingual_classification_and_cross_lingual_expansion():
    hindi_query = "सेगमेंट ग्रिड बनाने का पूरा कोड दें"
    lang_hi = QueryClassifier.detect_language(hindi_query)
    assert lang_hi == "hi"

    q_class_hi = QueryClassifier.classify(hindi_query, detected_language=lang_hi)
    assert q_class_hi == QueryClass.COMPLETE_CODE

    idents_hi = QueryClassifier.extract_identifiers(hindi_query, detected_language=lang_hi)
    assert "Segment" in idents_hi
    assert "grid" in idents_hi

    expansions_hi = QueryExpander.expand_query(hindi_query, q_class_hi, idents_hi, detected_language=lang_hi)
    assert any("Segment grid complete implementation" in e for e in expansions_hi)

    spanish_query = "Dame el código completo para crear un grid de Segment"
    lang_es = QueryClassifier.detect_language(spanish_query)
    assert lang_es == "es"

    q_class_es = QueryClassifier.classify(spanish_query, detected_language=lang_es)
    assert q_class_es == QueryClass.COMPLETE_CODE

    idents_es = QueryClassifier.extract_identifiers(spanish_query, detected_language=lang_es)
    assert "Segment" in idents_es

def test_out_of_scope_relevance_filtering_and_refusal():
    from backend.llm.prompts import get_refusal_response
    from backend.config import settings

    # Simulate an irrelevant query where candidates have very low similarity
    dummy_chunk = DocumentChunk(
        id="chunk_unrelated",
        doc_id="doc_1",
        source="AMS.pdf",
        page=1,
        section="Intro",
        content="General warehouse introduction."
    )

    chunks_with_semantic = [(dummy_chunk, 0.12)]  # Very low semantic score
    keyword_scores = {"chunk_unrelated": 0.0}

    reranked = CodeAwareReranker.rerank(
        chunks_with_semantic=chunks_with_semantic,
        keyword_scores=keyword_scores,
        query="How do I cook lasagna with cheese and tomato?",
        query_class=QueryClass.EXPLANATION,
        identifiers=[]
    )

    assert len(reranked) > 0
    assert reranked[0].final_score < settings.MIN_RELEVANCE_THRESHOLD

    # Test filtering helper
    filtered = CodeAwareReranker.filter_by_relevance(reranked, min_threshold=settings.MIN_RELEVANCE_THRESHOLD)
    assert len(filtered) == 0
    assert not CodeAwareReranker.has_sufficient_relevance(reranked, min_threshold=settings.MIN_RELEVANCE_THRESHOLD)

    # Test standardized refusal strings
    en_refusal = get_refusal_response("en")
    assert en_refusal == "The requested information is not available in the provided technical documentation."

    hi_refusal = get_refusal_response("hi")
    assert hi_refusal == "अनुरोधित जानकारी प्रदान किए गए तकनीकी दस्तावेज़ में उपलब्ध नहीं है।"

    es_refusal = get_refusal_response("es")
    assert es_refusal == "La información solicitada no está disponible en la documentación técnica proporcionada."

def test_exact_keyword_gating_prioritizes_exact_matches():
    # Construct a chunk with generic high semantic score but without the exact requested identifier
    generic_chunk = DocumentChunk(
        id="chunk_generic",
        doc_id="doc_1",
        source="AMS.pdf",
        page=5,
        section="Warehouse Logistics",
        content="General overview of warehouse management and storage units.",
        chunk_type=ChunkType.PROSE,
        code_completeness_score=0.0
    )

    # Construct a chunk containing the exact technical identifier 'dxDataGrid'
    exact_chunk = DocumentChunk(
        id="chunk_exact",
        doc_id="doc_1",
        source="AMS.pdf",
        page=17,
        section="Complete Segment Grid Implementation",
        content="const gridConfig = { dataSource: data };\n$(container).dxDataGrid(gridConfig);",
        chunk_type=ChunkType.COMPLETE_CODE,
        code_completeness_score=0.90
    )

    # Even if generic chunk has a higher initial vector similarity (0.85 vs 0.60),
    # exact keyword gating MUST guarantee that the chunk with 'dxDataGrid' ranks #1!
    chunks_with_semantic = [
        (generic_chunk, 0.85),
        (exact_chunk, 0.60)
    ]

    keyword_scores = {
        "chunk_generic": 0.0,
        "chunk_exact": 0.95
    }

    reranked = CodeAwareReranker.rerank(
        chunks_with_semantic=chunks_with_semantic,
        keyword_scores=keyword_scores,
        query="Show me the dxDataGrid setup",
        query_class=QueryClass.CODE,
        identifiers=["dxDataGrid"]
    )

    assert reranked[0].chunk_id == "chunk_exact"
    assert reranked[0].final_score > reranked[1].final_score
    assert "dxDataGrid" in reranked[0].content

def test_missing_keyword_refusal():
    from backend.llm.prompts import get_missing_keyword_refusal

    en_msg = get_missing_keyword_refusal("MaterialInventoryAct", "en")
    assert en_msg == "The requested information regarding 'MaterialInventoryAct' is not present in the provided documentation."

    hi_msg = get_missing_keyword_refusal("MaterialInventoryAct", "hi")
    assert hi_msg == "प्रदान किए गए दस्तावेज़ में 'MaterialInventoryAct' के संबंध में अनुरोधित जानकारी मौजूद नहीं है।"

    es_msg = get_missing_keyword_refusal("MaterialInventoryAct", "es")
    assert es_msg == "La información solicitada sobre 'MaterialInventoryAct' no está presente en la documentación proporcionada."

def test_surrounding_context_expansion():
    from backend.retrieval.context_builder import ContextBuilder
    from backend.models.schemas import SearchResult

    candidate = SearchResult(
        chunk_id="chunk_target",
        doc_id="test_ams_doc",
        source="AMS.pdf",
        page=17,
        section="Complete Segment Grid Implementation",
        parent_section="",
        chunk_type="complete_code",
        content="$(container).dxDataGrid(gridConfig);",
        final_score=0.92,
        chunk_index=0,
        chunk_sequence=0
    )

    context_text, citations = ContextBuilder.assemble_context(
        [candidate],
        min_threshold=0.35,
        queried_identifiers=["dxDataGrid"]
    )

    assert len(citations) > 0
    assert "dxDataGrid" in context_text
    assert "COMPLETE_CODE" in context_text


