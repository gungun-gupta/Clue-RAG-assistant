from fastapi import APIRouter
from backend.models.schemas import SearchQueryRequest, SearchResponse
from backend.database.chroma import db_manager
from backend.retrieval.query_classifier import QueryClassifier
from backend.retrieval.query_expander import QueryExpander
from backend.retrieval.vector_search import VectorSearch
from backend.retrieval.keyword_search import BM25KeywordSearch
from backend.retrieval.reranker import CodeAwareReranker
from backend.config import settings

router = APIRouter(prefix="/search", tags=["search"])

def execute_hybrid_retrieval(query: str, top_k: int = 5, filter_chunk_type: str = None) -> SearchResponse:
    """Core retrieval pipeline shared by search endpoint and chat engine."""
    # 0. Language detection
    detected_language = QueryClassifier.detect_language(query)

    # 1. Query classification
    query_class = QueryClassifier.classify(query, detected_language=detected_language)

    # 2. Extract technical identifiers (CamelCase, snake_case, exact words, or cross-lingual terms)
    identifiers = QueryClassifier.extract_identifiers(query, detected_language=detected_language)

    # 3. Query expansion
    expanded_queries = QueryExpander.expand_query(
        query,
        query_class,
        identifiers,
        detected_language=detected_language
    )

    # 4. Multi-query vector search in ChromaDB
    fetch_k = max(top_k * 3, settings.TOP_K_RETRIEVAL)
    vector_results = VectorSearch.search(
        queries=expanded_queries,
        top_k=fetch_k,
        filter_chunk_type=filter_chunk_type
    )

    # 5. BM25 keyword search over all indexed chunks
    all_chunks = db_manager.get_all_chunks()
    bm25_searcher = BM25KeywordSearch(all_chunks)
    keyword_scores = bm25_searcher.search(query, identifiers=identifiers)

    # Exact Keyword Gating:
    # Guarantee inclusion of all chunks containing exact technical identifiers regardless of vector distance
    if identifiers:
        exact_chunks = bm25_searcher.find_exact_identifier_matches(identifiers)
        existing_candidate_ids = {c.id for c, _ in vector_results}
        for ec in exact_chunks:
            if ec.id not in existing_candidate_ids:
                vector_results.append((ec, 0.55))
                existing_candidate_ids.add(ec.id)

    # 6. Code-aware multi-factor reranking
    reranked_results = CodeAwareReranker.rerank(
        chunks_with_semantic=vector_results,
        keyword_scores=keyword_scores,
        query=query,
        query_class=query_class,
        identifiers=identifiers
    )

    return SearchResponse(
        query=query,
        query_class=query_class.value,
        detected_language=detected_language,
        expanded_queries=expanded_queries,
        extracted_identifiers=identifiers,
        results=reranked_results[:top_k]
    )

@router.post("", response_model=SearchResponse)
async def debug_search(req: SearchQueryRequest):
    """
    Debug retrieval endpoint: retrieves, scores, and reranks candidates
    without invoking the LLM. Ideal for verifying complete code vs snippet ranking.
    """
    return execute_hybrid_retrieval(
        query=req.query,
        top_k=req.top_k,
        filter_chunk_type=req.filter_chunk_type
    )
