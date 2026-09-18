import json
import logging
from typing import AsyncIterator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from backend.models.schemas import ChatRequest, ChatResponse, QueryClass
from backend.api.search import execute_hybrid_retrieval
from backend.retrieval.context_builder import ContextBuilder
from backend.retrieval.reranker import CodeAwareReranker
from backend.llm.prompts import build_rag_prompt, get_refusal_response, get_missing_keyword_refusal
from backend.llm.ollama_client import ollama_client
from backend.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

async def stream_chat_generator(req: ChatRequest) -> AsyncIterator[str]:
    """Streams SSE events with retrieval metadata first, followed by Ollama tokens."""
    # 1. Execute retrieval
    search_resp = execute_hybrid_retrieval(req.message, top_k=settings.TOP_K_RERANKED)
    query_class = QueryClass(search_resp.query_class)
    detected_lang = search_resp.detected_language or "en"
    identifiers = search_resp.extracted_identifiers

    # Check minimum relevance confidence score gating
    relevant_candidates = CodeAwareReranker.filter_by_relevance(
        search_resp.results,
        min_threshold=settings.MIN_RELEVANCE_THRESHOLD
    )

    if not relevant_candidates:
        # Zero-Hallucination Retrieval Threshold Filter:
        # Bypass LLM call entirely and immediately return standardized localized refusal
        init_meta = {
            "event": "metadata",
            "query_class": query_class.value,
            "detected_language": detected_lang,
            "sources": [],
            "debug_info": search_resp.dict() if req.debug else None
        }
        yield f"data: {json.dumps(init_meta)}\n\n"

        if identifiers:
            refusal_msg = get_missing_keyword_refusal(identifiers[0], detected_lang)
        else:
            refusal_msg = get_refusal_response(detected_lang)

        token_event = {"event": "token", "token": refusal_msg}
        yield f"data: {json.dumps(token_event)}\n\n"
        yield "data: {\"event\": \"done\"}\n\n"
        return

    # 2. Assemble context with threshold enforcement and surrounding window expansion
    context_text, citations = ContextBuilder.assemble_context(
        relevant_candidates,
        min_threshold=settings.MIN_RELEVANCE_THRESHOLD,
        queried_identifiers=identifiers
    )

    # 3. Send initial metadata event
    init_meta = {
        "event": "metadata",
        "query_class": query_class.value,
        "detected_language": detected_lang,
        "sources": [c.dict() for c in citations],
        "debug_info": search_resp.dict() if req.debug else None
    }
    yield f"data: {json.dumps(init_meta)}\n\n"

    # 4. Check if Ollama is accessible
    health = await ollama_client.check_health()
    if health["status"] != "online":
        err_event = {
            "event": "token",
            "token": f"⚠️ **Ollama Error**: {health.get('instructions', 'Ollama is not running. Please start Ollama.')}"
        }
        yield f"data: {json.dumps(err_event)}\n\n"
        yield "data: {\"event\": \"done\"}\n\n"
        return

    # 5. Build prompt with detected language and exact identifier grounding
    prompt = build_rag_prompt(
        req.message,
        query_class,
        context_text,
        detected_language=detected_lang,
        queried_identifiers=identifiers
    )
    async for token in ollama_client.generate_stream(prompt):
        token_event = {"event": "token", "token": token}
        yield f"data: {json.dumps(token_event)}\n\n"

    yield "data: {\"event\": \"done\"}\n\n"

@router.post("")
async def chat_endpoint(req: ChatRequest):
    """
    RAG chat endpoint supporting both streaming SSE responses and standard JSON responses.
    """
    if req.stream:
        return StreamingResponse(
            stream_chat_generator(req),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # Non-streaming implementation
    search_resp = execute_hybrid_retrieval(req.message, top_k=settings.TOP_K_RERANKED)
    query_class = QueryClass(search_resp.query_class)
    detected_lang = search_resp.detected_language or "en"
    identifiers = search_resp.extracted_identifiers

    # Check minimum relevance confidence score gating
    relevant_candidates = CodeAwareReranker.filter_by_relevance(
        search_resp.results,
        min_threshold=settings.MIN_RELEVANCE_THRESHOLD
    )

    if not relevant_candidates:
        # Bypass LLM call entirely and immediately return standardized localized refusal
        if identifiers:
            refusal_ans = get_missing_keyword_refusal(identifiers[0], detected_lang)
        else:
            refusal_ans = get_refusal_response(detected_lang)

        return ChatResponse(
            answer=refusal_ans,
            query_class=query_class.value,
            detected_language=detected_lang,
            sources=[],
            debug_info=search_resp if req.debug else None
        )

    context_text, citations = ContextBuilder.assemble_context(
        relevant_candidates,
        min_threshold=settings.MIN_RELEVANCE_THRESHOLD,
        queried_identifiers=identifiers
    )

    health = await ollama_client.check_health()
    if health["status"] != "online":
        answer = f"⚠️ Ollama is not running. Start Ollama and ensure '{settings.LLM_MODEL}' is installed."
    else:
        prompt = build_rag_prompt(
            req.message,
            query_class,
            context_text,
            detected_language=detected_lang,
            queried_identifiers=identifiers
        )
        answer = await ollama_client.generate(prompt)

    return ChatResponse(
        answer=answer,
        query_class=query_class.value,
        detected_language=detected_lang,
        sources=citations,
        debug_info=search_resp if req.debug else None
    )

