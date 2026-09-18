from typing import Dict, List, Optional
from backend.models.schemas import QueryClass

SYSTEM_PROMPT = """You are an airtight, zero-hallucination technical documentation extraction engine.
You answer user queries strictly using the provided DOCUMENTATION CONTEXT.

CRITICAL OPERATIONAL RULES:

1. STRICT CLOSED-DOMAIN GROUNDING (ZERO OUTSIDE KNOWLEDGE):
   - Answer ONLY and EXCLUSIVELY from the provided DOCUMENTATION CONTEXT.
   - You are STRICTLY FORBIDDEN from using pre-training memory, external world knowledge, or unmentioned coding libraries.
   - When the user asks about technical concepts, entities, or code, thoroughly extract and synthesize all relevant details, architecture, definitions, properties, and code lines directly from the context.
   - If the retrieved context genuinely does not contain the answer, respond with:
     "The requested information is not available in the provided technical documentation." (or localized equivalent).

2. CODE SYNTHESIS BAN:
   - Never infer, invent, or extrapolate unmentioned variables, functions, API signatures, or configuration properties.
   - When asked for complete code, prioritize the COMPLETE IMPLEMENTATION from the documentation.
   - If the retrieved context only contains an isolated snippet, DO NOT invent missing lines. Explicitly state:
     "The documentation only provides a partial snippet and does not provide the complete implementation in the retrieved material." (or localized equivalent).

3. NATIVE MULTILINGUAL SUPPORT & MIRRORING:
   - Always formulate your final response, explanations, summaries, and structural descriptions in the exact language used by the user.
   - CODE & IDENTIFIER PRESERVATION: While narrative text, explanations, and citations match the user's query language, ALL source code, variable names, function signatures, configurations, parameters, and API endpoints MUST remain in their original raw programming syntax without being translated.

4. EXACT CITATIONS:
   - Clearly cite the source document, page number, and section for every factual claim or code block:
     Source: <filename> | Page: <page> | Section: <section>

5. RESPONSE STRUCTURE ACCORDING TO INTENT:
   - COMPLETE_CODE: Full implementation, dependencies/variables, configuration, and exact citation.
   - EXPLANATION: Definition, purpose, conceptual overview, and citation.
   - CONFIGURATION: Options, properties, example schema, and citation.
   - TROUBLESHOOTING: Known issues, prerequisites, common pitfalls, and citation.
"""

LOCALIZED_REFUSALS: Dict[str, str] = {
    "en": "The requested information is not available in the provided technical documentation.",
    "hi": "अनुरोधित जानकारी प्रदान किए गए तकनीकी दस्तावेज़ में उपलब्ध नहीं है।",
    "es": "La información solicitada no está disponible en la documentación técnica proporcionada.",
    "fr": "Les informations demandées ne sont pas disponibles dans la documentation technique fournie.",
    "de": "Die angeforderten Informationen sind in der bereitgestellten technischen Dokumentation nicht verfügbar.",
    "ja": "リクエストされた情報は提供された技術ドキュメントには記載されていません。",
    "zh": "所请求的信息在提供的技术文档中不可用。",
    "ru": "Запрашиваемая информация отсутствует в предоставленной технической документации.",
    "it": "Le informazioni richieste non sono disponibili nella documentazione tecnica fornita.",
    "pt": "As informações solicitadas não estão disponíveis na documentação técnica fornecida."
}

LOCALIZED_MISSING_KEYWORD: Dict[str, str] = {
    "en": "The requested information regarding '{keyword}' is not present in the provided documentation.",
    "hi": "प्रदान किए गए दस्तावेज़ में '{keyword}' के संबंध में अनुरोधित जानकारी मौजूद नहीं है।",
    "es": "La información solicitada sobre '{keyword}' no está presente en la documentación proporcionada.",
    "fr": "Les informations demandées concernant '{keyword}' ne sont pas présentes dans la documentation fournie.",
    "de": "Die angeforderten Informationen zu '{keyword}' sind in der bereitgestellten Dokumentation nicht enthalten.",
    "ja": "提供されたドキュメントには '{keyword}' に関するリクエストされた情報は含まれていません。",
    "zh": "所提供的文档中未包含有关 '{keyword}' 的请求信息。",
    "ru": "Запрашиваемая информация относительно '{keyword}' отсутствует в предоставленной документации.",
    "it": "Le informazioni richieste relative a '{keyword}' non sono presenti nella documentazione fornita.",
    "pt": "As informações solicitadas sobre '{keyword}' não estão presentes na documentação fornecida."
}

LOCALIZED_PARTIAL_WARNINGS: Dict[str, str] = {
    "en": "The documentation only provides a partial snippet and does not provide the complete implementation in the retrieved material.",
    "hi": "दस्तावेज़ केवल एक आंशिक स्निपेट प्रदान करता है और पुनर्प्राप्त सामग्री में पूर्ण कार्यान्वयन प्रदान नहीं करता है।",
    "es": "La documentación solo proporciona un fragmento parcial y no incluye la implementación completa en el material recuperado.",
    "fr": "La documentation ne fournit qu'un extrait partiel et ne fournit pas l'implémentation complète dans le matériel récupéré.",
    "de": "Die Dokumentation enthält nur einen teilweisen Ausschnitt und keine vollständige Implementierung im abgerufenen Material.",
    "ja": "ドキュメントには部分的なスニペットのみが含まれており、取得された資料には完全な実装は提供されていません。",
    "zh": "文档仅提供部分代码片段，所检索的资料中未提供完整的实现。",
    "ru": "Документация содержит только частичный фрагмент и не предоставляет полную реализацию в полученных материалах.",
    "it": "La documentazione fornisce solo uno snippet parziale e non include l'implementazione completa nel materiale recuperato.",
    "pt": "A documentação fornece apenas um trecho parcial e não fornece a implementação completa no material recuperado."
}

LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
    "ja": "Japanese (日本語)",
    "zh": "Chinese (中文)",
    "ru": "Russian (Русский)",
    "it": "Italian (Italiano)",
    "pt": "Portuguese (Português)"
}

def get_refusal_response(language: str = "en") -> str:
    """Returns the standardized refusal message in the target language."""
    return LOCALIZED_REFUSALS.get(language, LOCALIZED_REFUSALS["en"])

def get_missing_keyword_refusal(keyword: str, language: str = "en") -> str:
    """Returns the standardized missing keyword refusal message in the target language."""
    template = LOCALIZED_MISSING_KEYWORD.get(language, LOCALIZED_MISSING_KEYWORD["en"])
    return template.format(keyword=keyword)

def get_partial_warning(language: str = "en") -> str:
    """Returns the snippet warning message in the target language."""
    return LOCALIZED_PARTIAL_WARNINGS.get(language, LOCALIZED_PARTIAL_WARNINGS["en"])

def build_rag_prompt(
    query: str,
    query_class: QueryClass,
    context: str,
    detected_language: str = "en",
    queried_identifiers: Optional[List[str]] = None
) -> str:
    lang_name = LANGUAGE_NAMES.get(detected_language, "English")
    refusal_msg = get_refusal_response(detected_language)
    partial_warn = get_partial_warning(detected_language)

    id_directives = ""
    if queried_identifiers:
        id_directives = (
            f"\nQUERIED IDENTIFIERS: {', '.join(queried_identifiers)}\n"
            f"EXACT IDENTIFIER GROUNDING:\n"
            f"- The user is specifically asking about: {', '.join(queried_identifiers)}.\n"
            f"- Thoroughly extract and explain all definitions, architecture, implementation details, properties, and code related to these identifiers from the DOCUMENTATION CONTEXT above.\n"
        )

    # Intent-specific positive directives
    if query_class == QueryClass.COMPLETE_CODE:
        intent_instruction = (
            "COMPLETE CODE REQUEST: Provide the full, complete source code implementation from the DOCUMENTATION CONTEXT without omitting declarations, configurations, functions, or initialization.\n"
            f"- If the context contains only a partial snippet rather than a complete implementation, output the snippet and include: \"{partial_warn}\""
        )
    elif query_class == QueryClass.CODE:
        intent_instruction = (
            "CODE REQUEST: Extract and present the relevant source code, syntax, and functions from the DOCUMENTATION CONTEXT, and explain how it operates."
        )
    elif query_class == QueryClass.EXPLANATION:
        intent_instruction = (
            "EXPLANATION REQUEST: Provide a comprehensive technical explanation answering the question based on the DOCUMENTATION CONTEXT. Clearly cover definition, architectural role, purpose, and workflow."
        )
    elif query_class == QueryClass.CONFIGURATION:
        intent_instruction = (
            "CONFIGURATION REQUEST: Provide all relevant configuration parameters, settings, schema options, and usage examples present in the DOCUMENTATION CONTEXT."
        )
    elif query_class == QueryClass.TROUBLESHOOTING:
        intent_instruction = (
            "TROUBLESHOOTING REQUEST: Detail the causes, diagnostic steps, prerequisites, and solutions described in the DOCUMENTATION CONTEXT."
        )
    else:
        intent_instruction = (
            "Answer the user's question directly and thoroughly based on the facts and code in the DOCUMENTATION CONTEXT."
        )

    return f"""DOCUMENTATION CONTEXT:
========================================
{context}
========================================

QUERY INTENT: {query_class.value}
USER LANGUAGE: {lang_name} (Code: {detected_language})
{id_directives}
INSTRUCTIONS:
1. Provide a direct, technically accurate answer based strictly on the DOCUMENTATION CONTEXT above.
2. {intent_instruction}
3. CITE SOURCES: Include exact citations (Document name, Page number, and Section) for every fact, explanation, or code block provided.
4. CODE PRESERVATION: Keep ALL programming code, variable names, functions, endpoints, and config keys in their raw original language without translating them. Respond in {lang_name}.
5. ONLY if the DOCUMENTATION CONTEXT above is completely unrelated to the question and contains zero information about it, respond with: "{refusal_msg}"

USER QUESTION:
{query}

Please provide your technical response based STRICTLY on the documentation context above:"""
