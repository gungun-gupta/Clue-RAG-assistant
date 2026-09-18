import re
from typing import List
from backend.models.schemas import QueryClass
from backend.retrieval.query_classifier import QueryClassifier

class QueryExpander:
    """Generates internal retrieval variations and query expansions."""

    @classmethod
    def expand_query(
        cls,
        query: str,
        query_class: QueryClass,
        identifiers: List[str],
        detected_language: str = "en"
    ) -> List[str]:
        variations = [query]

        # Clean base query (strip conversational prefixes)
        cleaned = re.sub(
            r"^(?:give\s+me|show\s+me|how\s+to|what\s+is|can\s+you\s+provide|please\s+show|i\s+need|dame|donne|gib\s+mir)\s+",
            "",
            query,
            flags=re.IGNORECASE
        ).strip()

        if cleaned and cleaned.lower() != query.lower():
            variations.append(cleaned)

        # Core subject extraction: prefer domain identifiers
        filtered_ids = [i for i in identifiers if i.lower() not in ["code", "complete", "implementation"]]
        core_terms = " ".join(filtered_ids) if filtered_ids else (" ".join(identifiers) if identifiers else cleaned)

        # Cross-lingual expansion: ensure core English technical terms are in variations
        if detected_language != "en" and core_terms:
            variations.append(core_terms)

        if query_class == QueryClass.COMPLETE_CODE:
            if core_terms:
                variations.append(f"{core_terms} complete implementation")
                variations.append(f"{core_terms} complete code")
                variations.append(f"{core_terms} configuration")
                variations.append(f"{core_terms} implementation example")
        elif query_class == QueryClass.CODE:
            if core_terms:
                variations.append(f"{core_terms} code")
                variations.append(f"{core_terms} function implementation")
                variations.append(f"{core_terms} initialization")
        elif query_class == QueryClass.CONFIGURATION:
            if core_terms:
                variations.append(f"{core_terms} configuration options")
                variations.append(f"{core_terms} settings properties")
                variations.append(f"{core_terms} config")
        elif query_class == QueryClass.TROUBLESHOOTING:
            if core_terms:
                variations.append(f"{core_terms} error troubleshooting")
                variations.append(f"{core_terms} common issues")
        elif query_class == QueryClass.EXPLANATION:
            if core_terms:
                variations.append(f"{core_terms} overview")
                variations.append(f"{core_terms} definition")

        # Deduplicate preserving order
        seen = set()
        unique_variations = []
        for v in variations:
            v_norm = v.strip().lower()
            if v_norm and v_norm not in seen:
                seen.add(v_norm)
                unique_variations.append(v.strip())

        return unique_variations[:5]
