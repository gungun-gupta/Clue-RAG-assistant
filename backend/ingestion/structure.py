import re
from typing import List, Dict, Any, Tuple
from backend.models.schemas import ChunkType

class StructureDetector:
    """Detects structural elements such as headings, procedures, configurations, and tables."""

    PROCEDURE_PATTERNS = [
        re.compile(r"^\s*(?:step\s*\d+|\d+\.)\s+", re.IGNORECASE),
        re.compile(r"^\s*-\s+(?:install|run|configure|create|open|add|update|verify|deploy|execute)\b", re.IGNORECASE)
    ]

    CONFIG_PATTERNS = [
        re.compile(r"(?:config|options|settings|properties)\s*[:=]\s*\{", re.IGNORECASE),
        re.compile(r"^\s*\{[\s\S]*\}\s*$", re.MULTILINE),
        re.compile(r"^\s*[a-zA-Z0-9_\.\-]+:\s+[a-zA-Z0-9_\.\-]+", re.MULTILINE)
    ]

    TABLE_PATTERNS = [
        re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE),
        re.compile(r"^\s*[\+\-]{3,}[\+\-\| ]+$", re.MULTILINE)
    ]

    HEADING_PATTERNS = [
        re.compile(r"^\s*#{1,6}\s+.+"),
        re.compile(r"^\s*\*\*[^*]{3,80}\*\*\s*$"),
        re.compile(r"^\s*(?:section|chapter|part)\s+\d+[:\.]?\s+.+", re.IGNORECASE)
    ]

    @classmethod
    def is_heading(cls, text: str) -> bool:
        stripped = text.strip()
        if not stripped or len(stripped) > 120 or "\n" in stripped:
            return False
        return any(p.match(stripped) for p in cls.HEADING_PATTERNS)


    @classmethod
    def is_procedure(cls, text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return False

        step_matches = sum(
            1 for line in lines
            if any(pattern.search(line) for pattern in cls.PROCEDURE_PATTERNS)
        )
        return (step_matches / len(lines)) >= 0.4

    @classmethod
    def is_table(cls, text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return False

        pipe_lines = sum(1 for line in lines if line.startswith("|") and line.endswith("|"))
        return (pipe_lines / len(lines)) >= 0.6

    @classmethod
    def is_configuration(cls, text: str) -> bool:
        if cls.is_table(text):
            return False
        # Check for config keywords or JSON/YAML shape
        has_config_keyword = bool(re.search(r"\b(config|options|settings|datasource|columns|properties)\b", text, re.IGNORECASE))
        has_key_value_pairs = len(re.findall(r"^\s*['\"]?[a-zA-Z0-9_-]+['\"]?\s*:\s*", text, re.MULTILINE)) >= 3
        has_braces = "{" in text and "}" in text
        return has_config_keyword and (has_key_value_pairs or has_braces)
