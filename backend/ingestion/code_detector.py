import re
from typing import Dict, Any, Tuple, Optional, List
from backend.models.schemas import ChunkType

class CodeDetector:
    """Detects code blocks, programming languages, and evaluates code completeness."""

    # Language syntax patterns
    LANG_PATTERNS = {
        "javascript": [
            re.compile(r"\b(?:const|let|var|function|return|=>|\$\(.*?\)|\.dxDataGrid|\.addEventListener)\b"),
            re.compile(r"\bconsole\.(?:log|error|warn)\b"),
            re.compile(r"\b(?:module\.exports|exports|require\(['\"][^'\"]+['\"]\))\b")
        ],
        "typescript": [
            re.compile(r"\b(?:interface|type\s+[A-Z]|:\s*(?:string|number|boolean|any|void)|implements|enum)\b"),
            re.compile(r"\bimport\s+.*\s+from\s+['\"][^'\"]+['\"]")
        ],
        "python": [
            re.compile(r"\b(?:def\s+[a-zA-Z0-9_]+\s*\(|class\s+[a-zA-Z0-9_]+|import\s+[a-zA-Z0-9_]+|from\s+[a-zA-Z0-9_]+\s+import|elif|self\.)\b"),
            re.compile(r"^\s*#.*$", re.MULTILINE)
        ],
        "sql": [
            re.compile(r"\b(?:SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM|CREATE\s+TABLE|ALTER\s+TABLE|JOIN|WHERE|GROUP\s+BY)\b", re.IGNORECASE)
        ],
        "html": [
            re.compile(r"<\s*(?:div|span|table|form|input|button|script|style|link|html|body|head)\b[^>]*>", re.IGNORECASE)
        ],
        "css": [
            re.compile(r"[a-zA-Z0-9_\-\.\#]+\s*\{\s*[a-zA-Z\-]+\s*:\s*[^;]+;\s*\}")
        ]
    }

    # Signals for complete code detection
    SIGNAL_PATTERNS = {
        "imports": re.compile(r"\b(?:import\s+|from\s+[a-zA-Z0-9_\.\-]+\s+import|require\(|include\b)", re.IGNORECASE),
        "declarations": re.compile(r"\b(?:const|let|var|val)\s+[a-zA-Z0-9_]+\s*=", re.IGNORECASE),
        "functions": re.compile(r"\b(?:function\s+[a-zA-Z0-9_]+\s*\(|def\s+[a-zA-Z0-9_]+\s*\(|=>|\bclass\s+[a-zA-Z0-9_]+)", re.IGNORECASE),
        "initialization": re.compile(r"(?:\$\(.*?\)\.[a-zA-Z0-9_]+|new\s+[a-zA-Z0-9_]+\(|\b[a-zA-Z0-9_]+\.init\(|\.render\(|\.mount\(|\.start\()", re.IGNORECASE),
        "event_handlers": re.compile(r"\b(?:\.on\(|\.addEventListener\(|\bon[A-Z][a-zA-Z0-9]+\s*[:=]|\.click\(|\.change\()", re.IGNORECASE),
        "data_source": re.compile(r"\b(?:dataSource|columns|fields|items|data|api|endpoint)\s*[:=]", re.IGNORECASE),
        "configuration": re.compile(r"\b[a-zA-Z0-9_]*(?:Config|Options|Settings|Props)\s*=\s*\{", re.IGNORECASE)
    }

    # Heading signals
    HEADING_COMPLETE_SIGNALS = [
        "complete code", "complete implementation", "full implementation",
        "example", "full example", "implementation", "grid creation",
        "configuration and usage", "setup guide", "production example"
    ]
    HEADING_SNIPPET_SIGNALS = [
        "snippet", "quick tip", "note", "syntax", "one-liner", "short example"
    ]

    @classmethod
    def detect_language(cls, text: str) -> Optional[str]:
        scores = {}
        for lang, patterns in cls.LANG_PATTERNS.items():
            matches = sum(len(p.findall(text)) for p in patterns)
            if matches > 0:
                scores[lang] = matches

        if not scores:
            return None
        return max(scores, key=scores.get)

    @classmethod
    def is_code_block(cls, text: str) -> bool:
        """Determines if the entire block is primarily code."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False

        # If it looks like natural paragraphs of English, it's not pure code
        sentences = [l for l in lines if l.endswith((".", "?", "!")) and " " in l and not l.startswith(("//", "#", "/*", "*"))]
        if len(sentences) > 0 and len(sentences) >= len(lines) * 0.5:
            return False

        # Count code markers
        code_markers = sum(
            1 for line in lines
            if line.endswith((";", "{", "}", ")", "]", ",")) or
            line.startswith(("const ", "let ", "var ", "function ", "def ", "import ", "from ", "class ", "return ", "if (", "for (", "$( ", "$(")) or
            line.startswith(("//", "#", "/*", "*"))
        )

        return (code_markers / len(lines)) >= 0.45 or len(lines) == 1 and bool(re.search(r"[\(\)\{\}\;\=\>]", lines[0]))

    @classmethod
    def analyze_completeness(cls, code: str, section: str = "", parent_section: str = "") -> Tuple[float, Dict[str, Any]]:
        """
        Analyzes a code snippet and returns (code_completeness_score, signals).
        Score ranges from 0.0 to 1.0.
        """
        lines = [l for l in code.splitlines() if l.strip()]
        line_count = len(lines)
        if line_count == 0:
            return 0.0, {}

        signals = {
            "has_imports": bool(cls.SIGNAL_PATTERNS["imports"].search(code)),
            "has_declarations": bool(cls.SIGNAL_PATTERNS["declarations"].search(code)),
            "has_functions": bool(cls.SIGNAL_PATTERNS["functions"].search(code)),
            "has_initialization": bool(cls.SIGNAL_PATTERNS["initialization"].search(code)),
            "has_event_handlers": bool(cls.SIGNAL_PATTERNS["event_handlers"].search(code)),
            "has_data_source": bool(cls.SIGNAL_PATTERNS["data_source"].search(code)),
            "has_config_object": bool(cls.SIGNAL_PATTERNS["configuration"].search(code)),
            "line_count": line_count,
            "char_count": len(code)
        }

        # Base score from size/length
        if line_count <= 2:
            base_score = 0.15  # Tiny isolated snippet like $(container).dxDataGrid(gridConfig);
        elif line_count <= 5:
            base_score = 0.30
        elif line_count <= 10:
            base_score = 0.50
        elif line_count <= 25:
            base_score = 0.70
        else:
            base_score = 0.85

        # Add points for structural signals
        score = base_score

        # If it has both a config/data declaration AND an initialization, that's a complete working unit!
        if (signals["has_declarations"] or signals["has_config_object"]) and signals["has_initialization"]:
            score += 0.25

        if signals["has_data_source"]:
            score += 0.10

        if signals["has_functions"]:
            score += 0.10

        if signals["has_imports"]:
            score += 0.10

        if signals["has_event_handlers"]:
            score += 0.05

        # Balanced braces check
        open_braces = code.count("{")
        close_braces = code.count("}")
        open_parens = code.count("(")
        close_parens = code.count(")")
        if open_braces > 0 and open_braces == close_braces and open_parens == close_parens:
            score += 0.05

        # Check section heading context
        heading_context = f"{section} {parent_section}".lower()
        if any(h in heading_context for h in cls.HEADING_COMPLETE_SIGNALS):
            score += 0.15
        if any(h in heading_context for h in cls.HEADING_SNIPPET_SIGNALS):
            score -= 0.15

        # Tiny isolated snippets (< 4 lines) must not exceed 0.35 regardless of other factors
        if line_count <= 3 and not (signals["has_declarations"] and signals["has_initialization"]):
            score = min(score, 0.30)

        # Cap between 0.0 and 1.0
        final_score = max(0.0, min(1.0, round(score, 2)))
        return final_score, signals

    @classmethod
    def classify_chunk_type(cls, text: str, is_code: bool, completeness_score: float, is_table: bool = False, is_procedure: bool = False, is_config: bool = False) -> ChunkType:
        if is_table:
            return ChunkType.TABLE
        if is_procedure:
            return ChunkType.PROCEDURE
        if is_code:
            if completeness_score >= 0.65:
                return ChunkType.COMPLETE_CODE
            elif is_config:
                return ChunkType.CONFIGURATION
            else:
                return ChunkType.CODE_SNIPPET
        else:
            if is_config:
                return ChunkType.CONFIGURATION
            # Check if prose contains significant embedded code
            if "`" in text or "function" in text or "{" in text:
                return ChunkType.MIXED
            return ChunkType.PROSE
