import re
import unicodedata
from typing import List, Dict, Tuple, Optional
from backend.models.schemas import QueryClass

class QueryClassifier:
    """Classifies user query intent, detects language, and extracts technical entities/identifiers."""

    PATTERNS = {
        QueryClass.COMPLETE_CODE: [
            # English
            re.compile(r"\b(?:complete\s+code|production[- ]ready|full\s+implementation|entire\s+code|complete\s+implementation|give\s+me\s+complete|full\s+code|full\s+working\s+code|all\s+the\s+code)\b", re.IGNORECASE),
            re.compile(r"\b(?:complete\s+example|full\s+example)\b", re.IGNORECASE),
            # Hindi
            re.compile(r"(?:पूरा\s+कोड|पूर्ण\s+कोड|संपूर्ण\s+कोड|पूरा\s+कार्यान्वयन|पूर्ण\s+कार्यान्वयन|पूरा\s+काम\s+करने\s+वाला\s+कोड)", re.IGNORECASE),
            # Spanish
            re.compile(r"\b(?:c[oó]digo\s+completo|implementaci[oó]n\s+completa|todo\s+el\s+c[oó]digo|ejemplo\s+completo)\b", re.IGNORECASE),
            # French
            re.compile(r"\b(?:code\s+complet|impl[eé]mentation\s+compl[eè]te|tout\s+le\s+code|exemple\s+complet)\b", re.IGNORECASE),
            # German
            re.compile(r"\b(?:vollst[aä]ndiger\s+code|kompletter\s+code|vollst[aä]ndige\s+implementierung|ganzer\s+code)\b", re.IGNORECASE),
            # Japanese
            re.compile(r"(?:完全なコード|完全な実装|すべてのコード|全体コード|完全な例)"),
            # Chinese
            re.compile(r"(?:完整代码|全部代码|完整实现|全套代码|完整示例)"),
            # Russian
            re.compile(r"(?:полный\s+код|полная\s+реализация|весь\s+код|полный\s+пример)", re.IGNORECASE)
        ],
        QueryClass.CONFIGURATION: [
            # English
            re.compile(r"\b(?:how\s+(?:do\s+i|to)\s+configure|configuration|options?|settings?|properties|columns?|datasource|config)\b", re.IGNORECASE),
            # Hindi
            re.compile(r"(?:कॉन्फ़िगर|कॉन्फ़िगरेशन|सेटिंग्स?|विकल्प|प्रॉपर्टी|कॉन्फ़िग)", re.IGNORECASE),
            # Spanish
            re.compile(r"\b(?:c[oó]mo\s+configurar|configuraci[oó]n|opciones|ajustes|propiedades|config)\b", re.IGNORECASE),
            # French
            re.compile(r"\b(?:comment\s+configurer|configuration|options?|param[eè]tres|propri[eé]t[eé]s)\b", re.IGNORECASE),
            # German
            re.compile(r"\b(?:wie\s+konfiguriere\s+ich|konfiguration|einstellungen|optionen|eigenschaften)\b", re.IGNORECASE),
            # Japanese
            re.compile(r"(?:設定方法|構成|設定|オプション|プロパティ)"),
            # Chinese
            re.compile(r"(?:如何配置|配置|设置|选项|属性)"),
            # Russian
            re.compile(r"(?:как\s+настроить|конфигурация|настройки|параметры|свойства)", re.IGNORECASE)
        ],
        QueryClass.TROUBLESHOOTING: [
            # English
            re.compile(r"\b(?:why\s+is\b.*?\b(?:not\s+working|failing|broken|not\s+loading)|error|exception|debug|issue|fail(?:ed|s|ing)?|crash|wrong)\b", re.IGNORECASE),
            # Hindi
            re.compile(r"(?:काम\s+नहीं\s+कर\s+रहा|त्रुटि|समस्या|अपवाद|विफल|खराब|लोड\s+नहीं\s+हो\s+रहा)", re.IGNORECASE),
            # Spanish
            re.compile(r"\b(?:no\s+funciona|no\s+carga|error|excepci[oó]n|fallo|fallando|problema)\b", re.IGNORECASE),
            # French
            re.compile(r"\b(?:ne\s+fonctionne\s+pas|ne\s+charge\s+pas|erreur|exception|[eé]chec|probl[eè]me)\b", re.IGNORECASE),
            # German
            re.compile(r"\b(?:funktioniert\s+nicht|l[aä]dt\s+nicht|fehler|ausnahme|problem)\b", re.IGNORECASE),
            # Japanese
            re.compile(r"(?:動作しない|動かない|読み込めない|エラー|例外|失敗|問題)"),
            # Chinese
            re.compile(r"(?:不工作|不能运行|无法加载|报错|错误|异常|失败|故障)"),
            # Russian
            re.compile(r"(?:не\s+работает|не\s+загружается|ошибка|исключение|сбой|проблема)", re.IGNORECASE)
        ],
        QueryClass.CODE: [
            # English
            re.compile(r"\b(?:give\s+me\s+code|show\s+code|code\s+for|how\s+to\s+write|syntax|snippet|initialization\s+code|initialize|implement|function\s+to|script)\b", re.IGNORECASE),
            re.compile(r"\b(?:create|build|generate|setup|make)\b.*?\b(?:grid|table|form|view|api|controller)\b", re.IGNORECASE),
            # Hindi
            re.compile(r"(?:कोड\s+दें|कोड\s+दिखाएं|कोड|फ़ंक्शन|फंक्शन|स्क्रिप्ट|आरंभीकरण|बनाने|बनाएं|कार्यान्वयन)", re.IGNORECASE),
            # Spanish
            re.compile(r"\b(?:dame\s+c[oó]digo|mostrar\s+c[oó]digo|c[oó]digo\s+para|c[oó]mo\s+escribir|sintaxis|funci[oó]n|script|crear|construir)\b", re.IGNORECASE),
            # French
            re.compile(r"\b(?:donne\s+le\s+code|code\s+pour|comment\s+[eé]crire|syntaxe|fonction|script|cr[eé]er)\b", re.IGNORECASE),
            # German
            re.compile(r"\b(?:gib\s+mir\s+code|code\s+f[uü]r|wie\s+schreibe\s+ich|syntax|funktion|skript|erstellen)\b", re.IGNORECASE),
            # Japanese
            re.compile(r"(?:コードをください|コード|構文|スニペット|初期化|関数|スクリプト|作成)"),
            # Chinese
            re.compile(r"(?:给代码|代码|语法|片段|初始化|函数|脚本|创建)"),
            # Russian
            re.compile(r"(?:дай\s+код|покажи\s+код|код\s+для|синтаксис|функция|скрипт|создать)", re.IGNORECASE)
        ],
        QueryClass.EXPLANATION: [
            # English
            re.compile(r"\b(?:what\s+is|what\s+are|explain|overview|describe|definition|concept|meaning\s+of|tell\s+me\s+about)\b", re.IGNORECASE),
            # Hindi
            re.compile(r"(?:क्या\s+है|क्या\s+होता\s+है|समझाएं|विवरण|परिभाषा|अवधारणा|के\s+बारे\s+में)", re.IGNORECASE),
            # Spanish
            re.compile(r"\b(?:qu[eé]\s+es|qu[eé]\s+son|explicar|explica|descripci[oó]n|definici[oó]n|concepto|h[aá]blame\s+de)\b", re.IGNORECASE),
            # French
            re.compile(r"\b(?:qu'est-ce\s+que|qu'est\s+ce|expliquer|vue\s+d'ensemble|d[eé]finition|concept|parle-moi\s+de)\b", re.IGNORECASE),
            # German
            re.compile(r"\b(?:was\s+ist|was\s+sind|erkl[aä]re|erkl[aä]ren|übersicht|definition|konzept)\b", re.IGNORECASE),
            # Japanese
            re.compile(r"(?:とは何ですか|とは|説明して|概要|定義|概念)"),
            # Chinese
            re.compile(r"(?:什么是|解释|概述|说明|定义|概念)"),
            # Russian
            re.compile(r"(?:что\s+такое|объясни|обзор|описание|определение|концепция)", re.IGNORECASE)
        ]
    }

    IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")

    # Cross-lingual technical entity mapping
    CROSS_LINGUAL_TERMS: Dict[str, str] = {
        # Hindi terms & transliterations
        "सेगमेंट": "Segment",
        "ग्रिड": "grid",
        "कोड": "code",
        "कार्यान्वयन": "implementation",
        "कंटेनर": "container",
        "कॉन्फ़िगरेशन": "configuration",
        "कॉन्फ़िगर": "configure",
        "सेटिंग्स": "settings",
        "डेटासोर्स": "dataSource",
        "कॉलम": "columns",
        "आरंभीकरण": "initialization",
        # Spanish terms
        "código": "code",
        "codigo": "code",
        "cuadrícula": "grid",
        "cuadricula": "grid",
        "implementación": "implementation",
        "implementacion": "implementation",
        "configuración": "configuration",
        "configuracion": "configuration",
        "columnas": "columns",
        "fuente de datos": "dataSource",
        "inicialización": "initialization",
        "inicializacion": "initialization",
        # French terms
        "grille": "grid",
        "mise en œuvre": "implementation",
        "paramètres": "settings",
        # German terms
        "gitter": "grid",
        "initialisierung": "initialization",
        # Japanese terms
        "セグメント": "Segment",
        "グリッド": "grid",
        "初期化": "initialization",
        "設定": "configuration",
        # Chinese terms
        "分段": "Segment",
        "网格": "grid",
        "配置": "configuration",
        "初始化": "initialization",
        # Russian terms
        "сегмент": "Segment",
        "сетка": "grid",
        "конфигурация": "configuration",
        "инициализация": "initialization"
    }

    # Distinctive stopwords / tokens for Latin-script language detection
    LATIN_LANGUAGE_MARKERS = {
        "en": {
            "tokens": {"the", "is", "are", "how", "what", "why", "when", "where", "who", "which", "with", "from", "for", "about", "give", "show", "tell", "please", "need", "want", "have", "has", "can", "could", "would", "should", "does", "did", "do", "make", "create", "creating", "code", "implementation"},
            "chars": set()
        },
        "es": {
            "tokens": {"el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en", "para", "por", "con", "como", "cómo", "que", "qué", "dame", "crear", "código", "codigo", "completo", "completa", "ejemplo", "explicar", "función", "funcion", "configurar", "configuración", "configuracion", "donde"},
            "chars": {"¿", "¡", "ñ"}
        },
        "fr": {
            "tokens": {"le", "la", "les", "un", "une", "des", "du", "de", "pour", "par", "avec", "comment", "quoi", "donne", "créer", "creer", "complet", "complète", "exemple", "expliquer", "fonction", "configurer", "configuration", "dans", "sur"},
            "chars": {"ç", "œ", "æ", "ê", "ë", "è", "é", "à", "ù", "î", "ï", "ô", "û"}
        },
        "de": {
            "tokens": {"der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "einer", "und", "für", "mit", "wie", "was", "warum", "vollständiger", "vollstaendiger", "vollständig", "geben", "erstelle", "erstellen", "beispiel", "erkläre", "nicht", "funktioniert"},
            "chars": {"ä", "ö", "ü", "ß"}
        },
        "it": {
            "tokens": {"il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "da", "con", "su", "per", "come", "cosa", "perché", "perche", "dammi", "creare", "codice", "mostra", "spiega", "funzione", "configurazione"},
            "chars": {"à", "è", "é", "ì", "ò", "ù"}
        },
        "pt": {
            "tokens": {"os", "as", "uns", "umas", "dos", "das", "no", "na", "em", "para", "por", "como", "dê-me", "de-me", "criar", "código", "codigo", "mostrar", "explicar", "função", "funcao", "você", "onde"},
            "chars": {"ã", "õ"}
        }
    }

    # Common English stop words to exclude from identifiers
    STOP_WORDS = {
        "the", "and", "for", "with", "this", "that", "what", "how", "why", "when", "where", "who", "which",
        "give", "show", "tell", "make", "create", "need", "want", "help", "please", "explain", "describe",
        "code", "from", "into", "page", "some", "more", "does", "have", "here", "there", "about", "your", "their"
    }

    @classmethod
    def detect_language(cls, query: str) -> str:
        """
        Detects user prompt language using Unicode character blocks (for Devanagari,
        Japanese, Chinese, Cyrillic) and lexical/diacritic markers (for Latin languages).
        Returns ISO-639-1 language code ('en', 'hi', 'es', 'fr', 'de', 'ja', 'zh', 'ru', 'it', 'pt').
        """
        if not query or not query.strip():
            return "en"

        # 1. Non-Latin Unicode Script Checks
        has_devanagari = False
        has_hiragana_katakana = False
        has_cjk_ideograph = False
        has_cyrillic = False

        for char in query:
            code = ord(char)
            if 0x0900 <= code <= 0x097F:
                has_devanagari = True
            elif (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF):
                has_hiragana_katakana = True
            elif 0x4E00 <= code <= 0x9FFF:
                has_cjk_ideograph = True
            elif 0x0400 <= code <= 0x04FF:
                has_cyrillic = True

        if has_devanagari:
            return "hi"
        if has_hiragana_katakana:
            return "ja"
        if has_cjk_ideograph and not has_hiragana_katakana:
            return "zh"
        if has_cyrillic:
            return "ru"

        # 2. Latin Script Analysis
        query_lower = query.lower()
        words = set(re.findall(r"\b\w+\b", query_lower))

        scores: Dict[str, int] = {"en": 0, "es": 0, "fr": 0, "de": 0, "it": 0, "pt": 0}

        for lang, markers in cls.LATIN_LANGUAGE_MARKERS.items():
            # Check matching tokens
            matched_tokens = words.intersection(markers["tokens"])
            scores[lang] += len(matched_tokens) * 2

            # Check special language characters
            for char in markers["chars"]:
                if char in query_lower:
                    scores[lang] += 3

        # If English matches are highest or tied, default to English
        en_score = scores["en"]
        non_en_scores = {l: s for l, s in scores.items() if l != "en"}
        best_non_en_lang, best_non_en_score = max(non_en_scores.items(), key=lambda item: item[1])

        if best_non_en_score > en_score and best_non_en_score >= 2:
            return best_non_en_lang

        return "en"

    @classmethod
    def classify(cls, query: str, detected_language: str = "en") -> QueryClass:
        # Check COMPLETE_CODE first because it takes precedence over general CODE
        for pattern in cls.PATTERNS[QueryClass.COMPLETE_CODE]:
            if pattern.search(query):
                return QueryClass.COMPLETE_CODE

        for pattern in cls.PATTERNS[QueryClass.TROUBLESHOOTING]:
            if pattern.search(query):
                return QueryClass.TROUBLESHOOTING

        for pattern in cls.PATTERNS[QueryClass.CONFIGURATION]:
            if pattern.search(query):
                return QueryClass.CONFIGURATION

        for pattern in cls.PATTERNS[QueryClass.CODE]:
            if pattern.search(query):
                return QueryClass.CODE

        for pattern in cls.PATTERNS[QueryClass.EXPLANATION]:
            if pattern.search(query):
                return QueryClass.EXPLANATION

        # Default fallback: if query contains code terms, treat as CODE; otherwise EXPLANATION
        code_indicators = [
            "function", "grid", "api", "var", "const", "class", "method",
            "फ़ंक्शन", "फंक्शन", "ग्रिड", "código", "función", "cuadrícula"
        ]
        if any(term in query.lower() for term in code_indicators):
            return QueryClass.CODE

        return QueryClass.EXPLANATION

    @classmethod
    def extract_identifiers(cls, query: str, detected_language: str = "en") -> List[str]:
        """
        Extracts technical identifiers:
        1. Translates/normalizes cross-lingual technical terms (e.g. 'सेगमेंट' -> 'Segment', 'ग्रिड' -> 'grid')
        2. Extracts explicit backticked tokens (e.g. `onInitialized`).
        3. Extracts CamelCase, snake_case, or Capitalized technical entities from the query.
        """
        identifiers: List[str] = []
        query_lower = query.lower()

        # 1. Cross-lingual mapping lookup
        for term, mapped_id in cls.CROSS_LINGUAL_TERMS.items():
            if term in query_lower or term in query:
                if mapped_id not in identifiers:
                    identifiers.append(mapped_id)

        # 2. Explicit backticked tokens
        backtick_tokens = re.findall(r"`([^`]+)`", query)
        for bt in backtick_tokens:
            bt_clean = bt.strip()
            if bt_clean and bt_clean not in identifiers:
                identifiers.append(bt_clean)

        # 3. ASCII technical identifiers
        tokens = cls.IDENTIFIER_PATTERN.findall(query)
        for t in tokens:
            lower = t.lower()
            if lower in cls.STOP_WORDS:
                continue

            # CamelCase, snake_case, or capitalized technical term
            is_camel = bool(re.search(r"[a-z][A-Z]", t))
            is_snake = "_" in t
            is_capitalized = t[0].isupper() and len(t) >= 4

            if is_camel or is_snake or is_capitalized:
                if t not in identifiers:
                    identifiers.append(t)

        return list(dict.fromkeys(identifiers))

