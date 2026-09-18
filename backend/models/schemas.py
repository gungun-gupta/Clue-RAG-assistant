from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ChunkType(str, Enum):
    PROSE = "prose"
    CODE_SNIPPET = "code_snippet"
    COMPLETE_CODE = "complete_code"
    CONFIGURATION = "configuration"
    TABLE = "table"
    PROCEDURE = "procedure"
    MIXED = "mixed"

class QueryClass(str, Enum):
    EXPLANATION = "EXPLANATION"
    CODE = "CODE"
    COMPLETE_CODE = "COMPLETE_CODE"
    CONFIGURATION = "CONFIGURATION"
    TROUBLESHOOTING = "TROUBLESHOOTING"

class DocumentChunk(BaseModel):
    id: str
    doc_id: str
    source: str
    page: int = 1
    section: str = "General"
    parent_section: str = ""
    chunk_type: ChunkType = ChunkType.PROSE
    language: Optional[str] = None
    code_size: int = 0
    code_completeness_score: float = 0.0
    has_imports: bool = False
    has_functions: bool = False
    has_initialization: bool = False
    implementation_id: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 1
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    file_size: int
    page_count: int
    chunk_count: int
    chunk_type_counts: Dict[str, int] = Field(default_factory=dict)
    upload_timestamp: str
    file_hash: str

class SearchQueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filter_chunk_type: Optional[str] = None

class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    page: int
    section: str
    parent_section: str = ""
    chunk_type: str
    language: Optional[str] = None
    content: str
    code_completeness_score: float = 0.0
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    completeness_score: float = 0.0
    section_score: float = 0.0
    final_score: float = 0.0
    implementation_id: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 1
    chunk_sequence: int = 0

class SearchResponse(BaseModel):
    query: str
    query_class: str
    detected_language: Optional[str] = "en"
    expanded_queries: List[str]
    extracted_identifiers: List[str]
    results: List[SearchResult]

class SourceCitation(BaseModel):
    source: str
    page: int
    section: str
    chunk_type: str
    snippet: str

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant" or "system"
    content: str

class ChatRequest(BaseModel):
    message: str
    stream: bool = True
    debug: bool = False
    conversation_history: List[ChatMessage] = Field(default_factory=list)

class ChatResponse(BaseModel):
    answer: str
    query_class: str
    detected_language: Optional[str] = "en"
    sources: List[SourceCitation]
    debug_info: Optional[SearchResponse] = None
