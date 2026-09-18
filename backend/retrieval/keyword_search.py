import re
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from backend.models.schemas import DocumentChunk

ENGLISH_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
    "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves"
}

class BM25KeywordSearch:
    """Performs BM25 keyword search and exact identifier matching."""

    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.corpus_tokens = [self._tokenize(self._get_searchable_text(c)) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def _get_searchable_text(self, chunk: DocumentChunk) -> str:
        # Boost section and identifiers in the searchable text
        sec = f"{chunk.section} {chunk.parent_section} " * 2
        return f"{sec} {chunk.content}"

    def _tokenize(self, text: str) -> List[str]:
        # Split on whitespace and non-alphanumeric except underscores and dots
        tokens = re.findall(r"[A-Za-z0-9_\.]+", text.lower())
        return tokens

    def search(self, query: str, identifiers: List[str] = None) -> Dict[str, float]:
        """
        Returns a dict mapping chunk_id -> keyword_score normalized to [0.0, 1.0].
        """
        if not self.bm25 or not self.chunks:
            return {}

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return {}

        # Filter out generic stop words to avoid artificial 1.0 scores on words like "at", "a", "to"
        meaningful_tokens = [t for t in query_tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]
        if not meaningful_tokens and not identifiers:
            return {c.id: 0.0 for c in self.chunks}

        search_tokens = meaningful_tokens if meaningful_tokens else [t for t in query_tokens if len(t) > 1]
        raw_scores = self.bm25.get_scores(search_tokens)
        max_score = max(raw_scores) if len(raw_scores) > 0 and max(raw_scores) > 0 else 0.0

        scores: Dict[str, float] = {}
        for idx, chunk in enumerate(self.chunks):
            norm_score = float(raw_scores[idx]) / max_score if max_score > 0 else 0.0

            # Exact identifier boost:
            # If the user query has specific identifiers like 'dxDataGrid' or 'gridConfig'
            # and they appear verbatim in the chunk, add deterministic priority
            if identifiers:
                chunk_lower = chunk.content.lower()
                section_lower = f"{chunk.section} {chunk.parent_section}".lower()
                meta_ids = [i.lower() for i in chunk.metadata.get("extracted_identifiers", [])]
                id_matches = sum(
                    1 for ident in identifiers
                    if ident.lower() in chunk_lower or ident.lower() in section_lower or ident.lower() in meta_ids
                )
                if id_matches > 0:
                    norm_score = min(1.0, max(norm_score, 0.75) + (0.10 * id_matches))

            scores[chunk.id] = round(max(0.0, min(1.0, norm_score)), 4)

        return scores

    def find_exact_identifier_matches(self, identifiers: List[str]) -> List[DocumentChunk]:
        """Finds all chunks in the corpus containing any of the exact identifiers verbatim."""
        if not identifiers or not self.chunks:
            return []

        matched_chunks = []
        for chunk in self.chunks:
            chunk_lower = chunk.content.lower()
            section_lower = f"{chunk.section} {chunk.parent_section}".lower()
            meta_ids = [i.lower() for i in chunk.metadata.get("extracted_identifiers", [])]

            has_match = any(
                ident.lower() in chunk_lower or
                ident.lower() in section_lower or
                ident.lower() in meta_ids
                for ident in identifiers
            )
            if has_match:
                matched_chunks.append(chunk)

        return matched_chunks

