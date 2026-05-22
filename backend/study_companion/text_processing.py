from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TextChunk:
    text: str
    page_start: int
    page_end: int
    chunk_index: int


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "will", "with", "you", "your", "we", "our", "can", "may",
    "into", "than", "then", "there", "their", "these", "those", "which",
}


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", clean_text(text))
    return [piece.strip() for piece in pieces if len(piece.strip()) > 20]


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size_words: int,
    overlap_words: int,
) -> list[TextChunk]:
    words_with_pages: list[tuple[str, int]] = []
    for page_number, page_text in pages:
        for word in clean_text(page_text).split():
            words_with_pages.append((word, page_number))

    if not words_with_pages:
        return []

    chunks: list[TextChunk] = []
    start = 0
    chunk_index = 0
    step = max(1, chunk_size_words - overlap_words)

    while start < len(words_with_pages):
        window = words_with_pages[start : start + chunk_size_words]
        words = [word for word, _ in window]
        page_numbers = [page for _, page in window]
        text = " ".join(words).strip()
        if text:
            chunks.append(
                TextChunk(
                    text=text,
                    page_start=min(page_numbers),
                    page_end=max(page_numbers),
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
        start += step

    return chunks


def keywords(text: str, limit: int = 8) -> list[str]:
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text)
        if token.lower() not in STOPWORDS
    ]
    counts = Counter(tokens)
    return [word for word, _ in counts.most_common(limit)]


def infer_topic(text: str, fallback: str = "General") -> str:
    top_words = keywords(text, limit=3)
    if not top_words:
        return fallback
    return " / ".join(word.title() for word in top_words)


def flatten(items: Iterable[Iterable[str]]) -> list[str]:
    return [value for group in items for value in group]

