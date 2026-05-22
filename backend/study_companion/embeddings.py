from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .config import EMBEDDING_MODEL_NAME
from .database import decode_vector


class EmbeddingService:
    """Local embeddings with a sentence-transformers path and a TF-IDF fallback."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._fallback_vectorizer: TfidfVectorizer | None = None

    @property
    def using_transformer(self) -> bool:
        return self._model is not None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if os.environ.get("ENABLE_SENTENCE_TRANSFORMERS", "0") != "1":
            self._model = None
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        except Exception:
            self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")
        self._load_model()
        if self._model is not None:
            vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vectors, dtype="float32")

        self._fallback_vectorizer = TfidfVectorizer(max_features=768, stop_words="english")
        matrix = self._fallback_vectorizer.fit_transform(texts)
        return matrix.astype("float32").toarray()

    def encode_query(self, text: str, corpus_texts: list[str] | None = None) -> np.ndarray:
        self._load_model()
        if self._model is not None:
            vector = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vector, dtype="float32")
        if self._fallback_vectorizer is None:
            self._fallback_vectorizer = TfidfVectorizer(max_features=768, stop_words="english")
            self._fallback_vectorizer.fit(corpus_texts or [text])
        return self._fallback_vectorizer.transform([text]).astype("float32").toarray()


@dataclass
class SearchHit:
    chunk_id: int
    document_id: int
    document_title: str
    text: str
    topic: str
    page_start: int | None
    page_end: int | None
    score: float


class VectorSearch:
    """FAISS semantic retrieval, falling back to sklearn nearest neighbors if FAISS is missing."""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    def _build_from_rows(self, rows: list) -> tuple[np.ndarray, list]:
        texts = [row["text"] for row in rows]
        decoded = [decode_vector(row["embedding"]) for row in rows]
        if decoded and all(vector is not None for vector in decoded):
            matrix = np.vstack(decoded).astype("float32")
        else:
            matrix = self.embedding_service.encode(texts)
        return matrix, rows

    def search(self, query: str, rows: list, top_k: int = 5) -> list[SearchHit]:
        if not rows:
            return []

        matrix, rows = self._build_from_rows(rows)
        if matrix.size == 0:
            return []

        query_vector = self.embedding_service.encode_query(query, [row["text"] for row in rows])
        top_k = min(top_k, len(rows))

        try:
            import faiss

            index = faiss.IndexFlatIP(matrix.shape[1])
            faiss.normalize_L2(matrix)
            faiss.normalize_L2(query_vector)
            index.add(matrix)
            scores, indices = index.search(query_vector, top_k)
            pairs = list(zip(indices[0].tolist(), scores[0].tolist()))
        except Exception:
            model = NearestNeighbors(n_neighbors=top_k, metric="cosine")
            model.fit(matrix)
            distances, indices = model.kneighbors(query_vector)
            pairs = [(int(i), float(1 - d)) for i, d in zip(indices[0], distances[0])]

        hits: list[SearchHit] = []
        for row_index, score in pairs:
            if row_index < 0:
                continue
            row = rows[row_index]
            hits.append(
                SearchHit(
                    chunk_id=int(row["id"]),
                    document_id=int(row["document_id"]),
                    document_title=row["document_title"],
                    text=row["text"],
                    topic=row["topic"] or "General",
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    score=float(score),
                )
            )
        return hits
