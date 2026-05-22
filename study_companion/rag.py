from __future__ import annotations

import os
from dataclasses import dataclass

from .config import QA_MODEL_NAME, TOP_K_RETRIEVAL
from .database import get_chunks
from .embeddings import EmbeddingService, SearchHit, VectorSearch
from .text_processing import split_sentences


@dataclass
class RagAnswer:
    answer: str
    confidence: float
    sources: list[SearchHit]


class RagEngine:
    def __init__(self, embedding_service: EmbeddingService, qa_model_name: str = QA_MODEL_NAME):
        self.searcher = VectorSearch(embedding_service)
        self.qa_model_name = qa_model_name
        self._qa_pipeline = None

    def _load_qa(self) -> None:
        if self._qa_pipeline is not None:
            return
        if os.environ.get("ENABLE_LOCAL_QA_MODEL", "0") != "1":
            self._qa_pipeline = False
            return
        try:
            from transformers import pipeline

            self._qa_pipeline = pipeline("question-answering", model=self.qa_model_name)
        except Exception:
            self._qa_pipeline = False

    def _fallback_answer(self, question: str, context: str) -> tuple[str, float]:
        sentences = split_sentences(context)
        if not sentences:
            return "I could not find enough information in the uploaded material.", 0.0
        terms = {term.lower().strip("?.!,") for term in question.split() if len(term) > 3}
        ranked = []
        for sentence in sentences:
            score = sum(1 for term in terms if term in sentence.lower())
            ranked.append((score, sentence))
        ranked.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
        selected = [sentence for score, sentence in ranked[:3] if score > 0] or [ranked[0][1]]
        return " ".join(selected), min(0.75, 0.35 + 0.1 * len(selected))

    def ask(self, question: str, document_id: int | None = None, top_k: int = TOP_K_RETRIEVAL) -> RagAnswer:
        rows = get_chunks(document_id)
        hits = self.searcher.search(question, rows, top_k=top_k)
        if not hits:
            return RagAnswer("Upload or index study material before asking questions.", 0.0, [])

        context = "\n\n".join(hit.text for hit in hits)
        self._load_qa()
        if self._qa_pipeline:
            try:
                result = self._qa_pipeline(question=question, context=context)
                return RagAnswer(result["answer"], float(result.get("score", 0)), hits)
            except Exception:
                pass

        answer, confidence = self._fallback_answer(question, context)
        return RagAnswer(answer, confidence, hits)
