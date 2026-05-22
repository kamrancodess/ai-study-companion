from __future__ import annotations

from .config import SUMMARIZER_MODEL_NAME
from .text_processing import keywords, split_sentences


class Summarizer:
    def __init__(self, model_name: str = SUMMARIZER_MODEL_NAME):
        self.model_name = model_name
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline

            self._pipeline = pipeline("summarization", model=self.model_name)
        except Exception:
            self._pipeline = False

    def summarize(self, text: str, max_words: int = 180) -> str:
        text = text.strip()
        if not text:
            return "No text available to summarize."

        self._load()
        if self._pipeline and len(text.split()) > 80:
            try:
                clipped = " ".join(text.split()[:900])
                result = self._pipeline(clipped, max_length=150, min_length=45, do_sample=False)
                return result[0]["summary_text"].strip()
            except Exception:
                pass

        return self._extractive_summary(text, max_words=max_words)

    def chapter_summaries(self, chunks: list, max_items: int = 8) -> list[dict[str, str]]:
        summaries = []
        for chunk in chunks[:max_items]:
            topic = chunk["topic"] or f"Pages {chunk['page_start']}-{chunk['page_end']}"
            summaries.append({"topic": topic, "summary": self.summarize(chunk["text"], max_words=90)})
        return summaries

    def _extractive_summary(self, text: str, max_words: int) -> str:
        sentences = split_sentences(text)
        if not sentences:
            return text[:600]
        top_terms = set(keywords(text, limit=16))
        scored = []
        for index, sentence in enumerate(sentences):
            score = sum(1 for term in top_terms if term in sentence.lower())
            scored.append((score, -index, sentence))
        selected = [item[2] for item in sorted(scored, reverse=True)[:5]]
        selected.sort(key=lambda sentence: sentences.index(sentence))
        words = " ".join(selected).split()
        return " ".join(words[:max_words])

