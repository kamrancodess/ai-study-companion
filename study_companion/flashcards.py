from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .text_processing import infer_topic, keywords, split_sentences


def generate_flashcards(text: str, document_id: int | None = None, count: int = 8) -> list[dict[str, str | int | None]]:
    cards = []
    sentences = split_sentences(text)
    for sentence in sentences:
        if len(cards) >= count:
            break
        terms = keywords(sentence, limit=2)
        if not terms:
            continue
        term = terms[0].title()
        cards.append(
            {
                "document_id": document_id,
                "topic": infer_topic(sentence),
                "front": f"What should you remember about {term}?",
                "back": sentence,
                "due_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
    return cards


def next_due_date(quality: int, repetitions: int, interval_days: int, ease: float) -> tuple[str, int, int, float]:
    if quality < 3:
        repetitions = 0
        interval_days = 1
    else:
        repetitions += 1
        if repetitions == 1:
            interval_days = 1
        elif repetitions == 2:
            interval_days = 3
        else:
            interval_days = max(1, round(interval_days * ease))
        ease = max(1.3, ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
    due_at = datetime.now(timezone.utc) + timedelta(days=interval_days)
    return due_at.isoformat(timespec="seconds"), repetitions, interval_days, ease

