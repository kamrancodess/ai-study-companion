from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .text_processing import infer_topic, keywords, split_sentences


@dataclass
class GeneratedQuestion:
    topic: str
    difficulty: str
    question_type: str
    question: str
    correct_answer: str
    explanation: str
    options: list[str] | None = None

    def to_db_row(self, document_id: int | None) -> dict[str, str | int | None]:
        options = self.options or []
        padded = options + [None] * (4 - len(options))
        return {
            "document_id": document_id,
            "topic": self.topic,
            "difficulty": self.difficulty,
            "question_type": self.question_type,
            "question": self.question,
            "option_a": padded[0],
            "option_b": padded[1],
            "option_c": padded[2],
            "option_d": padded[3],
            "correct_answer": self.correct_answer,
            "explanation": self.explanation,
        }


class QuizGenerator:
    def generate(self, text: str, count: int = 6, difficulty: str = "Medium") -> list[GeneratedQuestion]:
        sentences = split_sentences(text)
        terms = keywords(text, limit=30)
        if not sentences or not terms:
            return []

        random.shuffle(sentences)
        questions: list[GeneratedQuestion] = []
        for sentence in sentences:
            if len(questions) >= count:
                break
            term = self._choose_term(sentence, terms)
            if not term:
                continue
            topic = infer_topic(sentence)
            questions.append(self._mcq(sentence, term, terms, topic, difficulty))
            if len(questions) < count:
                questions.append(self._short_answer(sentence, term, topic, difficulty))

        return questions[:count]

    def _choose_term(self, sentence: str, terms: list[str]) -> str | None:
        sentence_lower = sentence.lower()
        candidates = [term for term in terms if re.search(rf"\b{re.escape(term)}\b", sentence_lower)]
        return max(candidates, key=len) if candidates else None

    def _mcq(self, sentence: str, answer: str, terms: list[str], topic: str, difficulty: str) -> GeneratedQuestion:
        distractors = [term.title() for term in terms if term.lower() != answer.lower()]
        random.shuffle(distractors)
        options = distractors[:3] + [answer.title()]
        random.shuffle(options)
        blanked = re.sub(rf"\b{re.escape(answer)}\b", "_____", sentence, flags=re.IGNORECASE)
        return GeneratedQuestion(
            topic=topic,
            difficulty=difficulty,
            question_type="MCQ",
            question=f"Which concept best completes this statement: {blanked}",
            options=options,
            correct_answer=answer.title(),
            explanation=f"The source sentence states: {sentence}",
        )

    def _short_answer(self, sentence: str, answer: str, topic: str, difficulty: str) -> GeneratedQuestion:
        return GeneratedQuestion(
            topic=topic,
            difficulty=difficulty,
            question_type="Short Answer",
            question=f"In one line, explain the role of {answer.title()} in this topic.",
            correct_answer=answer.title(),
            explanation=f"Expected answer should connect back to: {sentence}",
        )


def evaluate_answer(selected: str, correct: str, question_type: str) -> bool:
    if question_type == "MCQ":
        return selected.strip().lower() == correct.strip().lower()
    selected_terms = set(re.findall(r"[a-zA-Z]{3,}", selected.lower()))
    correct_terms = set(re.findall(r"[a-zA-Z]{3,}", correct.lower()))
    return bool(selected_terms & correct_terms)

