from __future__ import annotations

from datetime import date, timedelta

from .ml_models import WeakTopicClassifier, concept_keywords


class StudyRecommender:
    def __init__(self):
        self.classifier = WeakTopicClassifier()

    def recommendations(self) -> list[dict[str, str]]:
        weak_topics = self.classifier.weak_topics(limit=5)
        if not weak_topics:
            return [{"topic": "Upload notes", "action": "Add a PDF and generate your first quiz.", "priority": "High"}]

        recs = []
        for item in weak_topics:
            accuracy = float(item.get("accuracy", 0))
            priority = "High" if accuracy < 0.6 else "Medium"
            recs.append(
                {
                    "topic": str(item["topic"]),
                    "action": "Review summaries, answer 5 practice questions, then revisit missed flashcards.",
                    "priority": priority,
                }
            )
        return recs

    def weekly_plan(self, minutes_per_day: int = 45) -> list[dict[str, str | int]]:
        recs = self.recommendations()
        keywords = concept_keywords(limit=10)
        plan = []
        today = date.today()
        for offset in range(7):
            topic = recs[offset % len(recs)]["topic"] if recs else (keywords[offset % len(keywords)] if keywords else "General Review")
            plan.append(
                {
                    "date": (today + timedelta(days=offset)).isoformat(),
                    "topic": topic,
                    "minutes": minutes_per_day,
                    "task": "Read summary, active recall, quiz, and flashcard review",
                }
            )
        return plan

