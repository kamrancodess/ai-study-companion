from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

from .database import analytics_rows, get_chunks, update_chunk_topic
from .text_processing import infer_topic, keywords


@dataclass
class TopicCluster:
    cluster_id: int
    label: str
    size: int
    sample: str


class TopicClusterer:
    def cluster(self, rows: list | None = None, n_clusters: int | None = None) -> tuple[list[TopicCluster], pd.DataFrame]:
        rows = rows or get_chunks()
        if not rows:
            return [], pd.DataFrame()

        texts = [row["text"] for row in rows]
        n_clusters = n_clusters or min(5, max(2, len(texts) // 3))
        n_clusters = min(n_clusters, len(texts))

        vectorizer = TfidfVectorizer(max_features=900, stop_words="english")
        matrix = vectorizer.fit_transform(texts)
        if len(texts) == 1:
            labels = np.array([0])
        else:
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(matrix)

        records = []
        clusters = []
        terms = np.array(vectorizer.get_feature_names_out())
        dense = matrix.toarray()
        for cluster_id in sorted(set(labels.tolist())):
            indices = np.where(labels == cluster_id)[0]
            centroid = dense[indices].mean(axis=0)
            label_terms = terms[np.argsort(centroid)[-3:]][::-1].tolist()
            label = " / ".join(term.title() for term in label_terms) or infer_topic(texts[indices[0]])
            sample = texts[indices[0]][:220]
            clusters.append(TopicCluster(cluster_id=int(cluster_id), label=label, size=len(indices), sample=sample))
            for index in indices:
                row = rows[index]
                update_chunk_topic(int(row["id"]), label)
                records.append(
                    {
                        "chunk_id": row["id"],
                        "document": row["document_title"],
                        "topic": label,
                        "cluster": int(cluster_id),
                        "page_start": row["page_start"],
                        "page_end": row["page_end"],
                    }
                )

        frame = pd.DataFrame.from_records(records)
        return clusters, frame

    def quality_hint(self, rows: list | None = None) -> str:
        rows = rows or get_chunks()
        if len(rows) < 4:
            return "Add more material for meaningful cluster quality measurement."
        texts = [row["text"] for row in rows]
        matrix = TfidfVectorizer(max_features=600, stop_words="english").fit_transform(texts)
        n_clusters = min(4, len(texts) - 1)
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(matrix)
        score = silhouette_score(matrix, labels)
        return f"Silhouette score: {score:.2f}. Higher means topics are more clearly separated."


class WeakTopicClassifier:
    """Predicts whether a topic is weak using quiz history.

    Features are intentionally simple and explainable for a portfolio app:
    answered count, accuracy, and recency proxy can later be expanded with embeddings.
    """

    def topic_summary(self) -> pd.DataFrame:
        rows = analytics_rows()["answers"]
        if not rows:
            return pd.DataFrame(columns=["topic", "answered", "correct", "accuracy", "weak_label"])
        frame = pd.DataFrame([dict(row) for row in rows])
        grouped = frame.groupby("topic").agg(answered=("id", "count"), correct=("is_correct", "sum")).reset_index()
        grouped["accuracy"] = grouped["correct"] / grouped["answered"]
        grouped["weak_label"] = (grouped["accuracy"] < 0.7).astype(int)
        return grouped.sort_values(["weak_label", "accuracy"], ascending=[False, True])

    def train_if_possible(self) -> tuple[Pipeline | None, str]:
        summary = self.topic_summary()
        if len(summary) < 4 or summary["weak_label"].nunique() < 2:
            return None, "Need at least four topics with mixed strong/weak outcomes to train a classifier."

        x = summary[["answered", "accuracy"]]
        y = summary["weak_label"]
        model = Pipeline([("classifier", LogisticRegression())])
        model.fit(x, y)
        return model, "Logistic regression weak-topic classifier trained on local quiz history."

    def weak_topics(self, limit: int = 5) -> list[dict[str, float | str | int]]:
        summary = self.topic_summary()
        if summary.empty:
            chunk_topics = [infer_topic(row["text"]) for row in get_chunks()[:limit]]
            return [{"topic": topic, "accuracy": 0.0, "answered": 0, "reason": "No quiz attempts yet"} for topic in chunk_topics]
        weak = summary.sort_values(["accuracy", "answered"], ascending=[True, False]).head(limit)
        return [
            {
                "topic": row["topic"],
                "accuracy": float(row["accuracy"]),
                "answered": int(row["answered"]),
                "reason": "Accuracy below target" if row["accuracy"] < 0.7 else "Needs periodic review",
            }
            for _, row in weak.iterrows()
        ]


def concept_keywords(limit: int = 25) -> list[str]:
    text = " ".join(row["text"] for row in get_chunks())
    return keywords(text, limit=limit)

