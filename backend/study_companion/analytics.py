from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .database import analytics_rows, list_documents


def metrics() -> dict[str, float | int]:
    rows = analytics_rows()
    attempts = [row for row in rows["attempts"] if row["completed_at"] or int(row["total_questions"] or 0) > 0]
    answers = rows["answers"]
    sessions = rows["sessions"]
    total_answers = len(answers)
    correct = sum(int(row["is_correct"]) for row in answers)
    return {
        "documents": len(list_documents()),
        "attempts": len(attempts),
        "answers": total_answers,
        "accuracy": correct / total_answers if total_answers else 0.0,
        "study_minutes": sum(int(row["minutes"]) for row in sessions),
    }


def topic_accuracy_frame() -> pd.DataFrame:
    rows = analytics_rows()["topics"]
    if not rows:
        return pd.DataFrame(columns=["topic", "answered", "accuracy"])
    frame = pd.DataFrame([dict(row) for row in rows])
    frame["accuracy"] = frame["accuracy"].fillna(0)
    return frame


def accuracy_chart():
    frame = topic_accuracy_frame()
    if frame.empty:
        return go.Figure()
    return px.bar(
        frame,
        x="topic",
        y="accuracy",
        color="accuracy",
        range_y=[0, 1],
        title="Topic Accuracy",
        color_continuous_scale="RdYlGn",
    )


def study_minutes_chart():
    rows = analytics_rows()["sessions"]
    if not rows:
        return go.Figure()
    frame = pd.DataFrame([dict(row) for row in rows])
    frame["studied_at"] = pd.to_datetime(frame["studied_at"]).dt.date
    grouped = frame.groupby("studied_at", as_index=False)["minutes"].sum()
    return px.line(grouped, x="studied_at", y="minutes", markers=True, title="Study Consistency")
