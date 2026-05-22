from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .config import DB_PATH, ensure_directories
from .text_processing import infer_topic


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def encode_vector(vector: np.ndarray | None) -> bytes | None:
    if vector is None:
        return None
    return np.asarray(vector, dtype="float32").tobytes()


def decode_vector(blob: bytes | None) -> np.ndarray | None:
    if blob is None:
        return None
    return np.frombuffer(blob, dtype="float32")


@contextmanager
def connect(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    ensure_directories()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            PRAGMA journal_mode = WAL;

            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                page_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                page_start INTEGER,
                page_end INTEGER,
                topic TEXT,
                embedding BLOB,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS quiz_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                topic TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                question_type TEXT NOT NULL,
                question TEXT NOT NULL,
                option_a TEXT,
                option_b TEXT,
                option_c TEXT,
                option_d TEXT,
                correct_answer TEXT NOT NULL,
                explanation TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                score REAL DEFAULT 0,
                total_questions INTEGER DEFAULT 0,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS user_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id INTEGER NOT NULL,
                question_id INTEGER NOT NULL,
                selected_answer TEXT NOT NULL,
                is_correct INTEGER NOT NULL,
                topic TEXT NOT NULL,
                answered_at TEXT NOT NULL,
                FOREIGN KEY(attempt_id) REFERENCES quiz_attempts(id) ON DELETE CASCADE,
                FOREIGN KEY(question_id) REFERENCES quiz_questions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS study_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                topic TEXT NOT NULL,
                minutes INTEGER NOT NULL,
                notes TEXT,
                studied_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS flashcards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                topic TEXT NOT NULL,
                front TEXT NOT NULL,
                back TEXT NOT NULL,
                ease REAL DEFAULT 2.5,
                interval_days INTEGER DEFAULT 1,
                repetitions INTEGER DEFAULT 0,
                due_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE SET NULL
            );
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO users(id, name, created_at) VALUES (?, ?, ?)",
            (1, "Local Student", utc_now()),
        )


def create_document(filename: str, title: str, text: str, page_count: int) -> int:
    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO documents(filename, title, text, page_count, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (filename, title, text, page_count, utc_now()),
        )
        return int(cursor.lastrowid)


def add_chunks(document_id: int, chunks: list[Any], embeddings: np.ndarray | None) -> None:
    with connect() as conn:
        rows = []
        for index, chunk in enumerate(chunks):
            vector = embeddings[index] if embeddings is not None and len(embeddings) > index else None
            topic = getattr(chunk, "topic", None) or infer_topic(chunk.text)
            rows.append(
                (
                    document_id,
                    chunk.chunk_index,
                    chunk.text,
                    chunk.page_start,
                    chunk.page_end,
                    topic,
                    encode_vector(vector),
                )
            )
        conn.executemany(
            """
            INSERT INTO chunks(document_id, chunk_index, text, page_start, page_end, topic, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def update_chunk_topic(chunk_id: int, topic: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE chunks SET topic = ? WHERE id = ?", (topic, chunk_id))


def list_documents() -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            """
            SELECT id, filename, title, page_count, created_at
            FROM documents
            ORDER BY created_at DESC
            """
        ).fetchall()


def get_document(document_id: int) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()


def get_chunks(document_id: int | None = None) -> list[sqlite3.Row]:
    query = "SELECT chunks.*, documents.title AS document_title FROM chunks JOIN documents ON documents.id = chunks.document_id"
    params: tuple[Any, ...] = ()
    if document_id:
        query += " WHERE document_id = ?"
        params = (document_id,)
    query += " ORDER BY document_id, chunk_index"
    with connect() as conn:
        return conn.execute(query, params).fetchall()


def save_questions(questions: list[dict[str, Any]]) -> list[int]:
    ids: list[int] = []
    with connect() as conn:
        for item in questions:
            cursor = conn.execute(
                """
                INSERT INTO quiz_questions(
                    document_id, topic, difficulty, question_type, question,
                    option_a, option_b, option_c, option_d, correct_answer,
                    explanation, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.get("document_id"),
                    item["topic"],
                    item["difficulty"],
                    item["question_type"],
                    item["question"],
                    item.get("option_a"),
                    item.get("option_b"),
                    item.get("option_c"),
                    item.get("option_d"),
                    item["correct_answer"],
                    item.get("explanation"),
                    utc_now(),
                ),
            )
            ids.append(int(cursor.lastrowid))
    return ids


def list_questions(document_id: int | None = None, limit: int = 20) -> list[sqlite3.Row]:
    query = "SELECT * FROM quiz_questions"
    params: list[Any] = []
    if document_id:
        query += " WHERE document_id = ?"
        params.append(document_id)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with connect() as conn:
        return conn.execute(query, tuple(params)).fetchall()


def start_attempt(user_id: int = 1) -> int:
    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO quiz_attempts(user_id, started_at) VALUES (?, ?)",
            (user_id, utc_now()),
        )
        return int(cursor.lastrowid)


def record_answer(attempt_id: int, question_id: int, selected_answer: str, is_correct: bool, topic: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO user_answers(attempt_id, question_id, selected_answer, is_correct, topic, answered_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (attempt_id, question_id, selected_answer, int(is_correct), topic, utc_now()),
        )


def finish_attempt(attempt_id: int) -> None:
    with connect() as conn:
        stats = conn.execute(
            "SELECT COUNT(*) AS total, SUM(is_correct) AS correct FROM user_answers WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        total = int(stats["total"] or 0)
        correct = int(stats["correct"] or 0)
        score = correct / total if total else 0
        conn.execute(
            "UPDATE quiz_attempts SET completed_at = ?, score = ?, total_questions = ? WHERE id = ?",
            (utc_now(), score, total, attempt_id),
        )


def analytics_rows() -> dict[str, list[sqlite3.Row]]:
    with connect() as conn:
        return {
            "attempts": conn.execute("SELECT * FROM quiz_attempts ORDER BY started_at DESC").fetchall(),
            "answers": conn.execute("SELECT * FROM user_answers ORDER BY answered_at DESC").fetchall(),
            "sessions": conn.execute("SELECT * FROM study_sessions ORDER BY studied_at DESC").fetchall(),
            "topics": conn.execute(
                """
                SELECT topic, COUNT(*) AS answered, AVG(is_correct) AS accuracy
                FROM user_answers GROUP BY topic ORDER BY accuracy ASC, answered DESC
                """
            ).fetchall(),
        }


def log_study_session(user_id: int, topic: str, minutes: int, notes: str = "") -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO study_sessions(user_id, topic, minutes, notes, studied_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, topic, minutes, notes, utc_now()),
        )


def save_flashcards(cards: list[dict[str, Any]]) -> list[int]:
    ids: list[int] = []
    with connect() as conn:
        for card in cards:
            cursor = conn.execute(
                """
                INSERT INTO flashcards(document_id, topic, front, back, due_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    card.get("document_id"),
                    card["topic"],
                    card["front"],
                    card["back"],
                    card.get("due_at", utc_now()),
                    utc_now(),
                ),
            )
            ids.append(int(cursor.lastrowid))
    return ids


def due_flashcards(limit: int = 20) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM flashcards WHERE due_at <= ? ORDER BY due_at ASC LIMIT ?",
            (utc_now(), limit),
        ).fetchall()
