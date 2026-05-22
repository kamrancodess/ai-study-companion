from __future__ import annotations

import base64
import random
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from study_companion import analytics, database
from study_companion.config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS, DEFAULT_USER_ID, TOP_K_RETRIEVAL, UPLOAD_DIR, ensure_directories
from study_companion.demo_data import seed_demo_data
from study_companion.embeddings import EmbeddingService
from study_companion.flashcards import generate_flashcards
from study_companion.ml_models import TopicClusterer, WeakTopicClassifier
from study_companion.pdf_processing import pdf_to_chunks
from study_companion.quiz import QuizGenerator, evaluate_answer
from study_companion.rag import RagEngine
from study_companion.recommender import StudyRecommender
from study_companion.summarizer import Summarizer


app = FastAPI(title="AI Study Companion API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_service = EmbeddingService()
rag_engine = RagEngine(embedding_service)
summarizer = Summarizer()


class UploadRequest(BaseModel):
    filename: str
    content_base64: str


class AskRequest(BaseModel):
    question: str = Field(min_length=2)
    document_id: int | None = None


class SummarizeRequest(BaseModel):
    document_id: int
    mode: str = "document"


class QuizRequest(BaseModel):
    document_id: int
    count: int = 6
    difficulty: str = "Medium"


class AnswerRequest(BaseModel):
    question_id: int
    selected_answer: str


class SubmitQuizRequest(BaseModel):
    attempt_id: int | None = None
    answers: list[AnswerRequest]


class StudySessionRequest(BaseModel):
    topic: str
    minutes: int
    notes: str = ""


@app.on_event("startup")
def startup() -> None:
    ensure_directories()
    database.init_db()
    seed_demo_data()


def row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/documents")
def documents() -> dict[str, Any]:
    return {"documents": [row_to_dict(row) for row in database.list_documents()]}


@app.post("/documents/upload")
def upload_document(payload: UploadRequest) -> dict[str, Any]:
    ensure_directories()
    if not payload.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        file_bytes = base64.b64decode(payload.content_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 file content.") from exc

    safe_name = Path(payload.filename).name
    pdf_path = UPLOAD_DIR / safe_name
    pdf_path.write_bytes(file_bytes)

    try:
        full_text, chunks = pdf_to_chunks(pdf_path, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not extract text from this PDF: {exc}") from exc

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No readable text found. If this is a scanned PDF, export it with OCR/selectable text or install Tesseract OCR.",
        )

    document_id = database.create_document(safe_name, Path(safe_name).stem, full_text, chunks[-1].page_end)
    embeddings = embedding_service.encode([chunk.text for chunk in chunks])
    database.add_chunks(document_id, chunks, embeddings)
    cards = generate_flashcards(full_text, document_id=document_id, count=8)
    database.save_flashcards(cards)

    return {
        "document_id": document_id,
        "filename": safe_name,
        "chunks": len(chunks),
        "flashcards": len(cards),
        "pages": chunks[-1].page_end,
    }


@app.get("/dashboard")
def dashboard() -> dict[str, Any]:
    topic_frame = analytics.topic_accuracy_frame()
    weak_topics = WeakTopicClassifier().weak_topics(limit=5)
    return {
        "metrics": analytics.metrics(),
        "topic_accuracy": topic_frame.to_dict(orient="records"),
        "weak_topics": weak_topics,
        "recommendations": StudyRecommender().recommendations(),
        "documents": [row_to_dict(row) for row in database.list_documents()],
    }


@app.post("/ask")
def ask(payload: AskRequest) -> dict[str, Any]:
    result = rag_engine.ask(payload.question, document_id=payload.document_id, top_k=TOP_K_RETRIEVAL)
    return {
        "answer": result.answer,
        "confidence": result.confidence,
        "sources": [hit.__dict__ for hit in result.sources],
    }


@app.post("/summarize")
def summarize(payload: SummarizeRequest) -> dict[str, Any]:
    document = database.get_document(payload.document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    if payload.mode == "chunks":
        chunks = database.get_chunks(payload.document_id)
        return {"items": summarizer.chapter_summaries(chunks)}

    return {"summary": summarizer.summarize(document["text"])}


@app.post("/quiz/generate")
def generate_quiz(payload: QuizRequest) -> dict[str, Any]:
    document = database.get_document(payload.document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    question_bank = database.list_questions(payload.document_id, limit=200)
    difficulty_matches = [row for row in question_bank if row["difficulty"].lower() == payload.difficulty.lower()]
    reusable_questions = difficulty_matches or question_bank
    if len(reusable_questions) >= payload.count:
        selected_rows = random.sample(reusable_questions, k=min(payload.count, len(reusable_questions)))
        return {
            "attempt_id": None,
            "questions": [
                {
                    "id": row["id"],
                    "topic": row["topic"],
                    "difficulty": row["difficulty"],
                    "question_type": row["question_type"],
                    "question": row["question"],
                    "options": [value for value in [row["option_a"], row["option_b"], row["option_c"], row["option_d"]] if value],
                    "correct_answer": row["correct_answer"],
                    "explanation": row["explanation"],
                }
                for row in selected_rows
            ],
        }

    generated = QuizGenerator().generate(document["text"], count=payload.count, difficulty=payload.difficulty)
    ids = database.save_questions([question.to_db_row(payload.document_id) for question in generated])
    questions = []
    for question_id, question in zip(ids, generated):
        questions.append(
            {
                "id": question_id,
                "topic": question.topic,
                "difficulty": question.difficulty,
                "question_type": question.question_type,
                "question": question.question,
                "options": question.options or [],
                "correct_answer": question.correct_answer,
                "explanation": question.explanation,
            }
        )
    return {"attempt_id": None, "questions": questions}


@app.post("/quiz/submit")
def submit_quiz(payload: SubmitQuizRequest) -> dict[str, Any]:
    attempt_id = payload.attempt_id or database.start_attempt(DEFAULT_USER_ID)
    results = []
    correct_count = 0
    for answer in payload.answers:
        row = next((item for item in database.list_questions(limit=200) if item["id"] == answer.question_id), None)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Question {answer.question_id} not found.")
        is_correct = evaluate_answer(answer.selected_answer, row["correct_answer"], row["question_type"])
        correct_count += int(is_correct)
        database.record_answer(attempt_id, answer.question_id, answer.selected_answer, is_correct, row["topic"])
        results.append(
            {
                "question_id": answer.question_id,
                "is_correct": is_correct,
                "correct_answer": row["correct_answer"],
                "explanation": row["explanation"],
                "topic": row["topic"],
            }
        )
    database.finish_attempt(attempt_id)
    total = len(payload.answers)
    return {"score": correct_count, "total": total, "accuracy": correct_count / total if total else 0, "results": results}


@app.post("/topics/cluster")
def cluster_topics() -> dict[str, Any]:
    clusters, frame = TopicClusterer().cluster(database.get_chunks())
    return {
        "clusters": [cluster.__dict__ for cluster in clusters],
        "chunks": frame.to_dict(orient="records") if not frame.empty else [],
        "quality": TopicClusterer().quality_hint(database.get_chunks()),
    }


@app.get("/recommendations")
def recommendations() -> dict[str, Any]:
    recommender = StudyRecommender()
    return {
        "recommendations": recommender.recommendations(),
        "weekly_plan": recommender.weekly_plan(),
        "flashcards": [row_to_dict(row) for row in database.due_flashcards(limit=12)],
    }


@app.post("/study-sessions")
def log_session(payload: StudySessionRequest) -> dict[str, str]:
    database.log_study_session(DEFAULT_USER_ID, payload.topic or "General Review", payload.minutes, payload.notes)
    return {"status": "saved"}
