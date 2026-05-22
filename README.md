# AI Study Companion

Free, local-first AI-powered Study Companion with a FastAPI backend and a Next.js frontend. It supports PDF ingestion, OCR fallback, RAG question answering, summaries, quiz generation, weak-topic detection, recommendations, analytics, flashcards, topic clustering, and seeded demo data.

## Features

- FastAPI backend for the Next.js frontend
- PDF text extraction with PyPDF2 and PyMuPDF
- OCR fallback for scanned PDFs through local Tesseract
- SQLite persistence for documents, chunks, quizzes, answers, study sessions, and flashcards
- Local RAG retrieval using TF-IDF by default, with optional sentence-transformers
- Local demo dataset: `ML Unit 1-5 Demo Notes`
- 120-question demo quiz bank
- Weak-topic analytics and recommendation logic
- Topic clustering with scikit-learn
- Streamlit legacy app still available in `app.py`

## Run Locally

Backend:

```powershell
cd C:\Users\ibrah\Downloads\b_HUDWK5qTByF\ai_study_companion
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd C:\Users\ibrah\Downloads\b_HUDWK5qTByF\ai_study_companion\frontend
npm install
npm run dev
```

App URL:

```text
http://127.0.0.1:3000
```

Backend URL:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

## Install

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For scanned PDFs, install Tesseract OCR locally. On Windows, the app automatically checks:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

## API Endpoints

- `GET /health`
- `GET /documents`
- `POST /documents/upload`
- `GET /dashboard`
- `POST /ask`
- `POST /summarize`
- `POST /quiz/generate`
- `POST /quiz/submit`
- `POST /topics/cluster`
- `GET /recommendations`
- `POST /study-sessions`

## Backend Structure

```text
ai_study_companion/
  api_server.py                  FastAPI app and REST endpoints
  app.py                         Legacy Streamlit app
  frontend/                      Next.js frontend
    app/                         App router pages and global CSS
    components/                  Dashboard, sections, and UI components
    lib/                         Frontend API client and utilities
    public/                      Static assets
    package.json                 Frontend dependencies and scripts
  requirements.txt               Python dependencies
  docs/
    ARCHITECTURE.md              Architecture and ML workflow
    DEPLOYMENT.md                Free deployment notes
  study_companion/
    analytics.py                 Dashboard metrics and charts
    config.py                    Paths, model names, runtime settings
    database.py                  SQLite schema and data access
    demo_data.py                 ML Unit 1-5 demo seed data
    embeddings.py                Retrieval vectors and search
    flashcards.py                Flashcard generation helpers
    ml_models.py                 Weak-topic classifier and clustering
    pdf_processing.py            PDF extraction and OCR fallback
    quiz.py                      MCQ and short-answer generation
    rag.py                       Retrieval-augmented answering
    recommender.py               Recommendations and weekly plans
    summarizer.py                Local summarization fallback
    text_processing.py           Cleaning, chunking, keywords
    ui.py                        Legacy Streamlit UI helpers
```

## Notes

This project uses only free and open-source tools. No OpenAI API or paid cloud API is required. Local runtime data such as SQLite databases, uploaded PDFs, model caches, and Python bytecode are ignored by Git.
