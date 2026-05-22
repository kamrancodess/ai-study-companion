<<<<<<< HEAD
# AI Study Companion

AI Study Companion is a free, local-first Streamlit application for students who want to learn from their own PDFs and notes. It implements Phase 2 features: RAG question answering, AI summaries, quiz generation, weak-topic detection, study recommendations, analytics, and topic clustering without paid APIs.

## Roadmap

1. Phase 1: Upload PDFs, extract text, chunk notes, store documents in SQLite.
2. Phase 2: Add FAISS retrieval, local RAG QA, summaries, quizzes, weak-topic analytics, recommendations, topic clustering, flashcards.
3. Phase 3: Multi-user auth, better note editors, OCR, citation exports, background indexing, richer ML models.
4. Phase 4: SaaS-ready architecture with async workers, Postgres, object storage, observability, and optional GPU hosting.

## Folder Structure

```text
ai_study_companion/
  app.py                         Streamlit frontend
  requirements.txt               Free/open-source Python packages
  README.md                      Project guide
  docs/
    ARCHITECTURE.md              System design, data flow, ML workflow
    DEPLOYMENT.md                Free deployment notes
  study_companion/
    config.py                    Paths, model names, chunk settings
    database.py                  SQLite schema and data operations
    pdf_processing.py            PyPDF2 extraction
    text_processing.py           Cleaning, chunking, keyword/topic helpers
    embeddings.py                sentence-transformers plus FAISS retrieval
    rag.py                       Retrieval augmented question answering
    summarizer.py                Local summarization and extractive fallback
    quiz.py                      MCQ and short-answer generation
    ml_models.py                 Weak-topic classifier and KMeans clustering
    recommender.py               Study plan and personalized actions
    analytics.py                 Metrics and Plotly charts
    flashcards.py                Spaced repetition flashcard helpers
    ui.py                        Shared Streamlit styling and frontend components
  data/                          Local generated database and uploads
```

## Setup

```bash
cd ai_study_companion
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first run may download HuggingFace models. That is still free, but it needs disk space and internet access once. After download, models are cached locally.

## Required Packages

- `streamlit`: frontend
- `PyPDF2`: PDF extraction
- `sentence-transformers`: local embeddings
- `transformers` and `torch`: local QA and summarization models
- `faiss-cpu`: vector database
- `scikit-learn`: clustering, classification, TF-IDF fallback
- `xgboost`: optional future weak-topic classifier
- `sqlite3`: built into Python for local storage
- `plotly`, `matplotlib`, `pandas`, `numpy`: analytics and visualization

## Beginner-Friendly Module Explanation

- `pdf_processing.py`: reads PDFs and extracts text page by page.
- `text_processing.py`: cleans text, splits it into chunks, and finds keywords.
- `embeddings.py`: converts text chunks into numeric vectors and searches similar chunks.
- `rag.py`: finds the best chunks for a question and generates an answer from those chunks.
- `summarizer.py`: summarizes notes using a local model or extractive scoring.
- `quiz.py`: creates MCQs and short-answer questions from key terms.
- `database.py`: saves documents, chunks, questions, answers, study sessions, and flashcards.
- `ml_models.py`: detects weak topics from quiz accuracy and clusters related concepts.
- `recommender.py`: converts weak topics into study tasks and a weekly plan.
- `analytics.py`: prepares dashboard metrics and charts.
- `flashcards.py`: creates simple spaced-repetition cards.
- `ui.py`: provides the reusable frontend shell, cards, metrics, empty states, and styling.
- `app.py`: connects all backend modules into the Streamlit UI.

## Where ML Is Used

- Embeddings: sentence-transformers convert notes and questions into semantic vectors.
- Retrieval: FAISS finds the most relevant chunks for RAG.
- Summarization and QA: HuggingFace transformer models run locally.
- Classification: logistic regression predicts weak topics from quiz performance once enough quiz history exists.
- Clustering: KMeans groups chunks into concept clusters.
- Recommendation: rule-based logic uses weak-topic predictions, quiz accuracy, and study sessions.

## Data Needed

- PDFs and notes for document chunks.
- Quiz attempts and answer correctness for weak-topic detection.
- Study sessions for consistency analytics.
- Flashcard review history for spaced repetition expansion.

## UI Plan

- Dashboard: accuracy, attempts, study minutes, weak topics, charts.
- Upload: PDF upload, extraction, chunking, embedding, flashcard creation.
- Ask: RAG question answering with page-level sources.
- Summaries: whole-document and chapter-wise summaries.
- Quiz: MCQ and short-answer practice with difficulty selection.
- Topics: clustering and semantic topic search.
- Study Plan: recommendations, weekly plan, due flashcards.
- Architecture: simple explanation of the AI workflow.

## Deployment Workflow

Local-first is the main target. For free deployment, push this folder to GitHub and deploy on Streamlit Community Cloud. Use a small model and commit only code, not local `data/`, model caches, or uploaded PDFs.
=======
# ai-study-companion
AI-powered Study Companion using RAG, ML, NLP, and recommendation systems.
>>>>>>> 2130fc51aeb6934a4d7e215362bdab781fdc00c9
