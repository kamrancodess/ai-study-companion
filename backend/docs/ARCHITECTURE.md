# Architecture

## Data Flow

1. Student uploads PDFs in Streamlit.
2. `PyPDF2` extracts page text.
3. Text is cleaned and split into overlapping chunks.
4. Chunks are embedded with `sentence-transformers`.
5. SQLite stores documents, chunks, embeddings, quizzes, answers, sessions, and flashcards.
6. RAG queries rebuild a FAISS index from stored embeddings and retrieve the most relevant chunks.
7. Local HuggingFace QA answers from retrieved context, with an extractive fallback.
8. Quiz results update weak-topic analytics and recommendations.

## Database Schema

- `users`: local student profile.
- `documents`: uploaded PDF metadata and full extracted text.
- `chunks`: chunk text, pages, topic, and embedding bytes.
- `quiz_questions`: generated MCQs and short-answer questions.
- `quiz_attempts`: quiz session summary.
- `user_answers`: answer-level correctness and topic labels.
- `study_sessions`: manual study logging.
- `flashcards`: generated flashcards with spaced-repetition fields.

## RAG Pipeline

```text
PDF -> text -> chunks -> embeddings -> SQLite
question -> query embedding -> FAISS top-k chunks -> context -> local QA -> sourced answer
```

## Model Pipeline

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector database: `faiss-cpu` `IndexFlatIP`
- QA model: `deepset/minilm-uncased-squad2`
- Summarizer: `sshleifer/distilbart-cnn-12-6`
- Fallbacks: TF-IDF retrieval, extractive QA, extractive summarization

## ML Workflow

1. Cluster chunks with TF-IDF + KMeans.
2. Label clusters using top TF-IDF terms.
3. Store cluster labels as chunk topics.
4. Track quiz correctness by topic.
5. Mark weak topics when accuracy is below 70%.
6. Train logistic regression when enough topic-level outcomes exist.
7. Recommend review tasks and weekly schedules from weak topics.

## Scalability Ideas

- Replace SQLite with Postgres.
- Move PDF processing and embedding to background workers.
- Persist FAISS indexes per user or course.
- Add OCR for scanned PDFs.
- Use FastAPI as a backend service if Streamlit becomes too limiting.
- Add multi-user auth and role-based dashboards.
- Add evaluation datasets for answer faithfulness and quiz quality.
- Add model quantization or ONNX for faster CPU inference.

