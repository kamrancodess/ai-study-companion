from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from study_companion import analytics, database
from study_companion.config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    DEFAULT_USER_ID,
    TOP_K_RETRIEVAL,
    UPLOAD_DIR,
    ensure_directories,
)
from study_companion.embeddings import EmbeddingService
from study_companion.flashcards import generate_flashcards
from study_companion.ml_models import TopicClusterer, WeakTopicClassifier
from study_companion.pdf_processing import pdf_to_chunks
from study_companion.quiz import QuizGenerator, evaluate_answer
from study_companion.rag import RagEngine
from study_companion.recommender import StudyRecommender
from study_companion.summarizer import Summarizer
from study_companion.ui import apply_theme, empty_state, hero, metric_strip, pill, sidebar_brand, source_box, surface


st.set_page_config(page_title="AI Study Companion", page_icon="AI", layout="wide")


@st.cache_resource(show_spinner=False)
def embedding_service() -> EmbeddingService:
    return EmbeddingService()


@st.cache_resource(show_spinner=False)
def rag_engine() -> RagEngine:
    return RagEngine(embedding_service())


@st.cache_resource(show_spinner=False)
def summarizer() -> Summarizer:
    return Summarizer()


def bootstrap() -> None:
    ensure_directories()
    database.init_db()
    st.session_state.setdefault("current_quiz", [])
    st.session_state.setdefault("attempt_id", None)


def document_selector(label: str = "Choose document") -> int | None:
    documents = database.list_documents()
    if not documents:
        empty_state("Upload a PDF first to unlock this workspace.")
        return None
    options = {f"{row['title']} ({row['page_count']} pages)": row["id"] for row in documents}
    return options[st.selectbox(label, list(options.keys()))]


def upload_view() -> None:
    hero(
        "Build Your Study Library",
        "Upload PDFs once. The app extracts text, creates searchable chunks, builds embeddings, and starts a flashcard memory for revision.",
    )
    left, right = st.columns([1.1, 0.9])

    with left:
        st.subheader("Upload PDFs")
        uploaded_files = st.file_uploader("PDF notes, chapters, or slides", type=["pdf"], accept_multiple_files=True)

        if uploaded_files:
            st.caption(f"{len(uploaded_files)} file(s) ready for processing.")

        process_clicked = uploaded_files and st.button("Process PDFs", type="primary", use_container_width=True)

    with right:
        surface("Local-first", "Files stay on this machine. SQLite stores metadata and FAISS-ready vectors power semantic retrieval.", "good")
        surface("Pipeline", "PDF extraction -> chunking -> embeddings -> topics -> flashcards.", "")

    if process_clicked:
        for uploaded in uploaded_files:
            pdf_path = UPLOAD_DIR / uploaded.name
            pdf_path.write_bytes(uploaded.getbuffer())
            with st.spinner(f"Extracting and embedding {uploaded.name}"):
                full_text, chunks = pdf_to_chunks(pdf_path, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
                if not chunks:
                    st.warning(f"No readable text found in {uploaded.name}.")
                    continue
                document_id = database.create_document(uploaded.name, Path(uploaded.name).stem, full_text, chunks[-1].page_end)
                embeddings = embedding_service().encode([chunk.text for chunk in chunks])
                database.add_chunks(document_id, chunks, embeddings)
                cards = generate_flashcards(full_text, document_id=document_id, count=8)
                database.save_flashcards(cards)
                st.success(f"Indexed {uploaded.name}: {len(chunks)} chunks and {len(cards)} flashcards.")

    st.subheader("Indexed Documents")
    docs = database.list_documents()
    if docs:
        frame = pd.DataFrame([dict(row) for row in docs])
        st.dataframe(frame[["title", "filename", "page_count", "created_at"]], use_container_width=True, hide_index=True)
    else:
        empty_state("No documents indexed yet. Add a readable PDF to begin.")


def ask_view() -> None:
    hero(
        "Ask Your Notes",
        "The RAG engine retrieves the most relevant chunks, answers from local context, and shows the sources it used.",
    )
    document_id = document_selector("Search scope")
    question = st.text_input("Question", placeholder="Example: What is supervised learning?")

    if st.button("Ask AI", type="primary", disabled=not question, use_container_width=True):
        with st.spinner("Retrieving chunks and generating answer"):
            result = rag_engine().ask(question, document_id=document_id, top_k=TOP_K_RETRIEVAL)
        st.markdown("### Answer")
        surface("Confidence", f"{result.confidence:.2f}", "good" if result.confidence >= 0.5 else "medium")
        st.write(result.answer)
        with st.expander("Sources used"):
            for hit in result.sources:
                source_box(
                    f"{hit.document_title}, pages {hit.page_start}-{hit.page_end} | score {hit.score:.2f} | {hit.topic}",
                    hit.text[:900],
                )


def summaries_view() -> None:
    hero(
        "Summarize Study Material",
        "Create concise revision summaries for an entire document or skim chunk-level explanations topic by topic.",
    )
    document_id = document_selector()
    if not document_id:
        return
    document = database.get_document(document_id)
    chunks = database.get_chunks(document_id)

    col1, col2 = st.columns([1, 1])
    with col1:
        surface("Whole document", "Best for quick chapter revision before a quiz.", "good")
        if st.button("Generate Whole-Document Summary", type="primary", use_container_width=True):
            with st.spinner("Summarizing locally"):
                st.markdown("#### Summary")
                st.write(summarizer().summarize(document["text"]))
    with col2:
        surface("Chunk summaries", "Best for scanning topic groups and page-level concepts.", "")
        if st.button("Generate Chunk Summaries", use_container_width=True):
            with st.spinner("Building chunk summaries"):
                for item in summarizer().chapter_summaries(chunks):
                    st.markdown(f"**{item['topic']}**")
                    st.write(item["summary"])


def quiz_view() -> None:
    hero(
        "Practice With Generated Quizzes",
        "Generate MCQs and short-answer questions from uploaded notes. Your answers feed weak-topic detection and recommendations.",
    )
    document_id = document_selector()
    if not document_id:
        return

    document = database.get_document(document_id)
    controls = st.columns([1, 1, 1])
    with controls[0]:
        difficulty = st.radio("Difficulty", ["Easy", "Medium", "Hard"], index=1, horizontal=True)
    with controls[1]:
        count = st.slider("Questions", min_value=3, max_value=12, value=6)
    with controls[2]:
        st.caption("Quiz type")
        st.markdown(pill("MCQ + Short Answer", "good"), unsafe_allow_html=True)

    if st.button("Generate Quiz", type="primary", use_container_width=True):
        questions = QuizGenerator().generate(document["text"], count=count, difficulty=difficulty)
        ids = database.save_questions([question.to_db_row(document_id) for question in questions])
        st.session_state.current_quiz = list(zip(ids, questions))
        st.session_state.attempt_id = database.start_attempt(DEFAULT_USER_ID)

    if st.session_state.current_quiz:
        with st.form("quiz_form"):
            answers = {}
            for question_id, question in st.session_state.current_quiz:
                st.markdown(f"#### {question.question_type}: {question.question}")
                if question.question_type == "MCQ" and question.options:
                    answers[question_id] = st.radio("Choose one", question.options, key=f"q_{question_id}")
                else:
                    answers[question_id] = st.text_input("Your short answer", key=f"q_{question_id}")
            submitted = st.form_submit_button("Submit Quiz", type="primary")

        if submitted:
            correct_count = 0
            for question_id, question in st.session_state.current_quiz:
                selected = answers.get(question_id, "")
                is_correct = evaluate_answer(selected, question.correct_answer, question.question_type)
                correct_count += int(is_correct)
                database.record_answer(st.session_state.attempt_id, question_id, selected, is_correct, question.topic)
            database.finish_attempt(st.session_state.attempt_id)
            st.success(f"Score: {correct_count}/{len(st.session_state.current_quiz)}")
            for _, question in st.session_state.current_quiz:
                st.markdown(f"**Answer:** {question.correct_answer}")
                st.caption(question.explanation)
            st.session_state.current_quiz = []
            st.session_state.attempt_id = None


def dashboard_view() -> None:
    hero(
        "Learning Command Center",
        "Track documents, quiz accuracy, weak topics, study consistency, and the next actions your revision system recommends.",
    )
    values = analytics.metrics()
    metric_strip(
        [
            ("Documents", str(values["documents"])),
            ("Quiz Attempts", str(values["attempts"])),
            ("Answers", str(values["answers"])),
            ("Accuracy", f"{values['accuracy']:.0%}"),
            ("Study Minutes", str(values["study_minutes"])),
        ]
    )

    chart_col, plan_col = st.columns([1.2, 0.8])
    with chart_col:
        st.subheader("Analytics")
        if not analytics.topic_accuracy_frame().empty:
            st.plotly_chart(analytics.accuracy_chart(), use_container_width=True, key="topic_accuracy_chart")
        else:
            empty_state("Quiz answers will appear here as topic accuracy charts.")
        if values["study_minutes"]:
            st.plotly_chart(analytics.study_minutes_chart(), use_container_width=True, key="study_minutes_chart")
        else:
            empty_state("Log study sessions to see study consistency over time.")
    with plan_col:
        st.subheader("Weak Topics")
        weak_topics = WeakTopicClassifier().weak_topics()
        if weak_topics:
            for item in weak_topics:
                tone = "high" if float(item["accuracy"]) < 0.6 else "medium"
                surface(str(item["topic"]), f"Accuracy {float(item['accuracy']):.0%}. {item['reason']}.", tone)
        else:
            empty_state("Take a quiz to generate weak-topic analytics.")

        st.subheader("Log Study Session")
        with st.form("study_session"):
            topic = st.text_input("Topic")
            minutes = st.number_input("Minutes", min_value=5, max_value=240, value=45, step=5)
            notes = st.text_area("Notes")
            if st.form_submit_button("Save Session"):
                database.log_study_session(DEFAULT_USER_ID, topic or "General Review", int(minutes), notes)
                st.success("Study session saved.")


def topics_view() -> None:
    hero(
        "Explore Topic Clusters",
        "Group related chunks with KMeans, label concepts automatically, and search semantically across your study library.",
    )
    rows = database.get_chunks()
    if not rows:
        empty_state("Upload documents to cluster concepts.")
        return

    if st.button("Cluster Topics", type="primary", use_container_width=True):
        clusters, frame = TopicClusterer().cluster(rows)
        cluster_columns = st.columns(2)
        for index, cluster in enumerate(clusters):
            with cluster_columns[index % 2]:
                surface(f"Cluster {cluster.cluster_id}: {cluster.label}", f"{cluster.size} chunks. {cluster.sample}", "good")
        if not frame.empty:
            st.dataframe(frame, use_container_width=True, hide_index=True)
        st.info(TopicClusterer().quality_hint(rows))

    query = st.text_input("Semantic topic search", placeholder="Find related notes about neural networks")
    if query:
        hits = rag_engine().searcher.search(query, rows, top_k=6)
        for hit in hits:
            source_box(f"{hit.topic} | {hit.document_title} pages {hit.page_start}-{hit.page_end}", hit.text[:500])


def recommendations_view() -> None:
    hero(
        "Personal Study Plan",
        "Turn weak topics into a practical weekly schedule with flashcard review and focused revision blocks.",
    )
    minutes = st.slider("Daily study minutes", min_value=20, max_value=180, value=45, step=5)
    recommender = StudyRecommender()

    st.subheader("Personalized Actions")
    for rec in recommender.recommendations():
        tone = "high" if rec["priority"] == "High" else "medium"
        surface(f"{rec['priority']} priority: {rec['topic']}", rec["action"], tone)

    st.subheader("7-Day Study Plan")
    st.dataframe(pd.DataFrame(recommender.weekly_plan(minutes)), use_container_width=True, hide_index=True)

    st.subheader("Due Flashcards")
    cards = database.due_flashcards(limit=10)
    if not cards:
        empty_state("No flashcards due. Generate them by uploading PDFs.")
    for card in cards:
        with st.expander(card["front"]):
            st.write(card["back"])
            st.caption(f"Topic: {card['topic']}")


def architecture_view() -> None:
    hero(
        "AI Workflow Architecture",
        "A beginner-readable view of where RAG, local NLP, clustering, classification, and recommendations fit together.",
    )
    st.markdown(
        """
        **RAG pipeline:** PDF text extraction -> chunking -> sentence-transformer embeddings -> FAISS semantic search -> local QA model or extractive fallback -> answer with page sources.

        **Summarization:** HuggingFace summarization model when available, otherwise an extractive sentence scorer based on important terms.

        **Quiz generation:** local NLP heuristics convert key concepts into MCQs and short-answer questions. Attempts are persisted in SQLite.

        **Weak-topic ML:** quiz history becomes supervised data. A logistic regression classifier predicts weak topics once enough labeled outcomes exist.

        **Topic clustering:** TF-IDF vectors and KMeans group related chunks, then the app labels clusters with high-weight concept terms.

        **Recommendation engine:** weak-topic scores, study sessions, flashcards, and clustered topics drive revision actions and weekly schedules.
        """
    )


def main() -> None:
    bootstrap()
    apply_theme()

    pages = {
        "Dashboard": dashboard_view,
        "Upload": upload_view,
        "Ask": ask_view,
        "Summaries": summaries_view,
        "Quiz": quiz_view,
        "Topics": topics_view,
        "Study Plan": recommendations_view,
        "Architecture": architecture_view,
    }
    sidebar_brand()
    choice = st.sidebar.radio("Navigation", list(pages.keys()), label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Cost model**")
    st.sidebar.caption("Zero paid APIs. Local open-source models, SQLite, and FAISS.")
    st.sidebar.markdown("**Phase**")
    st.sidebar.caption("Phase 2 portfolio build.")
    pages[choice]()


if __name__ == "__main__":
    main()
