from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from . import database
from .text_processing import TextChunk


DEMO_FILENAME = "demo-ml-unit-1-5.pdf"
DEMO_TITLE = "ML Unit 1-5 Demo Notes"


DEMO_SECTIONS: list[dict[str, str]] = [
    {
        "topic": "Machine Learning Foundations",
        "text": (
            "Machine learning is the study of algorithms that improve performance from experience. "
            "A learning system observes examples, extracts patterns, and builds a model that can make predictions or decisions on unseen data. "
            "The usual pipeline contains data collection, preprocessing, feature engineering, model training, validation, testing, and deployment. "
            "The main learning paradigms are supervised learning, unsupervised learning, semi-supervised learning, reinforcement learning, and self-supervised learning. "
            "A model generalizes well when it performs strongly on new examples rather than memorizing training samples. "
            "Important sources of project failure include noisy labels, data leakage, class imbalance, poor evaluation design, and distribution shift between training and production data."
        ),
    },
    {
        "topic": "Supervised Learning",
        "text": (
            "Supervised learning uses labeled examples to learn a mapping from input features to target outputs. "
            "Regression predicts continuous values such as marks, temperature, or house price, while classification predicts categories such as pass or fail, spam or ham, and disease or healthy. "
            "Linear regression minimizes squared error and is interpretable when features are scaled and not highly collinear. "
            "Logistic regression estimates class probability using a sigmoid or softmax function and works as a strong baseline for classification. "
            "Decision trees split data into regions using feature thresholds, but deep trees can overfit unless regularized through depth limits, pruning, or minimum sample rules. "
            "Random forests reduce variance by averaging many decision trees trained on bootstrap samples, while gradient boosting builds trees sequentially to correct previous errors."
        ),
    },
    {
        "topic": "Model Evaluation",
        "text": (
            "Evaluation measures whether a model will be useful outside the training set. "
            "A train validation test split separates model fitting, model selection, and final reporting. "
            "K-fold cross validation estimates performance more reliably on small datasets by training multiple times on different folds. "
            "Accuracy is simple but can be misleading for imbalanced datasets, so precision, recall, F1 score, ROC AUC, confusion matrix, and calibration should be considered. "
            "Precision answers how many predicted positives were correct, while recall answers how many actual positives were found. "
            "For regression, common metrics include mean absolute error, mean squared error, root mean squared error, and R squared."
        ),
    },
    {
        "topic": "Bias Variance and Regularization",
        "text": (
            "Bias and variance explain common generalization errors. "
            "High bias means the model is too simple and underfits, producing poor training and test performance. "
            "High variance means the model is too sensitive to training data and overfits, producing strong training performance but weak test performance. "
            "Regularization reduces overfitting by discouraging overly complex models. "
            "L1 regularization can drive some coefficients to zero and supports feature selection, while L2 regularization shrinks coefficients smoothly. "
            "Early stopping, dropout, pruning, data augmentation, and cross validation are practical methods to improve generalization."
        ),
    },
    {
        "topic": "Unsupervised Learning",
        "text": (
            "Unsupervised learning discovers structure in unlabeled data. "
            "Clustering groups similar examples so that items in the same cluster are more alike than items in other clusters. "
            "K-means clustering represents each group by a centroid and assigns points to the nearest centroid, making it simple and fast but sensitive to scale and outliers. "
            "Hierarchical clustering builds nested groups and can be visualized with a dendrogram. "
            "DBSCAN discovers dense regions and can identify noise without preselecting the number of clusters. "
            "Dimensionality reduction methods such as PCA compress features while preserving important variance, helping visualization and noise reduction."
        ),
    },
    {
        "topic": "Feature Engineering",
        "text": (
            "Feature engineering transforms raw data into useful model inputs. "
            "Numeric features often require imputation, scaling, clipping, and transformation. "
            "Categorical features can be encoded using one-hot encoding, ordinal encoding, frequency encoding, or target encoding when leakage is carefully controlled. "
            "Text features can be represented through bag of words, TF-IDF, word embeddings, sentence embeddings, or transformer representations. "
            "Feature selection removes irrelevant or redundant variables to reduce overfitting and improve interpretability. "
            "Good feature engineering is guided by domain knowledge, exploratory data analysis, and validation performance."
        ),
    },
    {
        "topic": "Neural Networks",
        "text": (
            "Neural networks learn layered representations using differentiable functions. "
            "A neuron computes a weighted sum followed by a nonlinear activation such as ReLU, sigmoid, or tanh. "
            "Backpropagation calculates gradients of the loss with respect to parameters, and optimizers such as stochastic gradient descent or Adam update weights. "
            "Deep networks can model complex nonlinear relationships but require careful choices of learning rate, batch size, initialization, normalization, and regularization. "
            "Convolutional neural networks are effective for images because they use local filters and shared weights. "
            "Recurrent networks and transformers process sequences, with transformers using attention to model relationships between tokens."
        ),
    },
    {
        "topic": "Natural Language Processing",
        "text": (
            "Natural language processing helps machines work with human language. "
            "Traditional NLP pipelines include tokenization, stopword removal, stemming, lemmatization, part-of-speech tagging, and named entity recognition. "
            "TF-IDF highlights words that are frequent in one document but rare across the corpus. "
            "Embeddings represent words, sentences, or documents as dense vectors so semantic similarity can be measured numerically. "
            "Transformer models use self-attention to capture context and power tasks such as question answering, summarization, translation, and retrieval augmented generation. "
            "RAG combines retrieval from a knowledge base with generation so answers are grounded in source documents."
        ),
    },
    {
        "topic": "Recommendation Systems",
        "text": (
            "Recommendation systems personalize content, products, or study plans. "
            "Content-based recommendation compares item features with user preferences, while collaborative filtering learns from interactions between many users and items. "
            "Matrix factorization represents users and items in a shared latent space. "
            "Hybrid recommenders combine content, collaborative signals, rules, and contextual constraints. "
            "For a study companion, recommendations can prioritize weak topics, upcoming deadlines, spaced repetition due dates, quiz confidence, and study consistency. "
            "A useful recommender explains why a topic is recommended so students trust and act on the suggestion."
        ),
    },
    {
        "topic": "Responsible AI",
        "text": (
            "Responsible AI focuses on fairness, privacy, transparency, safety, and accountability. "
            "Bias can enter through historical data, sampling methods, labels, feature choices, and deployment feedback loops. "
            "Privacy-aware systems minimize sensitive data collection, protect stored data, and avoid exposing personal information in model outputs. "
            "Explainability helps users understand which factors influenced a prediction, especially in high-stakes domains. "
            "Monitoring is required after deployment because data distributions, user behavior, and performance can drift over time. "
            "A portfolio-grade AI system should document model limitations, evaluation metrics, and failure modes clearly."
        ),
    },
]


QUESTION_CONCEPTS = [
    ("Machine Learning Foundations", "generalization", "Generalization means performing well on unseen data instead of memorizing examples."),
    ("Machine Learning Foundations", "data leakage", "Data leakage happens when training includes information that would not be available at prediction time."),
    ("Machine Learning Foundations", "distribution shift", "Distribution shift occurs when production data differs from training data."),
    ("Supervised Learning", "classification", "Classification predicts discrete categories from labeled examples."),
    ("Supervised Learning", "regression", "Regression predicts continuous numeric values."),
    ("Supervised Learning", "logistic regression", "Logistic regression estimates class probabilities for classification problems."),
    ("Supervised Learning", "random forest", "A random forest averages many trees to reduce variance."),
    ("Supervised Learning", "gradient boosting", "Gradient boosting builds learners sequentially to correct previous errors."),
    ("Model Evaluation", "cross validation", "Cross validation estimates model performance across multiple data splits."),
    ("Model Evaluation", "confusion matrix", "A confusion matrix compares predicted classes with actual classes."),
    ("Model Evaluation", "precision", "Precision measures how many predicted positives were correct."),
    ("Model Evaluation", "recall", "Recall measures how many actual positives were found."),
    ("Model Evaluation", "F1 score", "F1 score balances precision and recall using their harmonic mean."),
    ("Bias Variance and Regularization", "overfitting", "Overfitting means memorizing training data and failing to generalize."),
    ("Bias Variance and Regularization", "underfitting", "Underfitting means the model is too simple to capture the pattern."),
    ("Bias Variance and Regularization", "L1 regularization", "L1 regularization can shrink some coefficients to zero."),
    ("Bias Variance and Regularization", "L2 regularization", "L2 regularization shrinks coefficients smoothly."),
    ("Unsupervised Learning", "K-means", "K-means assigns points to the nearest centroid."),
    ("Unsupervised Learning", "DBSCAN", "DBSCAN finds dense regions and labels sparse points as noise."),
    ("Unsupervised Learning", "PCA", "PCA projects data into directions of maximum variance."),
    ("Feature Engineering", "one-hot encoding", "One-hot encoding converts categories into binary indicator columns."),
    ("Feature Engineering", "TF-IDF", "TF-IDF weights words by local frequency and global rarity."),
    ("Feature Engineering", "feature selection", "Feature selection removes irrelevant or redundant features."),
    ("Neural Networks", "backpropagation", "Backpropagation computes gradients for neural network training."),
    ("Neural Networks", "activation function", "Activation functions add nonlinearity to neural networks."),
    ("Neural Networks", "Adam", "Adam is an adaptive optimizer often used for deep learning."),
    ("Natural Language Processing", "tokenization", "Tokenization splits text into smaller units such as words or subwords."),
    ("Natural Language Processing", "embeddings", "Embeddings represent text as dense semantic vectors."),
    ("Natural Language Processing", "RAG", "RAG retrieves relevant sources before generating an answer."),
    ("Recommendation Systems", "content-based filtering", "Content-based filtering recommends items similar to user preferences."),
    ("Recommendation Systems", "collaborative filtering", "Collaborative filtering learns from many users' interactions."),
    ("Responsible AI", "model drift", "Model drift happens when performance changes after deployment."),
    ("Responsible AI", "explainability", "Explainability helps users understand why a model made a prediction."),
]


DISTRACTORS = [
    "Dropout",
    "Tokenization",
    "Gradient Descent",
    "Decision Boundary",
    "Feature Scaling",
    "Batch Size",
    "K-fold Split",
    "Confusion Matrix",
    "Embedding Vector",
    "Attention",
    "Centroid",
    "Regularization",
]


DEMO_STUDY_SESSIONS = [
    ("Generalization", 50, "Reviewed train validation test split and leakage examples."),
    ("Classification Metrics", 45, "Practiced precision, recall, F1, and confusion matrices."),
    ("Clustering", 35, "Compared K-means, hierarchical clustering, and DBSCAN."),
    ("Feature Engineering", 55, "Built notes on encoding, scaling, text vectors, and feature selection."),
    ("Neural Networks", 40, "Reviewed activations, backpropagation, and optimizers."),
    ("RAG and NLP", 60, "Studied embeddings, retrieval, and source-grounded answers."),
]


def demo_document_id() -> int | None:
    with database.connect() as conn:
        row = conn.execute("SELECT id FROM documents WHERE filename = ?", (DEMO_FILENAME,)).fetchone()
        return int(row["id"]) if row else None


def make_demo_chunks() -> list[TextChunk]:
    chunks = []
    for index, section in enumerate(DEMO_SECTIONS):
        page_start = index * 3 + 1
        chunks.append(
            TextChunk(
                text=section["text"],
                page_start=page_start,
                page_end=page_start + 2,
                chunk_index=index,
            )
        )
    return chunks


def make_demo_questions(document_id: int) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    difficulties = ["Easy", "Medium", "Hard"]

    for index in range(120):
        topic, concept, explanation = QUESTION_CONCEPTS[index % len(QUESTION_CONCEPTS)]
        difficulty = difficulties[index % len(difficulties)]
        if index % 2 == 0:
            options = [concept.title()]
            for distractor in DISTRACTORS[index % len(DISTRACTORS) :] + DISTRACTORS[: index % len(DISTRACTORS)]:
                if distractor.lower() != concept.lower() and distractor not in options:
                    options.append(distractor)
                if len(options) == 4:
                    break
            rotation = index % 4
            options = options[rotation:] + options[:rotation]
            questions.append(
                {
                    "document_id": document_id,
                    "topic": topic,
                    "difficulty": difficulty,
                    "question_type": "MCQ",
                    "question": f"Which concept matches this description: {explanation}",
                    "option_a": options[0],
                    "option_b": options[1],
                    "option_c": options[2],
                    "option_d": options[3],
                    "correct_answer": concept.title(),
                    "explanation": explanation,
                }
            )
        else:
            questions.append(
                {
                    "document_id": document_id,
                    "topic": topic,
                    "difficulty": difficulty,
                    "question_type": "Short Answer",
                    "question": f"Explain {concept.title()} in one or two lines and mention why it matters in ML.",
                    "option_a": None,
                    "option_b": None,
                    "option_c": None,
                    "option_d": None,
                    "correct_answer": concept.title(),
                    "explanation": explanation,
                }
            )
    return questions


def make_demo_flashcards(document_id: int) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    cards = []
    for index, (topic, concept, explanation) in enumerate(QUESTION_CONCEPTS[:30]):
        cards.append(
            {
                "document_id": document_id,
                "topic": topic,
                "front": f"What is {concept.title()}?",
                "back": explanation,
                "due_at": (now - timedelta(days=index % 5)).isoformat(timespec="seconds"),
            }
        )
    return cards


def seed_demo_data() -> None:
    existing_document_id = demo_document_id()
    if existing_document_id is not None:
        ensure_demo_activity(existing_document_id)
        return

    chunks = make_demo_chunks()
    full_text = "\n\n".join(chunk.text for chunk in chunks)
    document_id = database.create_document(DEMO_FILENAME, DEMO_TITLE, full_text, page_count=32)
    database.add_chunks(document_id, chunks, embeddings=None)
    database.save_questions(make_demo_questions(document_id))
    database.save_flashcards(make_demo_flashcards(document_id))

    for topic, minutes, notes in DEMO_STUDY_SESSIONS:
        database.log_study_session(1, topic, minutes, notes)

    ensure_demo_activity(document_id)


def ensure_demo_activity(document_id: int) -> None:
    with database.connect() as conn:
        demo_rows = conn.execute(
            """
            SELECT user_answers.id
            FROM user_answers
            JOIN quiz_questions ON quiz_questions.id = user_answers.question_id
            WHERE quiz_questions.document_id = ?
              AND user_answers.selected_answer LIKE 'Demo %'
            LIMIT 1
            """,
            (document_id,),
        ).fetchone()
        if demo_rows:
            return

        question_ids = [
            int(row["id"])
            for row in conn.execute(
                "SELECT id FROM quiz_questions WHERE document_id = ?",
                (document_id,),
            ).fetchall()
        ]
        if question_ids:
            placeholders = ",".join("?" for _ in question_ids)
            conn.execute(
                f"""
                DELETE FROM user_answers
                WHERE question_id IN ({placeholders})
                  AND selected_answer IN ('Correct demo answer', 'Needs revision')
                """,
                tuple(question_ids),
            )
            conn.execute(
                """
                DELETE FROM quiz_attempts
                WHERE id NOT IN (SELECT DISTINCT attempt_id FROM user_answers)
                """
            )

    answer_plan = [
        ("Machine Learning Foundations", 0.85),
        ("Supervised Learning", 0.78),
        ("Model Evaluation", 0.62),
        ("Bias Variance and Regularization", 0.58),
        ("Unsupervised Learning", 0.52),
        ("Feature Engineering", 0.74),
        ("Neural Networks", 0.47),
        ("Natural Language Processing", 0.68),
        ("Recommendation Systems", 0.56),
        ("Responsible AI", 0.71),
    ]

    questions_by_topic: dict[str, list[int]] = {}
    with database.connect() as conn:
        rows = conn.execute(
            "SELECT id, topic FROM quiz_questions WHERE document_id = ? ORDER BY id ASC",
            (document_id,),
        ).fetchall()
    for row in rows:
        questions_by_topic.setdefault(row["topic"], []).append(int(row["id"]))

    for attempt_index in range(8):
        attempt_id = database.start_attempt(1)
        for topic, target_accuracy in answer_plan:
            ids = questions_by_topic.get(topic, [])
            if not ids:
                continue
            question_id = ids[(attempt_index + len(topic)) % len(ids)]
            correct_slots = round(target_accuracy * 8)
            is_correct = attempt_index < correct_slots
            selected = "Demo correct answer" if is_correct else "Demo revision needed"
            database.record_answer(attempt_id, question_id, selected, is_correct, topic)
        database.finish_attempt(attempt_id)
