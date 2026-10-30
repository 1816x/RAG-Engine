"""Document classification and TF-IDF retrieval over a small synthetic corpus."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CATEGORIES = ("Incident Report", "Technical Guide", "Test Procedure", "Release Note")


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    category: str
    area: str
    concept: str
    text: str


def load_documents(data_dir: Path = DATA) -> list[Document]:
    """Read the index and local text files; fail on missing or inconsistent entries."""
    data_dir = Path(data_dir).resolve()
    docs_dir = (data_dir / "documents").resolve()
    rows: list[Document] = []
    with (data_dir / "index.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            file_path = (docs_dir / row["file"]).resolve()
            if not file_path.is_relative_to(docs_dir):
                raise ValueError("Document path escapes data/documents")
            if row["category"] not in CATEGORIES:
                raise ValueError(f"Unknown category: {row['category']}")
            rows.append(Document(
                row["id"], row["title"], row["category"], row["area"],
                row["concept"], file_path.read_text(encoding="utf-8").strip()
            ))
    if len({doc.id for doc in rows}) != len(rows):
        raise ValueError("Document IDs must be unique")
    if not rows or any(not doc.text for doc in rows):
        raise ValueError("Corpus must contain nonempty documents")
    return rows


def classifier():
    """TF-IDF text features followed by a simple, regularized classifier."""
    return make_pipeline(
        TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=2000, random_state=7),
    )


def evaluate(docs: list[Document]) -> dict:
    """Stratified holdout plus three-fold cross-validation on the corpus."""
    texts = [doc.text for doc in docs]
    labels = [doc.category for doc in docs]
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=7, stratify=labels
    )
    model = classifier().fit(x_train, y_train)
    predictions = model.predict(x_test)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
    scores = cross_val_score(classifier(), texts, labels, cv=cv, scoring="accuracy")
    return {
        "train_count": len(x_train),
        "test_count": len(x_test),
        "accuracy": accuracy_score(y_test, predictions),
        "cv_scores": scores.tolist(),
        "labels": list(CATEGORIES),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=CATEGORIES).tolist(),
        "report": classification_report(y_test, predictions, labels=CATEGORIES, zero_division=0),
    }


def classify(text: str, docs: list[Document]) -> tuple[str, float]:
    """Fit on all examples, then classify a newly supplied text."""
    if not text.strip():
        raise ValueError("Document text cannot be empty")
    model = classifier().fit([d.text for d in docs], [d.category for d in docs])
    probabilities = model.predict_proba([text])[0]
    index = int(probabilities.argmax())
    # This is a model score on a tiny synthetic sample, not a calibrated probability.
    return str(model.classes_[index]), float(probabilities[index])


def search(query: str, docs: list[Document], top_k: int = 5) -> list[tuple[Document, float]]:
    """Rank documents by cosine similarity in the corpus TF-IDF feature space."""
    if not query.strip():
        raise ValueError("Search query cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), sublinear_tf=True
    )
    matrix = vectorizer.fit_transform([doc.text for doc in docs])
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, matrix).ravel()
    ranked = sorted(enumerate(similarities), key=lambda item: (-item[1], docs[item[0]].id))
    return [(docs[index], float(score)) for index, score in ranked[:top_k] if score > 0]

