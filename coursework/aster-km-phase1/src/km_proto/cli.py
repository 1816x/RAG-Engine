"""Command line interface for reproducible classification and search examples."""

from __future__ import annotations

import argparse
import json
from .core import classify, evaluate, load_documents, search


def main() -> None:
    parser = argparse.ArgumentParser(description="Aster Systems KM prototype")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("evaluate", help="Holdout and cross-validation results")
    classify_parser = commands.add_parser("classify", help="Classify new text")
    classify_parser.add_argument("--text", required=True)
    search_parser = commands.add_parser("search", help="Rank documents for a query")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--top-k", type=int, default=5)
    commands.add_parser("demo", help="Run two classification and two search examples")
    args = parser.parse_args()
    docs = load_documents()

    if args.command == "evaluate":
        print(json.dumps(evaluate(docs), indent=2))
    elif args.command == "classify":
        label, score = classify(args.text, docs)
        print(f"Category: {label} | model score: {score:.3f}")
    elif args.command == "search":
        print(f"Query: {args.query}")
        for position, (doc, score) in enumerate(search(args.query, docs, args.top_k), 1):
            print(f"{position}. {doc.id} | {doc.title} | {doc.category} | cosine={score:.3f}")
    else:
        print(f"Loaded {len(docs)} fictional documents across 4 categories")
        for sample in [
            "The operator reported a blank command display, with logs and severity recorded for triage.",
            "Version 2.4.1 fixes the timeout and lists deployment notes and rollback impact.",
        ]:
            label, score = classify(sample, docs)
            print(f"CLASSIFY {sample}\n  => {label} (model score {score:.3f})")
        for query in [
            "How can QA measure early login session expiration interval with expected results?",
            "Which release corrected the display timeout and what changed?",
        ]:
            print(f"SEARCH {query}")
            for rank, (doc, score) in enumerate(search(query, docs, 3), 1):
                print(f"  {rank}. {doc.id} | {doc.title} | {doc.category} | cosine={score:.3f}")


if __name__ == "__main__":
    main()

