"""Upload the sample corpus to a running RAG service.

Usage:
    python scripts/seed.py [--url http://localhost:8000] [--docs sample_docs]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import urllib.error
import urllib.request
import json


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--docs",
        default=str(pathlib.Path(__file__).resolve().parent.parent / "sample_docs"),
    )
    args = parser.parse_args()

    docs_dir = pathlib.Path(args.docs)
    files = sorted(docs_dir.glob("*.md"))
    if not files:
        print(f"no .md files found in {docs_dir}", file=sys.stderr)
        return 1

    for f in files:
        title = f.stem.replace("_", " ")
        try:
            result = post_json(
                f"{args.url}/documents", {"title": title, "text": f.read_text()}
            )
        except urllib.error.URLError as e:
            print(f"failed to reach {args.url}: {e}", file=sys.stderr)
            return 1
        print(f"indexed {title!r}: {result['n_chunks']} chunk(s)")

    stats = json.load(urllib.request.urlopen(f"{args.url}/stats", timeout=30))
    print(f"index now holds {stats['chunks']} chunks across {stats['documents']} docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
