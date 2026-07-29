"use client";

import { useState } from "react";
import type { QueryResult } from "./lib/rag";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function ask(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, k: 5 }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.error ?? "request failed");
      setResult(body);
    } catch (err) {
      setError(err instanceof Error ? err.message : "unknown error");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="layout">
      <section>
        <h1>RAG Engine</h1>
        <p className="subtitle">
          Ask a question over the indexed documents. Retrieval runs on a
          from-scratch HNSW index; the answer is generated with cited sources.
        </p>

        <form onSubmit={ask}>
          <textarea
            rows={3}
            value={question}
            placeholder="e.g. How does HNSW search descend through layers?"
            onChange={(e) => setQuestion(e.target.value)}
          />
          <div className="row">
            <button type="submit" disabled={loading || !question.trim()}>
              {loading ? "Searching…" : "Ask"}
            </button>
          </div>
        </form>

        {error && <p className="error">{error}</p>}

        {result && (
          <div className="answer">
            <span className="model-badge">model: {result.model}</span>
            <div>{result.answer}</div>
          </div>
        )}
      </section>

      <aside className="sources">
        <h2>Retrieved sources</h2>
        {!result && <p className="subtitle">Sources appear here after you ask.</p>}
        {result?.sources.map((s, i) => (
          <div key={s.chunk_id} className={`source${s.cited ? " cited" : ""}`}>
            <div className="source-head">
              <span>
                [{i + 1}] {s.doc_title} · chunk {s.ordinal}
              </span>
              <span>
                {s.cited && <span className="cited-tag">cited · </span>}
                {s.score.toFixed(3)}
              </span>
            </div>
            <div>{s.text}</div>
          </div>
        ))}
      </aside>
    </main>
  );
}
