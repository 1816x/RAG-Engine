"use client";

import { useCallback, useEffect, useState } from "react";
import type { ChangeEvent, FormEvent } from "react";
import type { DocumentInfo, QueryResult, Stats } from "./lib/rag";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  const body = (await response.json()) as T & { error?: string };
  if (!response.ok) {
    throw new Error(body.error ?? "Request failed");
  }
  return body;
}

function ModeBadge({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: "blue" | "green" | "amber";
}) {
  return (
    <div className={`mode-badge ${tone}`}>
      <span className="status-dot" aria-hidden="true" />
      <span>
        <small>{label}</small>
        <strong>{value}</strong>
      </span>
    </div>
  );
}

function Metric({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<QueryResult | null>(null);
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryError, setQueryError] = useState<string | null>(null);

  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [workspaceLoading, setWorkspaceLoading] = useState(true);
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);

  const [documentTitle, setDocumentTitle] = useState("");
  const [documentText, setDocumentText] = useState("");
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadNotice, setUploadNotice] = useState<string | null>(null);
  const [fileInputKey, setFileInputKey] = useState(0);
  const [editingDocumentId, setEditingDocumentId] = useState<number | null>(null);
  const [mutationDocumentId, setMutationDocumentId] = useState<number | null>(null);

  const uploadsEnabled = stats?.uploads_enabled ?? false;
  const titleLimit = stats?.limits.title_chars ?? 200;
  const documentLimit = stats?.limits.document_chars ?? 1_000_000;
  const questionLimit = stats?.limits.question_chars ?? 2_000;

  const loadWorkspace = useCallback(async () => {
    setWorkspaceLoading(true);
    setWorkspaceError(null);
    try {
      const [nextDocuments, nextStats] = await Promise.all([
        requestJson<DocumentInfo[]>("/api/documents"),
        requestJson<Stats>("/api/stats"),
      ]);
      setDocuments(nextDocuments);
      setStats(nextStats);
    } catch (err) {
      setWorkspaceError(err instanceof Error ? err.message : "Unable to load index");
    } finally {
      setWorkspaceLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadWorkspace();
  }, [loadWorkspace]);

  async function ask(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) return;

    setQueryLoading(true);
    setQueryError(null);
    try {
      const nextResult = await requestJson<QueryResult>("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmedQuestion, k: 5 }),
      });
      setResult(nextResult);
    } catch (err) {
      setQueryError(err instanceof Error ? err.message : "Unable to run query");
      setResult(null);
    } finally {
      setQueryLoading(false);
    }
  }

  async function chooseFile(event: ChangeEvent<HTMLInputElement>) {
    if (!uploadsEnabled) return;
    const file = event.target.files?.[0];
    if (!file) return;

    setUploadError(null);
    try {
      const text = await file.text();
      if (text.length > documentLimit) {
        setUploadError(
          `Document exceeds the ${documentLimit.toLocaleString()} character limit.`,
        );
        event.target.value = "";
        return;
      }
      setDocumentText(text);
      if (!documentTitle.trim()) {
        setDocumentTitle(file.name.replace(/\.(md|markdown|txt)$/i, ""));
      }
    } catch {
      setUploadError("The selected file could not be read.");
    }
  }

  async function uploadDocument(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!uploadsEnabled || !documentTitle.trim() || !documentText.trim()) return;

    setUploadLoading(true);
    setUploadError(null);
    setUploadNotice(null);
    try {
      const created = await requestJson<DocumentInfo>(
        editingDocumentId === null
          ? "/api/documents"
          : `/api/documents/${editingDocumentId}`,
        {
          method: editingDocumentId === null ? "POST" : "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            title: documentTitle.trim(),
            text: documentText,
          }),
        },
      );
      setUploadNotice(
        `${editingDocumentId === null ? "Indexed" : "Replaced"} “${created.title}” as ${created.n_chunks} ${created.n_chunks === 1 ? "chunk" : "chunks"}.`,
      );
      setDocumentTitle("");
      setDocumentText("");
      setFileInputKey((key) => key + 1);
      setEditingDocumentId(null);
      await loadWorkspace();
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Unable to index document");
    } finally {
      setUploadLoading(false);
    }
  }

  function beginReplacement(document: DocumentInfo) {
    setEditingDocumentId(document.id);
    setDocumentTitle(document.title);
    setDocumentText("");
    setUploadError(null);
    setUploadNotice(`Paste the replacement text for “${document.title}”.`);
  }

  async function removeDocument(document: DocumentInfo) {
    if (
      !uploadsEnabled ||
      !window.confirm(`Delete “${document.title}” and all of its chunks?`)
    )
      return;
    setMutationDocumentId(document.id);
    setWorkspaceError(null);
    try {
      await requestJson<DocumentInfo>(`/api/documents/${document.id}`, {
        method: "DELETE",
      });
      if (editingDocumentId === document.id) {
        setEditingDocumentId(null);
        setDocumentTitle("");
        setDocumentText("");
      }
      await loadWorkspace();
    } catch (err) {
      setWorkspaceError(err instanceof Error ? err.message : "Unable to delete document");
    } finally {
      setMutationDocumentId(null);
    }
  }

  const hashedRetrieval = stats?.embedder === "HashedEmbedder";
  const retrievalLabel = stats
    ? hashedRetrieval
      ? "Lexical · hashed"
      : "Semantic embeddings"
    : "Loading";
  const generationLabel =
    stats?.generation === "claude" ? "Claude" : stats ? "Extractive mock" : "Loading";

  return (
    <main className="shell">
      <header className="hero">
        <div>
          <p className="eyebrow">FROM-SCRATCH VECTOR RETRIEVAL</p>
          <h1>See what your index knows.</h1>
          <p className="hero-copy">
            Add source material, inspect the live HNSW index, and ask grounded
            questions without hiding the retrieval pipeline.
          </p>
        </div>
        <div className="mode-strip" aria-label="Active runtime modes">
          <ModeBadge
            label="Retrieval"
            value={retrievalLabel}
            tone={hashedRetrieval ? "amber" : "green"}
          />
          <ModeBadge
            label="Answer"
            value={generationLabel}
            tone={stats?.generation === "claude" ? "green" : "blue"}
          />
        </div>
      </header>

      {workspaceError && (
        <div className="banner error-banner">
          <span>{workspaceError}</span>
          <button type="button" className="text-button" onClick={() => void loadWorkspace()}>
            Retry
          </button>
        </div>
      )}

      <section className="metrics-grid" aria-label="Index statistics">
        <Metric
          label="Corpus"
          value={stats ? `${stats.documents} docs` : "—"}
          detail={stats ? `${stats.chunks} indexed chunks` : "Loading index"}
        />
        <Metric
          label="Embedding space"
          value={stats ? `${stats.dim}D` : "—"}
          detail={stats ? `${stats.metric} · ${stats.embedder}` : "Loading backend"}
        />
        <Metric
          label="HNSW graph"
          value={stats ? `M = ${stats.m}` : "—"}
          detail={stats ? `ef construction ${stats.ef_construction}` : "Loading graph"}
        />
        <Metric
          label="Relevance floor"
          value={stats?.min_score == null ? "Off" : stats.min_score.toFixed(2)}
          detail="Weak matches are removed"
        />
      </section>

      <section className="query-grid">
        <div className="panel query-panel">
          <div className="panel-heading">
            <div>
              <p className="section-kicker">QUERY</p>
              <h2>Ask the indexed corpus</h2>
            </div>
            <span className="step-label">01</span>
          </div>

          <form onSubmit={ask}>
            <label htmlFor="question">Question</label>
            <textarea
              id="question"
              rows={4}
              maxLength={questionLimit}
              value={question}
              placeholder="How does HNSW search descend through its layers?"
              onChange={(event) => setQuestion(event.target.value)}
            />
            <div className="form-actions">
              <span className="form-hint">
                {question.length.toLocaleString()} / {questionLimit.toLocaleString()} characters
                · up to 5 sources
              </span>
              <button type="submit" disabled={queryLoading || !question.trim()}>
                {queryLoading ? "Searching…" : "Search index"}
              </button>
            </div>
          </form>

          {queryError && <p className="inline-message error-message">{queryError}</p>}

          {result && (
            <article className="answer-card">
              <div className="answer-meta">
                <span>ANSWER</span>
                <span className={`answer-mode ${result.model === "mock" ? "mock" : "live"}`}>
                  {result.model === "mock" ? "Extractive mock" : result.model}
                </span>
              </div>
              <div className="answer-copy">{result.answer}</div>
            </article>
          )}
        </div>

        <aside className="panel sources-panel">
          <div className="panel-heading compact">
            <div>
              <p className="section-kicker">EVIDENCE</p>
              <h2>Retrieved sources</h2>
            </div>
            <span className="source-count">{result?.sources.length ?? 0}</span>
          </div>

          {!result && (
            <div className="empty-state">
              <strong>No query yet</strong>
              <span>Relevant chunks and similarity scores will appear here.</span>
            </div>
          )}

          {result && result.sources.length === 0 && (
            <div className="empty-state">
              <strong>No relevant context</strong>
              <span>No chunk cleared the active relevance threshold.</span>
            </div>
          )}

          <div className="source-list">
            {result?.sources.map((source, index) => (
              <article
                key={source.chunk_id}
                className={`source-card${source.cited ? " cited" : ""}`}
              >
                <div className="source-card-head">
                  <span className="source-index">[{index + 1}]</span>
                  <span className="source-title">{source.doc_title}</span>
                  <span className="source-score">{source.score.toFixed(3)}</span>
                </div>
                <p>{source.text}</p>
                <div className="source-footer">
                  <span>Chunk {source.ordinal + 1}</span>
                  {source.cited && <span className="cited-label">Used in answer</span>}
                </div>
              </article>
            ))}
          </div>
        </aside>
      </section>

      <section className="document-section">
        <div className="section-heading">
          <div>
            <p className="section-kicker">DOCUMENT WORKSPACE</p>
            <h2>Shape the knowledge base</h2>
            <p>
              Add Markdown or plain text. The service chunks, embeds, and inserts it
              into the live in-memory HNSW index.
            </p>
          </div>
          <button
            type="button"
            className="ghost-button"
            onClick={() => void loadWorkspace()}
            disabled={workspaceLoading}
          >
            {workspaceLoading ? "Refreshing…" : "Refresh index"}
          </button>
        </div>

        <div className="document-grid">
          <form
            className={`panel upload-panel${uploadsEnabled ? "" : " locked"}`}
            onSubmit={uploadDocument}
            aria-disabled={!uploadsEnabled}
          >
            <div className="panel-heading compact">
              <div>
                <p className="section-kicker">
                  {editingDocumentId === null ? "ADD SOURCE" : "REPLACE SOURCE"}
                </p>
                <h3>
                  {editingDocumentId === null
                    ? "Index a document"
                    : `Replace document ${editingDocumentId}`}
                </h3>
              </div>
              <span className={`upload-status ${uploadsEnabled ? "enabled" : "locked"}`}>
                {stats ? (uploadsEnabled ? "Enabled" : "Locked") : "Loading"}
              </span>
            </div>

            {stats && !uploadsEnabled && (
              <div className="upload-lock-notice" role="status">
                <strong>Uploads are disabled on this deployment.</strong>
                <span>
                  The workspace stays visible for transparency. Set
                  {" "}<code>RAG_UPLOADS_ENABLED=1</code>{" "}on the service to unlock it.
                </span>
              </div>
            )}

            <label htmlFor="document-file">Choose a text file</label>
            <input
              key={fileInputKey}
              id="document-file"
              className="file-input"
              type="file"
              accept=".md,.markdown,.txt,text/plain,text/markdown"
              disabled={!uploadsEnabled || uploadLoading}
              onChange={(event) => void chooseFile(event)}
            />

            <div className="divider"><span>or paste text</span></div>

            <label htmlFor="document-title">Title</label>
            <input
              id="document-title"
              value={documentTitle}
              maxLength={titleLimit}
              disabled={!uploadsEnabled || uploadLoading}
              placeholder="Architecture notes"
              onChange={(event) => setDocumentTitle(event.target.value)}
            />
            <span className="character-count">
              {documentTitle.length.toLocaleString()} / {titleLimit.toLocaleString()} characters
            </span>

            <label htmlFor="document-text">Document text</label>
            <textarea
              id="document-text"
              rows={9}
              value={documentText}
              maxLength={documentLimit}
              disabled={!uploadsEnabled || uploadLoading}
              placeholder="Paste the source material to index…"
              onChange={(event) => setDocumentText(event.target.value)}
            />
            <span className="character-count">
              {documentText.length.toLocaleString()} / {documentLimit.toLocaleString()} characters
            </span>

            <div className="form-actions">
              <span className="form-hint">
                {uploadsEnabled
                  ? "Stored in memory for this service instance"
                  : "Read-only public demo"}
              </span>
              <button
                type="submit"
                disabled={
                  !uploadsEnabled ||
                  uploadLoading ||
                  !documentTitle.trim() ||
                  !documentText.trim()
                }
              >
                {uploadLoading
                  ? "Saving…"
                  : editingDocumentId === null
                    ? "Add to index"
                    : "Replace document"}
              </button>
            </div>

            {uploadError && <p className="inline-message error-message">{uploadError}</p>}
            {uploadNotice && <p className="inline-message success-message">{uploadNotice}</p>}
          </form>

          <div className="panel library-panel">
            <div className="panel-heading compact">
              <div>
                <p className="section-kicker">LIVE CORPUS</p>
                <h3>Indexed documents</h3>
              </div>
              <span className="document-total">{documents.length}</span>
            </div>

            {workspaceLoading && documents.length === 0 && (
              <div className="empty-state">
                <strong>Loading documents</strong>
                <span>Reading the current in-memory index.</span>
              </div>
            )}

            {!workspaceLoading && documents.length === 0 && (
              <div className="empty-state">
                <strong>The index is empty</strong>
                <span>Add a document to make the first grounded query.</span>
              </div>
            )}

            <div className="document-list">
              {documents.map((document, index) => (
                <div className="document-row" key={document.id}>
                  <span className="document-number">{String(index + 1).padStart(2, "0")}</span>
                  <div>
                    <strong>{document.title}</strong>
                    <span>
                      {document.n_chunks} {document.n_chunks === 1 ? "chunk" : "chunks"}
                    </span>
                  </div>
                  <div className="document-actions">
                    <button
                      type="button"
                      className="text-button"
                      disabled={!uploadsEnabled || mutationDocumentId !== null}
                      onClick={() => beginReplacement(document)}
                    >
                      Replace
                    </button>
                    <button
                      type="button"
                      className="text-button danger-button"
                      disabled={!uploadsEnabled || mutationDocumentId !== null}
                      onClick={() => void removeDocument(document)}
                    >
                      {mutationDocumentId === document.id ? "Deleting…" : "Delete"}
                    </button>
                  </div>
                </div>
              ))}
            </div>

            <div className="mode-explainer">
              <h4>What is real in this demo?</h4>
              <p>
                HNSW indexing and retrieval always run in Rust.{" "}
                {hashedRetrieval
                  ? "The active hashed embedder matches term overlap, so retrieval is lexical rather than semantic."
                  : "The active embedding model places meaningfully similar text close together."}
              </p>
              <p>
                {stats?.generation === "claude"
                  ? "Claude is generating the final answer from the retrieved chunks."
                  : "Answer generation is in keyless mock mode and extracts the top source directly."}
              </p>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
