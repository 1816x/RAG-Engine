// Server-side client for the Python RAG service. Runs only in route handlers,
// so RAG_SERVICE_URL and any secrets stay off the client.

import { DEFAULT_RAG_SERVICE_URL } from "./config";

const SERVICE_URL = process.env.RAG_SERVICE_URL ?? DEFAULT_RAG_SERVICE_URL;
const SERVICE_TIMEOUT_MS = 30_000;

export interface Source {
  chunk_id: number;
  doc_id: number;
  doc_title: string;
  ordinal: number;
  score: number;
  text: string;
  cited: boolean;
}

export interface QueryResult {
  question: string;
  answer: string;
  model: string;
  sources: Source[];
}

export interface DocumentInfo {
  id: number;
  title: string;
  n_chunks: number;
}

export interface Stats {
  documents: number;
  chunks: number;
  dim: number;
  metric: string;
  min_score: number | null;
  m: number;
  ef_construction: number;
  embedder: string;
  generation: "mock" | "claude";
  uploads_enabled: boolean;
  persistence: { enabled: boolean; loaded: boolean; format_version: number };
  limits: {
    title_chars: number;
    document_chars: number;
    question_chars: number;
    request_bytes: number;
  };
}

export class RagServiceError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "RagServiceError";
  }
}

function serviceTimeoutMs(): number {
  const value = Number(process.env.RAG_SERVICE_TIMEOUT_MS);
  return Number.isInteger(value) && value > 0 ? value : SERVICE_TIMEOUT_MS;
}

function errorMessage(body: string, status: number): string {
  try {
    const parsed = JSON.parse(body) as { detail?: unknown; error?: unknown };
    if (typeof parsed.detail === "string") return parsed.detail;
    if (typeof parsed.error === "string") return parsed.error;
    if (Array.isArray(parsed.detail)) {
      const messages = parsed.detail
        .map((item) =>
          typeof item === "object" &&
          item !== null &&
          "msg" in item &&
          typeof item.msg === "string"
            ? item.msg
            : null,
        )
        .filter((message): message is string => message !== null);
      if (messages.length) return messages.join("; ");
    }
  } catch {
    // Preserve a non-JSON service response below.
  }
  return body || `RAG service request failed with status ${status}`;
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${SERVICE_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
      cache: "no-store",
      signal: init?.signal ?? AbortSignal.timeout(serviceTimeoutMs()),
    });
  } catch (error) {
    if (error instanceof Error && (error.name === "TimeoutError" || error.name === "AbortError")) {
      throw new RagServiceError("RAG service request timed out", 504);
    }
    throw error;
  }
  if (!response.ok) {
    const body = await response.text();
    throw new RagServiceError(errorMessage(body, response.status), response.status);
  }
  return response.json() as Promise<T>;
}

export function query(question: string, k = 5): Promise<QueryResult> {
  return call<QueryResult>("/query", {
    method: "POST",
    body: JSON.stringify({ question, k }),
  });
}

export function addDocument(title: string, text: string): Promise<DocumentInfo> {
  return call<DocumentInfo>("/documents", {
    method: "POST",
    body: JSON.stringify({ title, text }),
  });
}

export function deleteDocument(id: number): Promise<DocumentInfo> {
  return call<DocumentInfo>(`/documents/${id}`, { method: "DELETE" });
}

export function replaceDocument(
  id: number,
  title: string,
  text: string,
): Promise<DocumentInfo> {
  return call<DocumentInfo>(`/documents/${id}`, {
    method: "PUT",
    body: JSON.stringify({ title, text }),
  });
}

export function listDocuments(): Promise<DocumentInfo[]> {
  return call<DocumentInfo[]>("/documents");
}

export function getStats(): Promise<Stats> {
  return call<Stats>("/stats");
}
