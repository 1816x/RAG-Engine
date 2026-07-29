// Server-side client for the Python RAG service. Runs only in route handlers,
// so RAG_SERVICE_URL and any secrets stay off the client.

const SERVICE_URL = process.env.RAG_SERVICE_URL ?? "http://localhost:8000";

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

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${SERVICE_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`RAG service ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
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

export function listDocuments(): Promise<DocumentInfo[]> {
  return call<DocumentInfo[]>("/documents");
}
