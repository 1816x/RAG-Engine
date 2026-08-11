const DEFAULT_MAX_REQUEST_BYTES = 4 * 1024 * 1024;

function positiveInt(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;
  const parsed = Number(raw);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

export const INPUT_LIMITS = {
  titleChars: positiveInt("RAG_MAX_TITLE_CHARS", 200),
  documentChars: positiveInt("RAG_MAX_DOCUMENT_CHARS", 1_000_000),
  questionChars: positiveInt("RAG_MAX_QUESTION_CHARS", 2_000),
  requestBytes: positiveInt("RAG_MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES),
} as const;

export class InputRequestError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "InputRequestError";
  }
}

export async function readLimitedJson<T>(request: Request): Promise<T> {
  const contentLength = request.headers.get("content-length");
  if (contentLength !== null) {
    const declaredLength = Number(contentLength);
    if (!Number.isInteger(declaredLength) || declaredLength < 0) {
      throw new InputRequestError("content-length header must be a non-negative integer", 400);
    }
    if (declaredLength > INPUT_LIMITS.requestBytes) {
      throw new InputRequestError("request body is too large", 413);
    }
  }

  if (!request.body) {
    throw new InputRequestError("request body must be valid JSON", 400);
  }

  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > INPUT_LIMITS.requestBytes) {
      await reader.cancel();
      throw new InputRequestError("request body is too large", 413);
    }
    chunks.push(value);
  }

  try {
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)) as T;
  } catch {
    throw new InputRequestError("request body must be valid JSON", 400);
  }
}
