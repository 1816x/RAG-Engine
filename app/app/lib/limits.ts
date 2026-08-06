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
  const declaredLength = Number(request.headers.get("content-length"));
  if (Number.isFinite(declaredLength) && declaredLength > INPUT_LIMITS.requestBytes) {
    throw new InputRequestError("request body is too large", 413);
  }

  const bytes = await request.arrayBuffer();
  if (bytes.byteLength > INPUT_LIMITS.requestBytes) {
    throw new InputRequestError("request body is too large", 413);
  }

  try {
    return JSON.parse(new TextDecoder().decode(bytes)) as T;
  } catch {
    throw new InputRequestError("request body must be valid JSON", 400);
  }
}
