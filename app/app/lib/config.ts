// Where the Next.js server-side routes reach the Python RAG service.
//
// This is a public backend URL, not a secret, so it is committed as a default:
// the Vercel deploy then needs no dashboard configuration. Override it with the
// RAG_SERVICE_URL environment variable for local development or a different
// backend.
//
// For local development against a service on port 8000:
//   RAG_SERVICE_URL=http://localhost:8000 npm run dev
export const DEFAULT_RAG_SERVICE_URL = "https://rag-engine-demo.fly.dev";
