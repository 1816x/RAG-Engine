// Where the Next.js server-side routes reach the Python RAG service.
//
// This is a public backend URL, not a secret, so it is committed as a default:
// the Vercel deploy then needs no dashboard configuration. Override it with the
// RAG_SERVICE_URL environment variable for local development or a different
// backend.
//
// Replace this with the deployed Fly.io URL (https://<app>.fly.dev) when the
// backend goes live.
export const DEFAULT_RAG_SERVICE_URL = "http://localhost:8000";
