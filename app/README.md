# app/ — RAG chat app (Next.js)

A minimal Next.js UI over the Python RAG service (`../service`): ask a question, get a generated answer, and see the retrieved source chunks with their similarity scores (cited chunks highlighted).

The browser only talks to same-origin `/api/*` route handlers, which proxy to the FastAPI service server-side — so `RAG_SERVICE_URL` and the Claude API key never reach the client.

## Run

Start the service first (see `../service`), then:

```sh
cd app
npm install
RAG_SERVICE_URL=http://localhost:8000 npm run dev
```

Open http://localhost:3000. Without `RAG_SERVICE_URL` it defaults to `http://localhost:8000`.

## Pipeline

```
browser → Next.js /api/query → FastAPI /query
  → embed question → HNSW search → Claude answer with cited sources → back to UI
```
