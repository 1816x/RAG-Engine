# app/ — RAG chat app (Next.js)

Lands in **Phase 4**. A small Next.js app that consumes the engine end-to-end: upload documents, ask questions, see the retrieved sources and the Claude-generated answer with citations.

Pipeline: document upload → chunking + embeddings (Python service using the PyO3 bindings) → HNSW retrieval → Claude API generation with cited sources → chat UI.
