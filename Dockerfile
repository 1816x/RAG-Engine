# Container for the RAG service (service/) — the Next.js app deploys separately
# to Vercel and is excluded via .dockerignore.
#
# Two stages because the service needs a *compiled* Rust extension: stage one
# has the Rust toolchain and builds the wheel, stage two installs only that
# wheel. The wheel is abi3 (see bindings/pyproject.toml), so it is portable
# across CPython >= 3.9 and the stage split is clean.

# ---------- stage 1: build the PyO3 wheel ----------
FROM python:3.11-slim-bookworm AS builder

# build-essential: pyo3 needs a C compiler/linker. curl: to fetch rustup.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --no-cache-dir maturin

WORKDIR /src
# bindings/ depends on the engine via `path = "../engine"`, and the workspace
# root manifest declares both members — so all of these are required for the
# build to resolve.
COPY Cargo.toml Cargo.lock ./
COPY engine/ engine/
COPY bindings/ bindings/

# Reads bindings/pyproject.toml, which sets the extension-module feature and
# ships the pure-Python hnsw_rag package alongside the compiled module.
RUN cd bindings && maturin build --release --out /wheels

# ---------- stage 2: runtime ----------
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # A hosted container has nobody to run scripts/seed.py, so seed in-process.
    RAG_SEED_ON_STARTUP=1

WORKDIR /app

COPY --from=builder /wheels/*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

COPY service/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY service/rag_service/ rag_service/
# app.py resolves the corpus as <parent of rag_service>/sample_docs, i.e.
# /app/sample_docs. Startup seeding reads this; without it the demo boots empty.
COPY service/sample_docs/ sample_docs/

# Drop privileges — nothing here needs root at runtime.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# --workers 1 is deliberate, not a placeholder: the HNSW index lives in this
# process's memory, so a second worker would serve a different (empty) index.
CMD ["uvicorn", "rag_service.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
