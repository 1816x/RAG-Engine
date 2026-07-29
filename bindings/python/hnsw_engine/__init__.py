"""HNSW vector index — Python bindings for the from-scratch Rust engine.

The heavy lifting lives in the compiled `._native` module; this package just
re-exports it under a clean name.
"""

from ._native import Hnsw

__all__ = ["Hnsw"]
