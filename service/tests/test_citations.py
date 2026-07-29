"""Tests for citation-marker parsing.

The `cited` flag in the UI is only meaningful if it reflects what the answer
actually referenced, so the mapping from `[n]` markers to chunk ids is worth
testing directly.
"""

from rag_service.generation import parse_citations
from rag_service.store import RetrievedChunk


def chunk(cid: int) -> RetrievedChunk:
    return RetrievedChunk(
        id=cid, text="…", doc_id=0, doc_title="doc", ordinal=0, score=0.5
    )


CHUNKS = [chunk(10), chunk(11), chunk(12)]


def test_maps_markers_to_chunk_ids():
    # [1] and [3] are 1-indexed into the sources list.
    assert parse_citations("Yes [1], and also [3].", CHUNKS) == [10, 12]


def test_no_markers_means_nothing_cited():
    assert parse_citations("I could not find an answer.", CHUNKS) == []


def test_out_of_range_markers_ignored():
    # The model inventing [9] for three sources must not crash or be trusted.
    assert parse_citations("See [9] and [2].", CHUNKS) == [11]
    assert parse_citations("See [0].", CHUNKS) == []


def test_deduplicated_in_first_mention_order():
    assert parse_citations("[2] then [1] then [2] again.", CHUNKS) == [11, 10]


def test_empty_chunk_list():
    assert parse_citations("Nothing indexed [1].", []) == []
