import sys
from types import SimpleNamespace

import pytest

from rag_service.generation import GenerationTimeoutError, generate_answer
from rag_service.store import RetrievedChunk


def test_anthropic_timeout_disables_retries_and_never_falls_back(monkeypatch):
    captured = {}

    class FakeAPITimeoutError(Exception):
        pass

    class FakeMessages:
        def create(self, **_kwargs):
            raise FakeAPITimeoutError

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.messages = FakeMessages()

    fake_anthropic = SimpleNamespace(
        Anthropic=FakeAnthropic,
        APITimeoutError=FakeAPITimeoutError,
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("RAG_CLAUDE_TIMEOUT_SECONDS", raising=False)

    chunk = RetrievedChunk(
        id=7,
        text="grounded context",
        doc_id=3,
        doc_title="source",
        ordinal=0,
        score=1.0,
    )

    with pytest.raises(GenerationTimeoutError) as error:
        generate_answer("question", [chunk])

    assert str(error.value) == "Claude generation timed out after 30 seconds"
    assert captured == {"timeout": 30.0, "max_retries": 0}
