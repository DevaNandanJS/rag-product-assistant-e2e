"""Offline unit tests for GenerationService orchestrator and event lifecycle."""

import pytest

from app.core.config import Settings
from app.core.schemas import AskRequest, Chunk
from app.generation.cache import GenerationCache
from app.generation.router import LLMRouter
from app.generation.service import GenerationService
from app.retrieval.gate import GateResult
from app.retrieval.pipeline import RetrievalResult


class FakeRetrievalPipeline:
    """Mock RetrievalPipeline returning predictable RetrievalResults."""

    def __init__(self, passed: bool = True, chunks: list[Chunk] | None = None) -> None:
        self.passed = passed
        self.chunks = chunks or []

    def retrieve(self, question: str, **kwargs) -> RetrievalResult:
        gate = GateResult(
            passed=self.passed,
            score=0.75 if self.passed else 0.20,
            threshold=0.45,
            signal="dense_cosine",
            reason="Score above threshold" if self.passed else "Relevance score below threshold",
        )
        return RetrievalResult(
            chunks=self.chunks,
            chunk_scores=[(c, 0.75) for c in self.chunks],
            gate=gate,
            entities=["PKG-120"],
            is_comparison=False,
            debug_info={"dense_ranks": [{"chunk_id": "PKG-120_CARD", "score": 0.75}]},
        )


class FakeProvider:
    def __init__(self, name: str = "mock-gemini", model: str = "gemini-3.1-flash-lite"):
        self.name = name
        self.model = model

    async def stream(self, messages, max_tokens, temperature):
        yield "CartonPro 1200 uses 48-72 mm tape "
        yield "[S1]."


def _make_chunk() -> Chunk:
    return Chunk(
        chunk_id="PKG-120_CARD",
        point_id="00000000-0000-0000-0000-000000000001",
        text="CartonPro 1200. Tape: 48-72 mm.",
        display_text="CartonPro 1200. Tape: 48-72 mm.",
        chunk_type="card",
        product_id="PKG-120",
        product_name="CartonPro 1200",
        document="catalog.json",
        page=1,
        source_type="structured",
        token_count=30,
    )


@pytest.mark.asyncio
async def test_generation_lifecycle_success(tmp_path):
    settings = Settings(LLM_MODE="live")
    chunks = [_make_chunk()]
    pipeline = FakeRetrievalPipeline(passed=True, chunks=chunks)
    router = LLMRouter(providers=[FakeProvider()], mode="live")
    cache = GenerationCache(tmp_path / "cache.jsonl", enabled=True)
    service = GenerationService(settings=settings, pipeline=pipeline, router=router, cache=cache)

    request = AskRequest(question="What tape width does CartonPro 1200 use?", debug=True)
    events = []
    async for ev in service.generate(request):
        events.append(ev)

    event_types = [e.type for e in events]
    assert "meta" in event_types
    assert "debug" in event_types
    assert "token" in event_types
    assert "sources" in event_types
    assert "grounding" in event_types
    assert "done" in event_types

    # Verify sources event content
    sources_ev = next(e for e in events if e.type == "sources")
    sources_list = sources_ev.payload["sources"]
    assert len(sources_list) == 1
    assert sources_list[0]["id"] == "S1"
    assert sources_list[0]["cited"] is True

    # Verify grounding event
    grounding_ev = next(e for e in events if e.type == "grounding")
    assert grounding_ev.payload["ok"] is True
    assert grounding_ev.payload["warnings"] == []

    # Verify done event
    done_ev = next(e for e in events if e.type == "done")
    assert done_ev.payload["status"] == "ok"


@pytest.mark.asyncio
async def test_gate_rejection_canned_refusal():
    settings = Settings(LLM_MODE="live")
    pipeline = FakeRetrievalPipeline(passed=False, chunks=[])
    router = LLMRouter(providers=[FakeProvider()], mode="live")
    service = GenerationService(settings=settings, pipeline=pipeline, router=router)

    request = AskRequest(question="What is the population of Mars?")
    events = []
    async for ev in service.generate(request):
        events.append(ev)

    event_types = [e.type for e in events]
    assert event_types == ["meta", "token", "sources", "grounding", "done"]

    token_ev = next(e for e in events if e.type == "token")
    assert "Not documented" in token_ev.payload["content"]

    done_ev = next(e for e in events if e.type == "done")
    assert done_ev.payload["status"] == "gated"


@pytest.mark.asyncio
async def test_retrieval_only_mode():
    settings = Settings(LLM_MODE="retrieval_only")
    chunks = [_make_chunk()]
    pipeline = FakeRetrievalPipeline(passed=True, chunks=chunks)
    router = LLMRouter(providers=[], mode="retrieval_only")
    service = GenerationService(settings=settings, pipeline=pipeline, router=router)

    request = AskRequest(question="What tape width does CartonPro 1200 use?")
    events = []
    async for ev in service.generate(request):
        events.append(ev)

    event_types = [e.type for e in events]
    assert "token" in event_types
    token_ev = next(e for e in events if e.type == "token")
    assert "[Retrieval-only mode]" in token_ev.payload["content"]
