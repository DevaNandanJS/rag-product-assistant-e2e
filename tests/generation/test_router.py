"""Offline unit tests for LLMRouter priority failover and cooldown state machine."""

import pytest

from app.core.errors import LLMTimeoutError, LLMUnavailableError
from app.generation.router import LLMRouter


class FakeRateLimitError(Exception):
    """Exception with RateLimitError in its name to test 429 handling."""
    pass


class FakeLLMProvider:
    """Mock LLM provider yielding tokens from an in-memory list."""

    def __init__(self, name: str, model: str = "test-model", tokens: list[str] | None = None) -> None:
        self._name = name
        self._model = model
        self.tokens = tokens if tokens is not None else ["token1 ", "token2"]
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    async def stream(self, messages, max_tokens, temperature):
        self.call_count += 1
        for t in self.tokens:
            yield t


class FakeFailingProvider:
    """Mock LLM provider that raises an exception before or during streaming."""

    def __init__(
        self,
        name: str,
        error_to_raise: Exception,
        yield_tokens_before_error: list[str] | None = None,
        model: str = "test-model",
    ) -> None:
        self._name = name
        self._model = model
        self.error_to_raise = error_to_raise
        self.yield_tokens_before_error = yield_tokens_before_error or []
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    async def stream(self, messages, max_tokens, temperature):
        self.call_count += 1
        for t in self.yield_tokens_before_error:
            yield t
        raise self.error_to_raise


@pytest.mark.asyncio
async def test_retrieval_only_yields_nothing():
    provider = FakeLLMProvider("gemini")
    router = LLMRouter(providers=[provider], mode="retrieval_only")
    emitted = []
    async for token in router.stream(messages=[], max_tokens=100, temperature=0.1):
        emitted.append(token)
    assert len(emitted) == 0
    assert provider.call_count == 0


@pytest.mark.asyncio
async def test_failover_on_rate_limit():
    p1 = FakeFailingProvider("gemini", error_to_raise=FakeRateLimitError("Rate limit hit"))
    p2 = FakeLLMProvider("groq", tokens=["hello ", "world"])
    router = LLMRouter(providers=[p1, p2], mode="live")

    emitted = []
    async for token in router.stream(messages=[], max_tokens=100, temperature=0.1):
        emitted.append(token)

    assert p1.call_count == 1
    assert p2.call_count == 1
    assert "".join(emitted) == "hello world"
    # Cooldown should be active on p1
    assert router._is_on_cooldown("gemini")


@pytest.mark.asyncio
async def test_cooldown_skips_provider():
    p1 = FakeLLMProvider("gemini")
    p2 = FakeLLMProvider("groq", tokens=["from ", "groq"])
    router = LLMRouter(providers=[p1, p2], mode="live")

    # Manually place p1 on cooldown
    router._set_cooldown("gemini")

    emitted = []
    async for token in router.stream(messages=[], max_tokens=100, temperature=0.1):
        emitted.append(token)

    assert p1.call_count == 0  # Skipped entirely
    assert p2.call_count == 1
    assert "".join(emitted) == "from groq"


@pytest.mark.asyncio
async def test_no_failover_after_first_token():
    # p1 emits a token, then times out
    p1 = FakeFailingProvider(
        "gemini",
        error_to_raise=LLMTimeoutError("Idle token timeout"),
        yield_tokens_before_error=["initial_token "],
    )
    p2 = FakeLLMProvider("groq", tokens=["should ", "not ", "run"])
    router = LLMRouter(providers=[p1, p2], mode="live")

    emitted = []
    with pytest.raises(LLMTimeoutError):
        async for token in router.stream(messages=[], max_tokens=100, temperature=0.1):
            emitted.append(token)

    # p1 emitted initial token, but then failed and error propagated without calling p2
    assert emitted == ["initial_token "]
    assert p1.call_count == 1
    assert p2.call_count == 0


@pytest.mark.asyncio
async def test_all_providers_exhausted_raises():
    p1 = FakeFailingProvider("gemini", error_to_raise=LLMTimeoutError("Timeout"))
    p2 = FakeFailingProvider("groq", error_to_raise=FakeRateLimitError("429"))
    router = LLMRouter(providers=[p1, p2], mode="live")

    with pytest.raises(LLMUnavailableError) as exc_info:
        async for _ in router.stream(messages=[], max_tokens=100, temperature=0.1):
            pass

    assert "All configured LLM providers" in str(exc_info.value)
