"""Offline unit tests for app/evaluation/judge.py (Phase 9).

All tests are fully offline — they mock the AsyncOpenAI client and never
make real network calls. These tests exercise:
- JudgeScore dataclass construction and serialization
- LLMJudge initialization with/without GROQ_API_KEY
- Successful scoring with valid JSON response
- Graceful handling of malformed JSON, timeout, and unexpected errors
- Batch scoring via score_batch() with concurrency control
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.evaluation.judge import JUDGE_PROMPT_TEMPLATE, JudgeScore, LLMJudge


# ---------------------------------------------------------------------------
# JudgeScore dataclass tests
# ---------------------------------------------------------------------------


class TestJudgeScore:
    def test_defaults(self):
        score = JudgeScore(question_id="q01")
        assert score.correctness == 0.0
        assert score.groundedness == 0.0
        assert score.unsupported_claims == []
        assert score.notes == ""
        assert score.error is None

    def test_as_dict_includes_all_fields(self):
        score = JudgeScore(
            question_id="q01",
            correctness=1.0,
            groundedness=0.5,
            unsupported_claims=["claim A"],
            notes="Looks good",
            error=None,
        )
        d = score.as_dict()
        assert d["question_id"] == "q01"
        assert d["correctness"] == 1.0
        assert d["groundedness"] == 0.5
        assert d["unsupported_claims"] == ["claim A"]
        assert d["notes"] == "Looks good"
        assert d["error"] is None

    def test_as_dict_with_error(self):
        score = JudgeScore(question_id="q99", error="Timed out")
        d = score.as_dict()
        assert d["error"] == "Timed out"
        assert d["correctness"] == 0.0


# ---------------------------------------------------------------------------
# LLMJudge initialization tests
# ---------------------------------------------------------------------------


class TestLLMJudgeInit:
    def test_no_api_key_sets_client_none(self, fake_settings_no_groq):
        judge = LLMJudge(settings=fake_settings_no_groq)
        assert judge._client is None
        assert not judge.is_available

    def test_with_api_key_creates_client(self, fake_settings_with_groq):
        with patch("app.evaluation.judge.AsyncOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            judge = LLMJudge(settings=fake_settings_with_groq)
            assert judge.is_available
            mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# LLMJudge.score() — success path
# ---------------------------------------------------------------------------


class TestLLMJudgeScore:
    @pytest.fixture
    def judge_with_mock_client(self, fake_settings_with_groq):
        """Return an LLMJudge with a fully mocked AsyncOpenAI client."""
        with patch("app.evaluation.judge.AsyncOpenAI"):
            judge = LLMJudge(settings=fake_settings_with_groq)
        # Manually inject mock client
        mock_client = MagicMock()
        judge._client = mock_client
        return judge

    async def test_score_returns_valid_result(self, judge_with_mock_client):
        judge = judge_with_mock_client
        payload = {
            "correctness": 1.0,
            "groundedness": 1.0,
            "unsupported_claims": [],
            "notes": "All facts present.",
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(payload)

        async def fake_create(**kwargs):
            return mock_response

        judge._client.chat = MagicMock()
        judge._client.chat.completions = MagicMock()
        judge._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await judge.score(
            question_id="q01",
            question="What is the throughput of PKG-120?",
            answerable=True,
            reference_answer="18 cartons/min",
            must_include=["18 cartons/min"],
            context="PKG-120 throughput: 18 ctn/min.",
            answer="The PKG-120 has a throughput of 18 cartons/min [S1].",
        )

        assert result.question_id == "q01"
        assert result.correctness == 1.0
        assert result.groundedness == 1.0
        assert result.notes == "All facts present."
        assert result.error is None

    async def test_score_strips_markdown_fences(self, judge_with_mock_client):
        judge = judge_with_mock_client
        payload = '```json\n{"correctness": 0.5, "groundedness": 1.0, "unsupported_claims": [], "notes": "partial"}\n```'
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = payload
        judge._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await judge.score(
            question_id="q02",
            question="Some question",
            answerable=True,
            reference_answer="ref",
            must_include=[],
            context="ctx",
            answer="ans",
        )

        assert result.correctness == 0.5
        assert result.error is None

    async def test_score_no_client_returns_error(self, fake_settings_no_groq):
        judge = LLMJudge(settings=fake_settings_no_groq)
        result = await judge.score(
            question_id="q03",
            question="Any question",
            answerable=True,
            reference_answer="ref",
            must_include=[],
            context="ctx",
            answer="ans",
        )
        assert result.error is not None
        assert "GROQ_API_KEY" in result.error

    async def test_score_json_parse_error_returns_error(self, judge_with_mock_client):
        judge = judge_with_mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is NOT JSON"
        judge._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await judge.score(
            question_id="q04",
            question="Some question",
            answerable=True,
            reference_answer="ref",
            must_include=[],
            context="ctx",
            answer="ans",
        )

        assert result.error is not None
        assert "JSON parse error" in result.error

    async def test_score_timeout_returns_error(self, judge_with_mock_client):
        judge = judge_with_mock_client
        judge._client.chat.completions.create = AsyncMock(
            side_effect=asyncio.TimeoutError
        )

        result = await judge.score(
            question_id="q05",
            question="Any question",
            answerable=False,
            reference_answer="Not documented.",
            must_include=[],
            context="",
            answer="Not documented.",
        )

        assert result.error is not None
        assert "Timed out" in result.error or "timed out" in result.error

    async def test_score_unanswerable_question(self, judge_with_mock_client):
        """Confirm judge can score unanswerable questions."""
        judge = judge_with_mock_client
        payload = {
            "correctness": 1.0,
            "groundedness": 1.0,
            "unsupported_claims": [],
            "notes": "Assistant correctly refused to answer.",
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(payload)
        judge._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await judge.score(
            question_id="q25",
            question="What is the range of SkyDrone 500?",
            answerable=False,
            reference_answer="Not documented in the provided knowledge base.",
            must_include=[],
            context="",
            answer="Not documented in the provided knowledge base.",
        )

        assert result.correctness == 1.0


# ---------------------------------------------------------------------------
# LLMJudge.score_batch() tests
# ---------------------------------------------------------------------------


class TestLLMJudgeScoreBatch:
    async def test_batch_returns_same_count(self, fake_settings_with_groq):
        with patch("app.evaluation.judge.AsyncOpenAI"):
            judge = LLMJudge(settings=fake_settings_with_groq)

        good_payload = json.dumps(
            {"correctness": 1.0, "groundedness": 1.0, "unsupported_claims": [], "notes": "ok"}
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = good_payload
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        judge._client = mock_client

        items = [
            {
                "question_id": f"q{i:02d}",
                "question": f"Question {i}",
                "answerable": True,
                "reference_answer": "ref",
                "must_include": [],
                "context": "ctx",
                "answer": "ans",
            }
            for i in range(5)
        ]

        results = await judge.score_batch(items, concurrency=2)
        assert len(results) == 5
        for r in results:
            assert r.error is None
            assert r.correctness == 1.0

    async def test_batch_empty_input(self, fake_settings_no_groq):
        judge = LLMJudge(settings=fake_settings_no_groq)
        results = await judge.score_batch([], concurrency=3)
        assert results == []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_settings_no_groq():
    """Settings with no Groq key."""
    from app.core.config import Settings

    return Settings(
        GROQ_API_KEY="",
        GEMINI_API_KEY="",
        QDRANT_MODE="memory",
        LLM_MODE="retrieval_only",
    )


@pytest.fixture
def fake_settings_with_groq():
    """Settings with a fake Groq key (no real API calls made in tests)."""
    from app.core.config import Settings

    return Settings(
        GROQ_API_KEY="gsk_fake_test_key_for_unit_tests_only",
        GEMINI_API_KEY="",
        QDRANT_MODE="memory",
        LLM_MODE="retrieval_only",
    )
