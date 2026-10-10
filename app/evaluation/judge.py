"""LLM-as-a-Judge evaluation runner using Groq Llama-3.3-70B.

Evaluates generated answers on two independent axes:
  - correctness: Are all mandatory facts present and accurate?
  - groundedness: Are all claims strictly supported by retrieved context?

Per Section 8 (Phase 9) of BUILD_GUIDE.md, the judge model deliberately
differs from the generator model (Gemini) to prevent self-evaluation bias.
Generator = gemini-3.1-flash-lite (Gemini), Judge = llama-3.3-70b-versatile (Groq).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from app.core.config import Settings

logger = logging.getLogger(__name__)

# Judge prompt from Section 9.5 of BUILD_GUIDE.md
JUDGE_PROMPT_TEMPLATE = """\
You are an impartial evaluator auditing an answer from a B2B product knowledge assistant.

EVALUATION INPUTS:
- QUESTION: {question}
- ANSWERABLE: {answerable}
- REFERENCE ANSWER: {reference_answer}
- MANDATORY FACTS: {must_include}
- RETRIEVED CONTEXT: {context}
- ASSISTANT ANSWER: {answer}

SCORING CRITERIA:
1. correctness:
   - 1.0: All MANDATORY FACTS are present and accurate without contradictions.
   - 0.5: Partially correct, minor factual omissions, or slight numerical rounding.
   - 0.0: Factual errors, major omissions, or answering an unanswerable query.
   - For unanswerable queries (ANSWERABLE=false): score 1.0 ONLY if the assistant
     explicitly states the information is not documented; score 0.0 if it hallucinates
     an answer.
2. groundedness:
   - 1.0: Every factual claim is strictly supported by the RETRIEVED CONTEXT.
   - 0.0: Assistant claims facts not present in the RETRIEVED CONTEXT.

OUTPUT FORMAT (Valid JSON only, no markdown fences):
{{"correctness": 1.0, "groundedness": 1.0, "unsupported_claims": [], "notes": "Clear, grounded response."}}
"""


@dataclass
class JudgeScore:
    """Result of a single LLM-as-a-judge evaluation."""

    question_id: str
    correctness: float = 0.0
    groundedness: float = 0.0
    unsupported_claims: list[str] = field(default_factory=list)
    notes: str = ""
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "correctness": self.correctness,
            "groundedness": self.groundedness,
            "unsupported_claims": self.unsupported_claims,
            "notes": self.notes,
            "error": self.error,
        }


class LLMJudge:
    """Async LLM-as-a-Judge runner using Groq Llama-3.3-70B.

    Uses a different provider (Groq) from the generator (Gemini) to prevent
    self-evaluation bias per Phase 9 study notes.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncOpenAI | None = None
        if settings.GROQ_API_KEY and settings.GROQ_API_KEY not in ("", "your_key_here"):
            self._client = AsyncOpenAI(
                api_key=settings.GROQ_API_KEY,
                base_url=settings.GROQ_BASE_URL,
                timeout=60.0,
            )
            logger.info(
                "[judge] Initialized with model '%s' via Groq", settings.JUDGE_MODEL
            )
        else:
            logger.warning(
                "[judge] GROQ_API_KEY not configured; judge scoring will be skipped."
            )

    @property
    def is_available(self) -> bool:
        """True if the judge client is configured and ready."""
        return self._client is not None

    async def score(
        self,
        *,
        question_id: str,
        question: str,
        answerable: bool,
        reference_answer: str,
        must_include: list[str],
        context: str,
        answer: str,
    ) -> JudgeScore:
        """Score a single (question, answer) pair on correctness and groundedness.

        Args:
            question_id: Unique identifier for the question (e.g. 'q01').
            question: The original user question.
            answerable: Whether the question is answerable from the knowledge base.
            reference_answer: Human-authored reference answer.
            must_include: List of mandatory fact strings that must appear.
            context: Concatenated retrieved passage text (truncated to 3000 chars).
            answer: The assistant's generated answer to evaluate.

        Returns:
            JudgeScore with correctness, groundedness, unsupported_claims, notes.
        """
        if self._client is None:
            return JudgeScore(
                question_id=question_id,
                correctness=0.0,
                groundedness=0.0,
                error="GROQ_API_KEY not configured; judge skipped.",
            )

        # Truncate context to stay within token budget (judge prompt + answer ~600 tokens)
        context_trimmed = context[:3000] if context else "No context retrieved."
        must_include_str = (
            ", ".join(must_include) if must_include else "None specified"
        )

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            answerable=str(answerable),
            reference_answer=reference_answer,
            must_include=must_include_str,
            context=context_trimmed,
            answer=answer if answer else "[No answer generated]",
        )

        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._settings.JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=300,
                ),
                timeout=45.0,
            )
            raw = response.choices[0].message.content or "{}"
            # Strip potential markdown fences from raw output
            raw = raw.strip()
            if raw.startswith("```"):
                lines = raw.splitlines()
                # Remove first and last fence lines
                raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            raw = raw.strip()

            data = json.loads(raw)
            return JudgeScore(
                question_id=question_id,
                correctness=float(data.get("correctness", 0.0)),
                groundedness=float(data.get("groundedness", 0.0)),
                unsupported_claims=list(data.get("unsupported_claims", [])),
                notes=str(data.get("notes", "")),
            )
        except asyncio.TimeoutError:
            logger.warning("[judge] Timeout scoring question %s", question_id)
            return JudgeScore(
                question_id=question_id,
                error="LLM judge timed out after 45s",
            )
        except json.JSONDecodeError as exc:
            logger.warning(
                "[judge] JSON parse error for %s: %s | raw=%r", question_id, exc, raw
            )
            return JudgeScore(
                question_id=question_id,
                error=f"JSON parse error: {exc}",
            )
        except Exception as exc:
            logger.error("[judge] Unexpected error scoring %s: %s", question_id, exc)
            return JudgeScore(
                question_id=question_id,
                error=str(exc),
            )

    async def score_batch(
        self,
        items: list[dict[str, object]],
        concurrency: int = 3,
    ) -> list[JudgeScore]:
        """Score a batch of questions concurrently with a semaphore.

        Args:
            items: List of dicts with keys matching `score()` keyword args
                   plus `question_id`.
            concurrency: Maximum parallel judge calls (default 3 to respect rate limits).

        Returns:
            List of JudgeScore in the same order as input items.
        """
        sem = asyncio.Semaphore(concurrency)

        async def _limited_score(item: dict[str, object]) -> JudgeScore:
            async with sem:
                return await self.score(
                    question_id=str(item["question_id"]),
                    question=str(item["question"]),
                    answerable=bool(item.get("answerable", True)),
                    reference_answer=str(item.get("reference_answer", "")),
                    must_include=list(item.get("must_include", [])),
                    context=str(item.get("context", "")),
                    answer=str(item.get("answer", "")),
                )

        return list(await asyncio.gather(*[_limited_score(it) for it in items]))
