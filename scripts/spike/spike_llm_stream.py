"""Spike: LLM streaming verification via OpenAI-compatible API.
Validates OpenAI client compatibility with Google's Gemini endpoint and Groq endpoint.
Confirms streaming token intervals and zero-temperature behavior.
Handles missing or placeholder credentials gracefully without unhandled exceptions.
"""

import asyncio
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI


def read_env() -> None:
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)


async def test_provider_stream(
    provider_name: str,
    base_url: str,
    api_key: str,
    model: str,
) -> bool:
    print(f"\n--- Testing Stream for {provider_name} ({model}) ---")
    placeholders = {
        "",
        "your_key_here",
        "your_gemini_api_key_here",
        "your_groq_api_key_here",
    }
    if not api_key or api_key.strip() in placeholders:
        print(f"SKIPPED: {provider_name} API key not configured or contains placeholder in .env")
        return False

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {
            "role": "system",
            "content": "You are a concise technical assistant. Respond in one sentence.",
        },
        {"role": "user", "content": "What is the primary function of an industrial carton sealer?"},
    ]

    try:
        start_time = time.perf_counter()
        first_token_time = None
        collected_tokens = []

        response_stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=60,
            stream=True,
        )

        async for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                token = chunk.choices[0].delta.content
                collected_tokens.append(token)
                print(token, end="", flush=True)

        total_time = time.perf_counter() - start_time
        print()
        if first_token_time:
            ttft = (first_token_time - start_time) * 1000
            print(
                f"Success! TTFT: {ttft:.1f}ms | Total latency: {total_time * 1000:.1f}ms | "
                f"Tokens: {len(collected_tokens)}"
            )
            return True
        else:
            print("Warning: Stream completed without any text tokens.")
            return False

    except Exception as exc:
        print(f"Error testing {provider_name}: {type(exc).__name__}: {exc}")
        return False


async def main() -> None:
    read_env()

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    gemini_base = os.getenv(
        "GEMINI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

    groq_key = os.getenv("GROQ_API_KEY", "")
    groq_base = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    gemini_ok = await test_provider_stream("Gemini", gemini_base, gemini_key, gemini_model)
    groq_ok = await test_provider_stream("Groq", groq_base, groq_key, groq_model)

    print("\nSummary:")
    print(f"  Gemini status: {'AVAILABLE' if gemini_ok else 'UNAVAILABLE / SKIPPED'}")
    print(f"  Groq status:   {'AVAILABLE' if groq_ok else 'UNAVAILABLE / SKIPPED'}")
    print("Spike script completed.")


if __name__ == "__main__":
    asyncio.run(main())
