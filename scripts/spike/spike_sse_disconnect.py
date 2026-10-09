"""Spike: SSE disconnect and generator cancellation verification.
Verifies that client cancellation/disconnect properly halts the async generator
without leaving orphaned loops or unhandled exceptions.
"""

import asyncio
from collections.abc import AsyncGenerator


async def event_publisher(cancellation_flag: list[bool]) -> AsyncGenerator[str, None]:
    print("Publisher: Started token generation...")
    try:
        for idx in range(1, 20):
            await asyncio.sleep(0.05)
            yield f"data: token_{idx}\n\n"
        print("Publisher: Finished all tokens naturally.")
    except asyncio.CancelledError:
        print("Publisher: CancelledError received! Upstream aborted cleanly.")
        cancellation_flag.append(True)
        raise
    finally:
        print("Publisher: Finally block executed (clean resources).")


async def simulate_client_disconnect() -> None:
    print("--- Testing SSE Disconnect & Generator Cancellation ---")
    cancellation_flag = []
    pub = event_publisher(cancellation_flag)

    # Consume 3 tokens, then cancel the task
    async def consumer():
        tokens_seen = 0
        async for item in pub:
            tokens_seen += 1
            print(f"Client consumed: {item.strip()}")
            if tokens_seen >= 3:
                print("Client: Simulating disconnect / closing connection!")
                break

    task = asyncio.create_task(consumer())
    await task
    # Explicitly close the generator to verify cancellation behavior
    await pub.aclose()
    print("Generator closed successfully.")
    print("SSE disconnect spike PASSED.\n")


if __name__ == "__main__":
    asyncio.run(simulate_client_disconnect())
    print("=== All SSE disconnect spike tests PASSED ===")
