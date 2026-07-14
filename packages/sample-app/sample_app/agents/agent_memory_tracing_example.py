"""
Agent Memory Retrieval Tracing Example
=======================================

This example demonstrates how to trace memory read/write operations in an agent
that uses an external memory store. It uses OpenTelemetry spans to track:

  - Memory write latency (storing a fact during a session)
  - Memory read latency and hit/miss outcome (retrieving facts on future turns)
  - How retrieved memories influence the final LLM response

This pattern is useful when your agent uses a retrieval-augmented memory layer
(e.g., Mem0, custom vector store, Redis, or a simple dict-based store) and you
want to measure retrieval quality and latency separately from LLM inference.

Usage:
  python -m sample_app.agents.agent_memory_tracing_example

Dependencies (already in sample-app requirements):
  opentelemetry-api, traceloop-sdk, openai
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from opentelemetry import trace
from traceloop.sdk import Traceloop

load_dotenv()

Traceloop.init(
    app_name="agent-memory-tracing-demo",
    disable_batch=True,  # flush immediately so spans appear right away in dev
)

tracer = trace.get_tracer(__name__)

# ---------------------------------------------------------------------------
# Minimal in-process memory store (replace with your vector DB / Mem0 / etc.)
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    key: str
    value: str
    created_at: float = field(default_factory=time.time)


class SimpleMemoryStore:
    """
    A trivial key-value store that simulates an external memory backend.
    Wrap the read/write calls with OpenTelemetry spans so you can measure
    latency and hit/miss rates across sessions.
    """

    def __init__(self) -> None:
        self._store: dict[str, MemoryEntry] = {}

    def write(self, key: str, value: str) -> None:
        """
        Store a memory entry and emit a span with write metadata.

        Span attributes:
          memory.operation  = "write"
          memory.key        = the lookup key
          memory.value_len  = number of characters written
        """
        with tracer.start_as_current_span("memory.write") as span:
            span.set_attribute("memory.operation", "write")
            span.set_attribute("memory.key", key)
            span.set_attribute("memory.value_len", len(value))

            self._store[key] = MemoryEntry(key=key, value=value)

    def read(self, key: str) -> tuple[str | None, bool]:
        """
        Retrieve a memory entry and emit a span with hit/miss metadata.

        Returns:
          (value, hit) — value is None on a miss.

        Span attributes:
          memory.operation  = "read"
          memory.key        = the lookup key
          memory.hit        = True / False
          memory.age_s      = seconds since the entry was written (on hit only)
        """
        with tracer.start_as_current_span("memory.read") as span:
            span.set_attribute("memory.operation", "read")
            span.set_attribute("memory.key", key)

            entry = self._store.get(key)
            hit = entry is not None
            span.set_attribute("memory.hit", hit)

            if hit:
                age = time.time() - entry.created_at
                span.set_attribute("memory.age_s", round(age, 3))
                return entry.value, True

            return None, False


# ---------------------------------------------------------------------------
# Agent turn — a lightweight loop that does not require an agent framework
# ---------------------------------------------------------------------------

def run_agent_turn(
    memory: SimpleMemoryStore,
    user_message: str,
    session_id: str,
) -> str:
    """
    Simulate one agent turn:
      1. Read relevant memories before calling the LLM.
      2. Build a context-augmented prompt.
      3. Call OpenAI (gpt-4o-mini) with the augmented prompt.
      4. Write new facts learned from the response back to memory.

    The entire turn is wrapped in a parent span so you can correlate
    LLM latency, memory latency, and total turn latency in one trace.
    """
    import openai

    client = openai.OpenAI()

    with tracer.start_as_current_span("agent.turn") as turn_span:
        turn_span.set_attribute("session.id", session_id)
        turn_span.set_attribute("user.message", user_message[:200])

        # --- Memory read: retrieve user preferences stored in a previous turn ---
        preference_key = f"{session_id}:user_preference"
        recalled_preference, hit = memory.read(preference_key)

        system_context = "You are a helpful assistant."
        if hit and recalled_preference:
            system_context += f" The user previously mentioned: {recalled_preference}"
            turn_span.set_attribute("memory.context_injected", True)
        else:
            turn_span.set_attribute("memory.context_injected", False)

        # --- LLM call (instrumented automatically by Traceloop) ---
        with tracer.start_as_current_span("llm.chat") as llm_span:
            llm_span.set_attribute("llm.model", "gpt-4o-mini")

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_context},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=256,
            )
            answer = response.choices[0].message.content or ""
            llm_span.set_attribute("llm.output_tokens", response.usage.completion_tokens)

        # --- Memory write: persist a preference extracted from this message ---
        # (In production you'd run an extraction step; here we keep it simple.)
        if "prefer" in user_message.lower() or "like" in user_message.lower():
            memory.write(preference_key, user_message[:120])
            turn_span.set_attribute("memory.new_fact_stored", True)
        else:
            turn_span.set_attribute("memory.new_fact_stored", False)

        turn_span.set_attribute("agent.response_len", len(answer))
        return answer


# ---------------------------------------------------------------------------
# Demo: two-turn session that demonstrates memory carry-over
# ---------------------------------------------------------------------------

def main() -> None:
    mem = SimpleMemoryStore()
    session_id = str(uuid.uuid4())

    print("=== Turn 1 — introduce a preference (will be stored in memory) ===")
    t1_msg = "I prefer concise bullet-point answers over long prose."
    t1_answer = run_agent_turn(mem, t1_msg, session_id)
    print(f"User:  {t1_msg}")
    print(f"Agent: {t1_answer}\n")

    print("=== Turn 2 — ask a question (memory hit should influence the answer) ===")
    t2_msg = "Summarise the benefits of OpenTelemetry for LLM applications."
    t2_answer = run_agent_turn(mem, t2_msg, session_id)
    print(f"User:  {t2_msg}")
    print(f"Agent: {t2_answer}\n")

    print("=== Turn 3 — new session, no memory carry-over (cold start) ===")
    cold_session_id = str(uuid.uuid4())
    t3_msg = "Summarise the benefits of OpenTelemetry for LLM applications."
    t3_answer = run_agent_turn(mem, t3_msg, cold_session_id)
    print(f"User:  {t3_msg}")
    print(f"Agent: {t3_answer}\n")

    print("Demo complete. Open your Traceloop dashboard to compare:")
    print("  - Turn 2: memory.hit=True  → context injected → likely bullet-point answer")
    print("  - Turn 3: memory.hit=False → no context       → default prose answer")
    print("Use the memory.hit and memory.age_s span attributes to track retrieval quality over time.")


if __name__ == "__main__":
    main()
