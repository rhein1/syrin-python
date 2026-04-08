"""Process-verify an Agoragentic workflow with checkpoints and trace inspection.

Demonstrates:
  - attaching tool hooks to audit which third-party tools ran
  - checkpointing around a marketplace-assisted workflow
  - inspecting `response.trace` for process-level verification
  - comparing actual tool usage against expected workflow checkpoints

Requires:
  - `OPENAI_API_KEY` to let the agent decide which tools to call
  - `AGORAGENTIC_API_KEY` only if you want live marketplace responses instead of skipped payloads

Run:
    uv run python examples/thirdparty/agroagentic/agoragentic_marketplace_process_verification.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from examples.thirdparty.agroagentic.agoragentic_marketplace import (
    agoragentic_execute,
    agoragentic_match,
    agoragentic_memory_search,
    agoragentic_save_learning_note,
)
from syrin import Agent, Budget, CheckpointConfig, CheckpointTrigger, Model
from syrin.enums import ExceedPolicy


def _build_model() -> Model:
    """Create a tool-calling model when configured, otherwise a fast mock model."""
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        return Model.OpenAI("gpt-4o-mini", api_key=openai_key)
    return Model.mock(latency_min=0, latency_max=0)


def _build_agent(checkpoint_path: str) -> Agent:
    """Create a checkpointed agent that reuses the third-party marketplace tools."""
    return Agent(
        model=_build_model(),
        budget=Budget(max_cost=1.00, exceed_policy=ExceedPolicy.STOP),
        checkpoint=CheckpointConfig(
            storage="sqlite",
            path=checkpoint_path,
            trigger=CheckpointTrigger.TOOL,
            max_checkpoints=10,
        ),
        system_prompt=(
            "You are a marketplace-native research agent. Use agoragentic_match before "
            "agoragentic_execute when fit is unclear. Search marketplace memory before "
            "repeating prior work. Save a learning note only when you discover a durable "
            "workflow lesson worth reusing later."
        ),
        tools=[
            agoragentic_match,
            agoragentic_execute,
            agoragentic_memory_search,
            agoragentic_save_learning_note,
        ],
    )


def main() -> None:
    """Run a small process-verification pass against the Agoragentic tool workflow."""
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "agoragentic_process.sqlite"
        agent = _build_agent(str(checkpoint_path))
        tool_events: list[str] = []

        agent.events.on_tool(lambda ctx: tool_events.append(ctx.get("tool_name", "")))

        print("Configured tools:", [tool_spec.name for tool_spec in agent.tools])
        print(f"Checkpoint store: {checkpoint_path}")

        if not os.getenv("OPENAI_API_KEY", "").strip():
            print("OPENAI_API_KEY not set; configure it to verify tool-calling behavior via agent.run().")
            return

        before_id = agent.save_checkpoint("before_process_verification")
        print(f"Saved baseline checkpoint: {before_id}")

        result = agent.run(
            "Use agoragentic_match to preview a marketplace provider for summarizing technical "
            "papers under $0.25. Then use agoragentic_memory_search to look for prior "
            "summarization workflow notes. Do not execute paid actions or save new notes."
        )

        after_id = agent.save_checkpoint("after_process_verification")
        print(f"Saved post-run checkpoint: {after_id}")

        expected_tools = [
            "agoragentic_match",
            "agoragentic_memory_search",
        ]
        actual_tools = [tool_name for tool_name in tool_events if tool_name]
        missing_tools = [tool_name for tool_name in expected_tools if tool_name not in actual_tools]
        unexpected_tools = [tool_name for tool_name in actual_tools if tool_name not in expected_tools]

        print("\n=== Verification summary ===")
        print("Expected tools:", expected_tools)
        print("Observed tools:", actual_tools)
        print("Missing tools:", missing_tools or "none")
        print("Unexpected tools:", unexpected_tools or "none")
        print("Checkpoints:", agent.list_checkpoints())

        print("\n=== Trace summary ===")
        print(f"Trace steps: {len(result.trace)}")
        for index, step in enumerate(result.trace, start=1):
            step_type = getattr(step, "step_type", "unknown")
            latency_ms = getattr(step, "latency_ms", 0.0)
            cost_usd = getattr(step, "cost_usd", 0.0)
            print(f"  Step {index}: {step_type}, latency={latency_ms:.1f}ms, cost=${cost_usd:.6f}")

        print("\n=== Agent output ===")
        print(result.content[:1200])
        print(f"\nTotal cost: ${result.cost:.6f}")


if __name__ == "__main__":
    main()
