"""Serve an Agoragentic-backed Syrin agent over HTTP.

Demonstrates:
  - exposing Agoragentic's third-party marketplace tools through `agent.serve()`
  - preview-first routing with the same live-call gates as the CLI example
  - serving the standard Syrin `/chat`, `/stream`, `/health`, `/ready`, and `/describe` routes

Requires: uv pip install -e ".[serve,openai]"

Run:
    uv run python examples/thirdparty/agroagentic/agoragentic_marketplace_serve.py

Visit:
    http://localhost:8000/playground
"""

from __future__ import annotations

import os
import sys
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
from syrin import Agent, Budget, Model
from syrin.enums import ExceedPolicy


def _build_model() -> Model:
    """Create a real tool-calling model when configured, otherwise a fast mock model."""
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        return Model.OpenAI("gpt-4o-mini", api_key=openai_key)
    return Model.mock(latency_min=0, latency_max=0)


def _build_agent() -> Agent:
    """Create the HTTP-served Agoragentic example agent."""
    return Agent(
        model=_build_model(),
        budget=Budget(max_cost=1.00, exceed_policy=ExceedPolicy.STOP),
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
    """Start a playground-enabled HTTP server for the Agoragentic marketplace agent."""
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    agent = _build_agent()

    print("Serving at http://localhost:8000")
    print("Open http://localhost:8000/playground to inspect the agent and its tools.")

    if not os.getenv("OPENAI_API_KEY", "").strip():
        print("OPENAI_API_KEY not set; the server will start with Model.mock() for route inspection.")
    if not os.getenv("AGORAGENTIC_API_KEY", "").strip():
        print("AGORAGENTIC_API_KEY not set; marketplace tool calls will return skipped payloads.")
    if os.getenv("AGORAGENTIC_RUN_LIVE", "").strip() != "1":
        print("AGORAGENTIC_RUN_LIVE is not 1; paid execution and note writes remain disabled.")

    agent.serve(port=8000, enable_playground=True, debug=True)


if __name__ == "__main__":
    main()
