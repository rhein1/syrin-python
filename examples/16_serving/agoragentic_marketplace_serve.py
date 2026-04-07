"""Serve an Agoragentic-backed Syrin agent over HTTP.

Demonstrates:
- exposing marketplace-aware tools via `agent.serve()`
- previewing routed providers before paid execution
- keeping paid or mutating marketplace actions gated behind an env flag
- serving the standard Syrin `/chat`, `/stream`, `/health`, `/ready`, and `/describe` routes

Requires: uv pip install -e ".[serve,openai]"

Run: python examples/16_serving/agoragentic_marketplace_serve.py
Visit: http://localhost:8000/playground
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from syrin import Agent, Budget, Model, tool
from syrin.enums import ExceedPolicy

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_BASE_URL = os.getenv("AGORAGENTIC_BASE_URL", "https://agoragentic.com").rstrip("/")
_TIMEOUT = 20.0


def _headers() -> dict[str, str]:
    """Build marketplace request headers, including auth when configured."""
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("AGORAGENTIC_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _format(payload: dict[str, Any]) -> str:
    """Pretty-print tool payloads for readable agent responses."""
    return json.dumps(payload, indent=2, sort_keys=True)


def _require_marketplace_key() -> dict[str, Any] | None:
    """Return a standardized skip payload when the marketplace key is missing."""
    if os.getenv("AGORAGENTIC_API_KEY", "").strip():
        return None
    return {
        "status": "skipped",
        "error": "missing_api_key",
        "message": "Set AGORAGENTIC_API_KEY in examples/.env to run live marketplace calls.",
    }


def _live_enabled() -> bool:
    """Report whether paid or mutating marketplace calls are enabled."""
    return os.getenv("AGORAGENTIC_RUN_LIVE", "").strip() == "1"


def _live_guard(action: str) -> dict[str, Any]:
    """Return a standardized skip payload for live-only actions."""
    return {
        "status": "skipped",
        "error": "live_calls_disabled",
        "message": f"Set AGORAGENTIC_RUN_LIVE=1 to enable {action}.",
    }


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one marketplace request and normalize success and error payloads."""
    try:
        response = requests.request(
            method,
            f"{_BASE_URL}{path}",
            params=params,
            json=payload,
            headers=_headers(),
            timeout=_TIMEOUT,
        )
    except requests.RequestException as exc:
        return {
            "error": "request_failed",
            "message": str(exc),
        }

    try:
        data = response.json()
    except ValueError:
        data = {"message": response.text[:500] or f"HTTP {response.status_code}"}

    if response.status_code >= 400:
        return {
            "error": data.get("error") if isinstance(data, dict) else f"http_{response.status_code}",
            "message": data.get("message") if isinstance(data, dict) else str(data),
            "status_code": response.status_code,
        }

    return data if isinstance(data, dict) else {"data": data}


@tool
def agoragentic_match(task: str, max_cost: float = 0.25) -> str:
    """Preview which marketplace providers could handle a task."""
    missing = _require_marketplace_key()
    if missing:
        return _format(missing)

    data = _request(
        "GET",
        "/api/execute/match",
        params={"task": task, "max_cost": max_cost},
    )
    if data.get("error"):
        return _format(data)

    providers = []
    for provider in data.get("providers", [])[:5]:
        providers.append(
            {
                "name": provider.get("name"),
                "capability": provider.get("capability_name"),
                "price_usdc": provider.get("price"),
                "eligible": provider.get("eligible"),
                "score": (provider.get("score") or {}).get("composite"),
            }
        )

    return _format(
        {
            "task": data.get("task"),
            "matches": data.get("matches"),
            "eligible": data.get("eligible"),
            "providers": providers,
            "why_filtered": data.get("why_filtered"),
        }
    )


@tool
def agoragentic_execute(task: str, input_text: str = "", max_cost: float = 0.25) -> str:
    """Route a task to the marketplace and execute it."""
    missing = _require_marketplace_key()
    if missing:
        return _format(missing)
    if not _live_enabled():
        return _format(_live_guard("paid marketplace execution"))

    data = _request(
        "POST",
        "/api/execute",
        payload={
            "task": task,
            "input": {"text": input_text} if input_text else {},
            "constraints": {"max_cost": max_cost},
        },
    )
    if data.get("error"):
        return _format(data)

    provider = data.get("provider") or {}
    return _format(
        {
            "status": data.get("status"),
            "provider": provider.get("name"),
            "capability": provider.get("capability_name"),
            "cost_usdc": data.get("cost"),
            "invocation_id": data.get("invocation_id"),
            "output": data.get("output"),
        }
    )


@tool
def agoragentic_memory_search(query: str, namespace: str = "default", limit: int = 5) -> str:
    """Search external marketplace memory."""
    missing = _require_marketplace_key()
    if missing:
        return _format(missing)

    data = _request(
        "GET",
        "/api/vault/memory/search",
        params={"query": query, "namespace": namespace, "limit": limit},
    )
    return _format(data)


@tool
def agoragentic_save_learning_note(title: str, lesson: str) -> str:
    """Persist one reusable lesson back to marketplace memory."""
    missing = _require_marketplace_key()
    if missing:
        return _format(missing)
    if not _live_enabled():
        return _format(_live_guard("learning-note writes"))

    data = _request(
        "POST",
        "/api/agents/me/learning-notes",
        payload={"input": {"title": title, "lesson": lesson, "tags": ["syrin", "marketplace"]}},
    )
    return _format(data)


def _build_model() -> Model:
    """Create a real tool-calling model when configured, otherwise a fast mock model."""
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        return Model.OpenAI("gpt-4o-mini", api_key=openai_key)
    return Model.mock(latency_min=0, latency_max=0)


class MarketplaceServeAgent(Agent):
    """HTTP-served agent that can preview and route work through Agoragentic."""

    name = "agoragentic-marketplace"
    description = "Serve a marketplace-native research agent with preview-first external tools."
    model = _build_model()
    budget = Budget(max_cost=1.00, exceed_policy=ExceedPolicy.STOP)
    system_prompt = (
        "You are a marketplace-native research agent. Use agoragentic_match before "
        "agoragentic_execute when fit is unclear. Search marketplace memory before repeating "
        "prior work. Save a learning note only when you discover a durable workflow lesson worth "
        "reusing later."
    )
    tools = [
        agoragentic_match,
        agoragentic_execute,
        agoragentic_memory_search,
        agoragentic_save_learning_note,
    ]


if __name__ == "__main__":
    agent = MarketplaceServeAgent()

    print("Serving at http://localhost:8000")
    print("Open http://localhost:8000/playground to inspect the agent and its tools.")

    if not os.getenv("OPENAI_API_KEY", "").strip():
        print("OPENAI_API_KEY not set; the server will start with Model.mock() for route inspection.")
    if not os.getenv("AGORAGENTIC_API_KEY", "").strip():
        print("AGORAGENTIC_API_KEY not set; marketplace tool calls will return skipped payloads.")
    if not _live_enabled():
        print("AGORAGENTIC_RUN_LIVE is not 1; paid execution and note writes remain disabled.")

    agent.serve(port=8000, enable_playground=True, debug=True)
