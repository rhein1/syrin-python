"""Agoragentic marketplace tools — execute-first third-party routing from Syrin.

Demonstrates:
  - wrapping an external capability marketplace as Syrin tools
  - previewing routed providers before paid execution
  - searching external memory before repeating work
  - gating paid or mutating actions behind an env flag

Run:
    uv run python examples/thirdparty/agroagentic/agoragentic_marketplace.py

Env:
    AGORAGENTIC_API_KEY   Required for live marketplace calls.
    AGORAGENTIC_RUN_LIVE  Set to 1 to allow paid execution and learning-note writes.
    OPENAI_API_KEY        Optional. If set, runs a full LLM-driven agent workflow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from syrin import Agent, Budget, Model, tool
from syrin.enums import ExceedPolicy

_DEFAULT_BASE_URL = "https://agoragentic.com"
_DEFAULT_MAX_COST = 0.25
_MAX_COST_HARD_CAP = 1.0
_TIMEOUT = 20.0


def _base_url() -> str:
    """Read the marketplace base URL from the current environment."""
    return os.getenv("AGORAGENTIC_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


def _headers() -> dict[str, str]:
    """Build marketplace request headers, including auth when configured."""
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("AGORAGENTIC_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _format(payload: dict[str, Any]) -> str:
    """Pretty-print tool payloads for readable CLI output."""
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


def _safe_max_cost(value: Any) -> float:
    """Coerce max_cost to a bounded float suitable for example marketplace calls."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = _DEFAULT_MAX_COST
    return max(0.0, min(parsed, _MAX_COST_HARD_CAP))


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
            f"{_base_url()}{path}",
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
        error_code = data.get("error") if isinstance(data, dict) else None
        return {
            "error": error_code or f"http_{response.status_code}",
            "message": data.get("message") if isinstance(data, dict) else str(data),
            "status_code": response.status_code,
        }

    return data if isinstance(data, dict) else {"data": data}


@tool
def agoragentic_match(task: str, max_cost: float = 0.25) -> str:
    """Preview which marketplace providers could handle a task.

    Args:
        task: Plain-English task description for the router.
        max_cost: Maximum allowed listing price in USDC. Values are clamped to 0.0..1.0.

    Returns:
        JSON object with the top marketplace providers and filter explanations.
    """
    missing = _require_marketplace_key()
    if missing:
        return _format(missing)

    safe_max_cost = _safe_max_cost(max_cost)
    data = _request(
        "GET",
        "/api/execute/match",
        params={"task": task, "max_cost": safe_max_cost},
    )
    if data.get("error"):
        return _format(data)

    raw_providers = data.get("providers")
    provider_items = raw_providers if isinstance(raw_providers, list) else []
    providers = []
    for provider in provider_items[:5]:
        if not isinstance(provider, dict):
            continue
        score = provider.get("score")
        score_data = score if isinstance(score, dict) else {}
        providers.append(
            {
                "name": provider.get("name"),
                "capability": provider.get("capability_name"),
                "price_usdc": provider.get("price"),
                "eligible": provider.get("eligible"),
                "score": score_data.get("composite"),
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
    """Route a task to the marketplace and execute it.

    Args:
        task: Plain-English task description for the router.
        input_text: Optional input text passed to the selected provider.
        max_cost: Maximum allowed listing price in USDC. Values are clamped to 0.0..1.0.

    Returns:
        JSON object with the routed provider, output, cost, and invocation ID.
    """
    missing = _require_marketplace_key()
    if missing:
        return _format(missing)
    if not _live_enabled():
        return _format(_live_guard("paid marketplace execution"))

    safe_max_cost = _safe_max_cost(max_cost)
    data = _request(
        "POST",
        "/api/execute",
        payload={
            "task": task,
            "input": {"text": input_text} if input_text else {},
            "constraints": {"max_cost": safe_max_cost},
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
    """Search external marketplace memory.

    Args:
        query: Natural-language search query for prior memory entries.
        namespace: Memory namespace bucket to search.
        limit: Maximum number of hits to return. Values are clamped to 1..50.

    Returns:
        JSON object with the top memory hits for the authenticated agent.
    """
    missing = _require_marketplace_key()
    if missing:
        return _format(missing)

    try:
        safe_limit = int(limit)
    except (TypeError, ValueError):
        safe_limit = 5
    safe_limit = max(1, min(safe_limit, 50))

    data = _request(
        "GET",
        "/api/vault/memory/search",
        params={"query": query, "namespace": namespace, "limit": safe_limit},
    )
    return _format(data)


@tool
def agoragentic_save_learning_note(title: str, lesson: str) -> str:
    """Persist one reusable lesson back to marketplace memory.

    Args:
        title: Short title for the learning note.
        lesson: Durable workflow lesson worth saving for future runs.

    Returns:
        JSON object describing the saved learning note and memory key.
    """
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


def _build_agent() -> Agent:
    """Create the example agent with marketplace-aware prompting and tools."""
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = (
        Model.OpenAI("gpt-4o-mini", api_key=openai_key)
        if openai_key
        else Model.mock(latency_min=0, latency_max=0)
    )

    return Agent(
        model=model,
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
    """Run the example in safe mode by default, with live actions gated by env vars."""
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    agent = _build_agent()
    print("Agent tools:", [tool_spec.name for tool_spec in agent.tools])

    missing = _require_marketplace_key()
    if missing:
        print(_format(missing))
        return

    print("\n=== Free preview: routed providers ===")
    print(agoragentic_match("Summarize a technical paper under $0.25"))

    print("\n=== Free recall: prior workflow memory ===")
    print(agoragentic_memory_search("summarization workflow", namespace="learning"))

    print("\n=== Safe agent.run flow ===")
    result = agent.run(
        "Use agoragentic_match to preview a marketplace provider for summarizing technical "
        "papers under $0.25, then use agoragentic_memory_search to look for prior workflow "
        "notes about summarization."
    )
    print(result.content[:1200])
    print(f"\nCost: ${result.cost:.6f}")

    if not _live_enabled():
        print("\nSkipping paid execution and learning-note writes.")
        print("Set AGORAGENTIC_RUN_LIVE=1 to enable the mutating flow.")
        return

    print("\n=== Paid execute: routed work ===")
    print(
        agoragentic_execute(
            task="Summarize this technical paper",
            input_text=(
                "Syrin focuses on production AI agents with budget control, persistent memory, "
                "multi-agent orchestration, observability, guardrails, and serving."
            ),
            max_cost=0.25,
        )
    )

    print("\n=== Saved lesson ===")
    print(
        agoragentic_save_learning_note(
            title="Execute-first marketplace workflow",
            lesson=(
                "Preview providers first, use a strict max_cost constraint, search prior "
                "memory before repeating work, and only persist lessons that are reusable."
            ),
        )
    )

    if not os.getenv("OPENAI_API_KEY", "").strip():
        print("\nSkipping live agent.run flow. Set OPENAI_API_KEY in examples/.env to enable it.")
        return

    print("\n=== Live agent.run flow ===")
    result = agent.run(
        "Preview a marketplace provider for summarizing technical papers under $0.25. "
        "If the fit looks reasonable, route the sample summary task, search prior "
        "marketplace memory for workflow notes, and save one concise reusable lesson."
    )
    print(result.content[:1200])
    print(f"\nCost: ${result.cost:.6f}")


if __name__ == "__main__":
    main()
