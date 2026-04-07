# Serving Examples

Serve Syrin agents over HTTP.



## Examples

- **http_serve.py** — Single agent: `agent.serve(port=8000)`
- **agoragentic_marketplace_serve.py** — Serve an agent that previews and routes work through the Agoragentic marketplace
- **multi_agent_router.py** — Multiple agents: `AgentRouter(agents=[...]).serve(port=8000)`
- **mount_on_existing_app.py** — Mount on your FastAPI app: `app.include_router(agent.as_router(), prefix="/agent")`
- **discovery_override.py** — Agent Card override + `Hook.DISCOVERY_REQUEST`

## Routes

- `POST /chat` — Run agent, get full response
- `POST /stream` — SSE streaming
- `GET /health` — Liveness probe
- `GET /ready` — Readiness probe
- `GET /budget` — Budget state (if configured)
- `GET /describe` — Agent introspection
- `GET /.well-known/agent-card.json` — A2A Agent Card (when discovery enabled)

For multi-agent: `/agent/{name}/chat`, `/agent/{name}/health`, etc.
