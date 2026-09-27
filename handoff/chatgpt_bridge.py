"""Spike: serve a ChatGPT plan to Reef as a plain `responses` upstream on loopback.

Reads the Codex CLI's sign-in from $CODEX_HOME/auth.json, maps /v1/responses to the ChatGPT backend's
/codex/responses, adjusts the body the way that backend requires, and folds a streamed reply into one JSON
response when the caller asked for no stream. Logs each call's adjustments and status, never a token.
Throwaway scaffolding for the RFC's feasibility check, not Reef code.
"""

import json
import os
import sys
import time
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web

BACKEND = "https://chatgpt.com/backend-api/codex/responses"
AUTH_FILE = Path(os.environ["CODEX_HOME"]) / "auth.json"
MODELS = ["gpt-5.4-mini", "gpt-5.4", "gpt-5.5"]
# Keys the backend is expected to refuse, per pi-ai's Codex client; the spike confirms or trims this list.
DROPPED_KEYS = ("max_output_tokens", "prompt_cache_retention", "prompt_cache_options", "service_tier")


def log(entry: dict) -> None:
    print(json.dumps({"t": round(time.time(), 1), **entry}), file=sys.stderr, flush=True)


def credentials() -> tuple[str, str]:
    tokens = json.loads(AUTH_FILE.read_text())["tokens"]
    return tokens["access_token"], tokens["account_id"]


def text_of(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return ""


def adjust(body: dict) -> tuple[dict, list[str]]:
    changes: list[str] = []
    instructions = [body["instructions"]] if body.get("instructions") else []
    kept = []
    for item in body.get("input", []):
        if isinstance(item, dict) and item.get("role") in ("system", "developer"):
            instructions.append(text_of(item.get("content")))
        else:
            kept.append(item)
    if len(kept) != len(body.get("input", [])):
        changes.append("system/developer input -> instructions")
    adjusted = {key: value for key, value in body.items() if key not in DROPPED_KEYS}
    changes += [f"dropped {key}" for key in DROPPED_KEYS if key in body]
    adjusted.update(
        input=kept,
        instructions="\n\n".join(instructions) or "You are a helpful assistant.",
        stream=True,
        store=False,
    )
    if "reasoning.encrypted_content" not in adjusted.get("include", []):
        adjusted["include"] = [*adjusted.get("include", []), "reasoning.encrypted_content"]
    if not body.get("stream"):
        changes.append("stream forced on, reply folded")
    return adjusted, changes


async def models(request: web.Request) -> web.Response:
    return web.json_response({"object": "list", "data": [{"id": m, "object": "model"} for m in MODELS]})


async def responses(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    adjusted, changes = adjust(body)
    token, account = credentials()
    headers = {
        "Authorization": f"Bearer {token}",
        "chatgpt-account-id": account,
        "originator": "reef",
        "OpenAI-Beta": "responses=experimental",
        "content-type": "application/json",
        "accept": "text/event-stream",
    }
    session: ClientSession = request.app["session"]
    async with session.post(BACKEND, json=adjusted, headers=headers) as upstream:
        if upstream.status >= 400:
            detail = await upstream.text()
            log({"model": body.get("model"), "changes": changes, "status": upstream.status, "error": detail[:500]})
            return web.Response(status=upstream.status, text=detail, content_type="application/json")
        log(
            {
                "model": body.get("model"),
                "changes": changes,
                "status": upstream.status,
                "stream": bool(body.get("stream")),
            }
        )
        if body.get("stream"):
            response = web.StreamResponse(status=200, headers={"content-type": "text/event-stream"})
            await response.prepare(request)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
            await response.write_eof()
            return response
        final = None
        items: dict[int, dict] = {}
        async for raw in upstream.content:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data:"):
                event = json.loads(line[5:].strip() or "{}")
                if event.get("type") == "response.output_item.done":
                    items[event.get("output_index", len(items))] = event["item"]
                elif event.get("type") in ("response.completed", "response.incomplete", "response.failed"):
                    final = event.get("response")
        if final is None:
            return web.json_response({"error": {"message": "the backend stream ended without a response"}}, status=502)
        # With store: false the completed response carries no output; the items arrived as events.
        if not final.get("output"):
            final["output"] = [items[index] for index in sorted(items)]
        return web.json_response(final)


async def open_session(app: web.Application) -> None:
    app["session"] = ClientSession(timeout=ClientTimeout(total=900))


async def close_session(app: web.Application) -> None:
    await app["session"].close()


app = web.Application(client_max_size=64 * 1024 * 1024)
app.on_startup.append(open_session)
app.on_cleanup.append(close_session)
app.router.add_get("/v1/models", models)
app.router.add_post("/v1/responses", responses)

if __name__ == "__main__":
    web.run_app(app, host="127.0.0.1", port=9902, print=None)
