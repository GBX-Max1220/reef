"""The ChatGPT upstream: a person's own ChatGPT plan, signed in through the Codex CLI."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from reef.artifact import Artifact
from reef.inference.chatgpt import ChatGPTInferenceHandler, ChatGPTProxyRuntime, CodexSignIn, backend_request_body
from reef.runtime.interfaces import UpstreamStatusError

MESSAGE = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "reef-ok"}]}
CALL = {"type": "function_call", "name": "exec_command", "arguments": "{}", "call_id": "c1"}


def signed_in(tmp_path: Path) -> CodexSignIn:
    (tmp_path / "auth.json").write_text(
        json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "access-1", "account_id": "account-1"}})
    )
    return CodexSignIn(tmp_path)


def sse(*events: dict[str, Any]) -> bytes:
    return b"".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)


# The backend streams every reply; with store false its completed response carries no output items.
REPLY = sse(
    {"type": "response.output_item.done", "output_index": 1, "item": MESSAGE},
    {"type": "response.output_item.done", "output_index": 0, "item": CALL},
    {"type": "response.completed", "response": {"id": "resp_1", "status": "completed", "output": []}},
)


def run_against(backend: web.Application, call: Any) -> Any:
    async def run() -> Any:
        server = TestServer(backend)
        await server.start_server()
        try:
            return await call(str(server.make_url("")).rstrip("/"))
        finally:
            await server.close()

    return asyncio.run(run())


def recording_backend(requests: list[dict[str, Any]], status: int = 200, body: bytes = REPLY) -> web.Application:
    async def codex_responses(request: web.Request) -> web.Response:
        requests.append({"headers": dict(request.headers), "body": await request.json()})
        content_type = "text/event-stream" if status == 200 else "application/json"
        return web.Response(status=status, body=body, content_type=content_type)

    app = web.Application()
    app.router.add_post("/codex/responses", codex_responses)
    return app


@pytest.mark.unit
def test_a_whole_reply_is_folded_from_the_item_events_and_the_call_is_signed_in(tmp_path: Path) -> None:
    requests: list[dict[str, Any]] = []
    payload = {
        "model": "gpt-5.5",
        "input": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}],
        "max_output_tokens": 100,
    }

    async def call(url: str) -> dict[str, Any]:
        handler = ChatGPTInferenceHandler(url, sign_in=signed_in(tmp_path), timeout_s=5)
        return await handler.inference(Artifact.local(tmp_path), "/v1/responses", payload)

    reply = run_against(recording_backend(requests), call)
    assert reply["id"] == "resp_1"
    assert reply["output"] == [CALL, MESSAGE]
    [request] = requests
    assert request["headers"]["Authorization"] == "Bearer access-1"
    assert request["headers"]["chatgpt-account-id"] == "account-1"
    assert request["body"]["stream"] is True and request["body"]["store"] is False
    assert request["body"]["instructions"] == "Be brief."
    assert request["body"]["input"] == [{"role": "user", "content": "hi"}]
    assert "max_output_tokens" not in request["body"]


@pytest.mark.unit
def test_a_streamed_reply_passes_through_as_the_backend_sent_it(tmp_path: Path) -> None:
    async def call(url: str) -> bytes:
        handler = ChatGPTInferenceHandler(url, sign_in=signed_in(tmp_path), timeout_s=5)
        stream = await handler.inference_stream(Artifact.local(tmp_path), "/v1/responses", {"stream": True})
        try:
            return b"".join([chunk async for chunk in stream.chunks])
        finally:
            await stream.close()

    assert run_against(recording_backend([]), call) == REPLY


@pytest.mark.unit
def test_the_backend_body_keeps_instructions_and_asks_for_encrypted_reasoning() -> None:
    body = backend_request_body(
        {
            "instructions": "Base prompt.",
            "input": [
                {"role": "developer", "content": [{"type": "input_text", "text": "Use the shell."}]},
                {"role": "user", "content": "hi"},
            ],
            "include": ["message.output_text.logprobs"],
            "service_tier": "flex",
        }
    )
    assert body["instructions"] == "Base prompt.\n\nUse the shell."
    assert body["input"] == [{"role": "user", "content": "hi"}]
    assert body["include"] == ["message.output_text.logprobs", "reasoning.encrypted_content"]
    assert "service_tier" not in body
    assert backend_request_body({"input": [{"role": "user", "content": "hi"}]})["instructions"]


@pytest.mark.unit
def test_only_the_responses_route_is_served(tmp_path: Path) -> None:
    async def call(url: str) -> None:
        handler = ChatGPTInferenceHandler(url, sign_in=signed_in(tmp_path), timeout_s=5)
        await handler.inference(Artifact.local(tmp_path), "/v1/chat/completions", {"messages": []})

    with pytest.raises(UpstreamStatusError, match="/v1/responses"):
        run_against(recording_backend([]), call)


@pytest.mark.unit
def test_no_sign_in_names_codex_login(tmp_path: Path) -> None:
    async def call(url: str) -> None:
        handler = ChatGPTInferenceHandler(url, sign_in=CodexSignIn(tmp_path), timeout_s=5)
        await handler.inference(Artifact.local(tmp_path), "/v1/responses", {"input": []})

    with pytest.raises(UpstreamStatusError, match="codex login"):
        run_against(recording_backend([]), call)


@pytest.mark.unit
def test_an_expired_sign_in_reaches_the_caller_with_the_backend_message(tmp_path: Path) -> None:
    refusal = b'{"detail": "Your authentication token has expired."}'

    async def call(url: str) -> None:
        handler = ChatGPTInferenceHandler(url, sign_in=signed_in(tmp_path), timeout_s=5)
        await handler.inference(Artifact.local(tmp_path), "/v1/responses", {"input": []})

    with pytest.raises(UpstreamStatusError, match=r"token has expired.*codex login") as raised:
        run_against(recording_backend([], status=401, body=refusal), call)
    assert raised.value.status == 401


@pytest.mark.unit
def test_the_chatgpt_upstream_builds_the_chatgpt_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from reef.service.assembly import _upstream_runtime
    from reef.service.deploy.service_config import ServiceConfig

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    settings = ServiceConfig(
        recipe="recipe",
        upstream_url="https://chatgpt.com/backend-api",
        upstream_model="gpt-5.5",
        upstream_api="chatgpt",
    )
    runtime = _upstream_runtime(settings)
    assert isinstance(runtime, ChatGPTProxyRuntime)
    assert (runtime.model_path, runtime.api) == ("gpt-5.5", "responses")
    handler = runtime.inference_handler
    assert isinstance(handler, ChatGPTInferenceHandler)
    assert handler.sign_in.path == tmp_path / "auth.json"


@pytest.mark.unit
def test_the_runtime_serves_the_responses_dialect(tmp_path: Path) -> None:
    runtime = ChatGPTProxyRuntime(
        model_path="gpt-5.5", base_url="https://chatgpt.com/backend-api", sign_in=signed_in(tmp_path)
    )
    assert runtime.api == "responses"
    assert runtime.api_key is None
    assert isinstance(runtime.inference_handler, ChatGPTInferenceHandler)
