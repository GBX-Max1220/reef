"""The ChatGPT upstream: a person's own ChatGPT plan, signed in with ``reef login chatgpt``."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import stat
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from reef.artifact import Artifact
from reef.inference.chatgpt import ChatGPTInferenceHandler, ChatGPTProxyRuntime, backend_request_body
from reef.inference.chatgpt_sign_in import ChatGPTSignIn, ChatGPTSignInError, ChatGPTTokens
from reef.runtime.interfaces import UpstreamStatusError

MESSAGE = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "reef-ok"}]}
CALL = {"type": "function_call", "name": "exec_command", "arguments": "{}", "call_id": "c1"}


def access_token(account: str, marker: str) -> str:
    """An access token shaped as OpenAI issues them: the account id sits in the auth claim of its payload."""
    claims = {"https://api.openai.com/auth": {"chatgpt_account_id": account}, "marker": marker}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"eyJhbGciOiJub25lIn0.{payload}.signature"


def signed_in(tmp_path: Path, auth_url: str = "http://127.0.0.1:9", expires_in: float = 3600) -> ChatGPTSignIn:
    sign_in = ChatGPTSignIn(tmp_path / "chatgpt.json", auth_base_url=auth_url)
    sign_in.save(ChatGPTTokens("access-1", "refresh-1", time.time() + expires_in, "account-1"))
    return sign_in


class AuthServer:
    """A stub of OpenAI's sign-in server: the device-code endpoints and the token endpoint."""

    def __init__(self, pending_polls: int = 1, refresh_status: int = 200, device_codes: bool = True) -> None:
        self.pending_polls = pending_polls
        self.refresh_status = refresh_status
        self.device_codes = device_codes
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.user_agents: set[str] = set()
        auth = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers["Content-Length"])).decode()
                if self.headers["Content-Type"] == "application/json":
                    body = json.loads(raw)
                else:
                    body = dict(parse_qsl(raw))
                auth.requests.append((self.path, body))
                auth.user_agents.add(self.headers["User-Agent"])
                status, reply = auth.answer(self.path, body)
                data = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def answer(self, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if path == "/api/accounts/deviceauth/usercode":
            if not self.device_codes:
                return 404, {"detail": "Not Found"}
            return 200, {"device_auth_id": "device-1", "user_code": "ABCD-1234", "interval": "0"}
        if path == "/api/accounts/deviceauth/token":
            if self.pending_polls:
                self.pending_polls -= 1
                return 403, {"error": {"code": "deviceauth_authorization_pending"}}
            return 200, {"authorization_code": "code-1", "code_verifier": "verifier-1"}
        if body.get("grant_type") == "refresh_token" and self.refresh_status != 200:
            return self.refresh_status, {"error": "invalid_grant"}
        marker = "renewed" if body.get("grant_type") == "refresh_token" else "new"
        return 200, {
            "access_token": access_token("account-1", marker),
            "refresh_token": f"refresh-{marker}",
            "expires_in": 864000,
        }

    def __enter__(self) -> AuthServer:
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *args: object) -> None:
        self.server.shutdown()
        self.server.server_close()


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
def test_a_streamed_reply_is_named_an_event_stream_though_the_backend_names_no_type(tmp_path: Path) -> None:
    """The backend sends its events with no text/event-stream content type; Reef relays a stream as events, and
    Codex reads it, only by that type."""

    async def untyped(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(REPLY)
        return response

    backend = web.Application()
    backend.router.add_post("/codex/responses", untyped)

    async def call(url: str) -> dict[str, str]:
        handler = ChatGPTInferenceHandler(url, sign_in=signed_in(tmp_path), timeout_s=5)
        stream = await handler.inference_stream(Artifact.local(tmp_path), "/v1/responses", {"stream": True})
        await stream.close()
        return stream.headers

    headers = run_against(backend, call)
    assert [value for name, value in headers.items() if name.lower() == "content-type"] == ["text/event-stream"]


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
    assert "instructions" not in backend_request_body({"input": [{"role": "user", "content": "hi"}]})


@pytest.mark.unit
def test_only_the_responses_route_is_served(tmp_path: Path) -> None:
    async def call(url: str) -> None:
        handler = ChatGPTInferenceHandler(url, sign_in=signed_in(tmp_path), timeout_s=5)
        await handler.inference(Artifact.local(tmp_path), "/v1/chat/completions", {"messages": []})

    with pytest.raises(UpstreamStatusError, match="/v1/responses"):
        run_against(recording_backend([]), call)


@pytest.mark.unit
def test_no_sign_in_names_reef_login(tmp_path: Path) -> None:
    async def call(url: str) -> None:
        handler = ChatGPTInferenceHandler(url, sign_in=ChatGPTSignIn(tmp_path / "chatgpt.json"), timeout_s=5)
        await handler.inference(Artifact.local(tmp_path), "/v1/responses", {"input": []})

    with pytest.raises(UpstreamStatusError, match="reef login chatgpt"):
        run_against(recording_backend([]), call)


def refusing_once(requests: list[dict[str, Any]]) -> web.Application:
    """A backend that refuses the first call as signed out and answers the next."""

    async def codex_responses(request: web.Request) -> web.Response:
        requests.append({"headers": dict(request.headers)})
        if len(requests) == 1:
            return web.Response(status=401, body=b'{"detail": "token revoked"}', content_type="application/json")
        return web.Response(status=200, body=REPLY, content_type="text/event-stream")

    app = web.Application()
    app.router.add_post("/codex/responses", codex_responses)
    return app


@pytest.mark.unit
@pytest.mark.parametrize("stream", [False, True])
def test_a_call_the_backend_refuses_as_signed_out_is_renewed_and_sent_again(tmp_path: Path, stream: bool) -> None:
    requests: list[dict[str, Any]] = []

    async def call(url: str) -> object:
        handler = ChatGPTInferenceHandler(url, sign_in=signed_in(tmp_path, auth.url), timeout_s=5)
        if not stream:
            return await handler.inference(Artifact.local(tmp_path), "/v1/responses", {"input": []})
        reply = await handler.inference_stream(Artifact.local(tmp_path), "/v1/responses", {"stream": True})
        await reply.close()
        return reply.status

    with AuthServer() as auth:
        result = run_against(refusing_once(requests), call)
    assert result == 200 if stream else result["id"] == "resp_1"
    assert [request["headers"]["Authorization"] for request in requests] == [
        "Bearer access-1",
        f"Bearer {access_token('account-1', 'renewed')}",
    ]


@pytest.mark.unit
def test_a_refusal_the_renewed_sign_in_cannot_fix_names_reef_login(tmp_path: Path) -> None:
    refusal = b'{"detail": "Your authentication token has expired."}'

    async def call(url: str) -> None:
        handler = ChatGPTInferenceHandler(url, sign_in=signed_in(tmp_path, auth.url), timeout_s=5)
        await handler.inference(Artifact.local(tmp_path), "/v1/responses", {"input": []})

    with AuthServer(refresh_status=400) as auth, pytest.raises(UpstreamStatusError, match="reef login chatgpt"):
        run_against(recording_backend([], status=401, body=refusal), call)


@pytest.mark.unit
def test_device_code_sign_in_keeps_the_tokens_for_its_owner_only(tmp_path: Path) -> None:
    output = io.StringIO()
    with AuthServer(pending_polls=2) as auth:
        sign_in = ChatGPTSignIn(tmp_path / "credentials" / "chatgpt.json", auth_base_url=auth.url)
        sign_in.sign_in_with_device_code(output)
    assert "ABCD-1234" in output.getvalue() and f"{auth.url}/codex/device" in output.getvalue()
    # OpenAI's sign-in server refuses Python's default user agent (a Cloudflare 530).
    assert [agent.split("/")[0] for agent in auth.user_agents] == ["reef"]
    assert stat.S_IMODE(sign_in.path.stat().st_mode) == 0o600
    assert sign_in.credentials() == (access_token("account-1", "new"), "account-1")
    exchange = [body for path, body in auth.requests if path == "/oauth/token"]
    assert exchange == [
        {
            "grant_type": "authorization_code",
            "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
            "code": "code-1",
            "code_verifier": "verifier-1",
            "redirect_uri": "https://auth.openai.com/deviceauth/callback",
        }
    ]


@pytest.mark.unit
def test_an_account_without_device_code_sign_in_is_told_so(tmp_path: Path) -> None:
    with AuthServer(device_codes=False) as auth, pytest.raises(ChatGPTSignInError, match="device code"):
        ChatGPTSignIn(tmp_path / "chatgpt.json", auth_base_url=auth.url).sign_in_with_device_code(io.StringIO())


@pytest.mark.unit
def test_a_sign_in_near_expiry_is_renewed_and_kept(tmp_path: Path) -> None:
    with AuthServer() as auth:
        sign_in = signed_in(tmp_path, auth.url, expires_in=60)
        assert sign_in.credentials() == (access_token("account-1", "renewed"), "account-1")
    assert json.loads(sign_in.path.read_text())["refresh_token"] == "refresh-renewed"
    assert [body["refresh_token"] for path, body in auth.requests] == ["refresh-1"]


@pytest.mark.unit
def test_a_renewal_the_server_refuses_names_reef_login(tmp_path: Path) -> None:
    with AuthServer(refresh_status=400) as auth, pytest.raises(UpstreamStatusError, match="reef login chatgpt"):
        signed_in(tmp_path, auth.url, expires_in=60).credentials()


@pytest.mark.unit
def test_sign_out_removes_the_sign_in(tmp_path: Path) -> None:
    sign_in = signed_in(tmp_path)
    assert sign_in.sign_out() is True
    assert not sign_in.path.exists()
    assert sign_in.sign_out() is False


@pytest.mark.unit
def test_the_chatgpt_upstream_builds_the_chatgpt_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from reef.service.assembly import _upstream_runtime
    from reef.service.deploy.service_config import ServiceConfig

    monkeypatch.setenv("HOME", str(tmp_path))
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
    assert handler.sign_in.path == tmp_path / ".reef" / "credentials" / "chatgpt.json"


@pytest.mark.unit
def test_the_runtime_serves_the_responses_dialect(tmp_path: Path) -> None:
    runtime = ChatGPTProxyRuntime(
        model_path="gpt-5.5", base_url="https://chatgpt.com/backend-api", sign_in=signed_in(tmp_path)
    )
    assert runtime.api == "responses"
    assert runtime.api_key is None
    assert isinstance(runtime.inference_handler, ChatGPTInferenceHandler)
