"""The ChatGPT backend as Reef's upstream: a person's own ChatGPT plan, signed in through the Codex CLI.

Selected with ``inference.upstream_api: chatgpt``. The backend speaks a
constrained Responses dialect at ``/codex/responses``: it takes only streamed,
unstored requests with the system text in ``instructions``, and with
``store: false`` its completed response carries no output items, which arrive
as ``response.output_item.done`` events instead. Clients and Reef's own calls
speak plain Responses to Reef; the handler adjusts each request for the
backend and folds a reply a caller asked for whole.

The sign-in is the one the Codex CLI keeps in ``$CODEX_HOME/auth.json``. Reef
reads it for every call, so a refresh Codex made is picked up, and never
writes or refreshes it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.inference.http import HttpInferenceHandler, InferenceProxyRuntime, RequestHeadersFactory
from reef.runtime.interfaces import InferenceHandler, UpstreamStatusError

RESPONSES_PATH = "/v1/responses"
BACKEND_PATH = "/codex/responses"
SIGN_IN_HINT = "sign in with `codex login`"
#: Request fields the backend refuses.
DROPPED_FIELDS = ("max_output_tokens", "prompt_cache_retention", "prompt_cache_options", "service_tier")
#: The backend requires instructions; a request without system text gets these.
DEFAULT_INSTRUCTIONS = "You are a helpful assistant."


class CodexSignIn:
    """The ChatGPT sign-in the Codex CLI keeps in ``<codex_home>/auth.json``."""

    def __init__(self, codex_home: Path) -> None:
        self.path = Path(codex_home) / "auth.json"

    @classmethod
    def from_environment(cls, environ: Mapping[str, str]) -> CodexSignIn:
        """The sign-in under ``CODEX_HOME``, else ``~/.codex``, as Codex itself resolves it."""
        return cls(Path(environ.get("CODEX_HOME") or Path.home() / ".codex"))

    def credentials(self) -> tuple[str, str]:
        """The access token and account id, read now; an error naming ``codex login`` when there are none."""
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise UpstreamStatusError(f"no ChatGPT sign-in at {self.path}: {SIGN_IN_HINT}", status=401) from None
        tokens = document.get("tokens") if isinstance(document, dict) else None
        if not isinstance(tokens, dict):
            raise UpstreamStatusError(f"{self.path} holds no ChatGPT sign-in: {SIGN_IN_HINT}", status=401)
        token, account = tokens.get("access_token"), tokens.get("account_id")
        if not isinstance(token, str) or not token or not isinstance(account, str) or not account:
            raise UpstreamStatusError(f"{self.path} holds no ChatGPT sign-in: {SIGN_IN_HINT}", status=401)
        return token, account


class ChatGPTRequestHeaders(RequestHeadersFactory):
    """The backend's sign-in headers, read from the sign-in for every call."""

    def __init__(self, sign_in: CodexSignIn) -> None:
        self.sign_in = sign_in

    def headers(self, artifact: Artifact, path: str) -> Mapping[str, str]:
        token, account = self.sign_in.credentials()
        return {
            "Authorization": f"Bearer {token}",
            "chatgpt-account-id": account,
            "originator": "reef",
            "OpenAI-Beta": "responses=experimental",
        }


def message_text(content: object) -> str:
    """The text of a Responses message's content: a string, or its parts' ``text``."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part["text"] for part in content if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def backend_request_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    """``payload`` as the backend takes it: streamed and unstored, its system and developer items moved into
    ``instructions``, encrypted reasoning included, and without the fields the backend refuses."""
    source = payload.get("input")
    items = source if isinstance(source, list) else []
    instructions = [payload["instructions"]] if payload.get("instructions") else []
    kept = []
    for item in items:
        if isinstance(item, dict) and item.get("role") in ("system", "developer"):
            instructions.append(message_text(item.get("content")))
        else:
            kept.append(item)
    body = {key: value for key, value in payload.items() if key not in DROPPED_FIELDS}
    if isinstance(source, list):
        body["input"] = kept
    include = list(payload.get("include") or [])
    if "reasoning.encrypted_content" not in include:
        include.append("reasoning.encrypted_content")
    body.update(
        instructions="\n\n".join(instructions) or DEFAULT_INSTRUCTIONS, stream=True, store=False, include=include
    )
    return body


class ChatGPTInferenceHandler(HttpInferenceHandler):
    """POST a caller's ``/v1/responses`` request to the backend, adjusted and signed in."""

    def __init__(self, upstream_url: str, *, sign_in: CodexSignIn, timeout_s: float = 300.0) -> None:
        super().__init__(
            upstream_url,
            request_headers=ChatGPTRequestHeaders(sign_in),
            timeout_s=timeout_s,
            error_label="ChatGPT backend",
        )

    def _post_arguments(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path != RESPONSES_PATH:
            raise UpstreamStatusError(f"the chatgpt upstream serves {RESPONSES_PATH} only, not {path}", status=404)
        return super()._post_arguments(artifact, BACKEND_PATH, backend_request_body(payload))

    def _status_error(self, status: int, body: str) -> UpstreamStatusError:
        error = super()._status_error(status, body)
        if status == 401:
            return UpstreamStatusError(f"{error}; {SIGN_IN_HINT}", status=status)
        return error

    async def inference(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """A reply the caller asked for whole: the backend streams every reply, so it is folded here."""
        from aiohttp import ClientSession, ClientTimeout

        async with (
            ClientSession(timeout=ClientTimeout(total=self._timeout_s)) as session,
            session.post(**self._post_arguments(artifact, path, payload)) as response,
        ):
            if response.status >= 400:
                raise self._status_error(response.status, await response.text())
            items: dict[int, dict[str, Any]] = {}
            completed: dict[str, Any] | None = None
            async for line in response.content:
                text = line.decode("utf-8", errors="replace").strip()
                if not text.startswith("data:"):
                    continue
                event = json.loads(text[len("data:") :])
                if event.get("type") == "response.output_item.done":
                    items[event["output_index"]] = event["item"]
                elif event.get("type") in ("response.completed", "response.incomplete", "response.failed"):
                    completed = event["response"]
        if completed is None:
            raise UpstreamStatusError("the ChatGPT backend's stream ended without a response", status=502)
        if not completed.get("output"):
            completed["output"] = [items[index] for index in sorted(items)]
        return completed


class ChatGPTProxyRuntime(InferenceProxyRuntime):
    """The proxy runtime on a person's ChatGPT plan; clients and Reef's own calls speak Responses to it."""

    def __init__(
        self, *, model_path: str, base_url: str, sign_in: CodexSignIn, inference_timeout_s: float = 300.0
    ) -> None:
        super().__init__(
            model_path=model_path, base_url=base_url, api="responses", inference_timeout_s=inference_timeout_s
        )
        self.chatgpt_handler = ChatGPTInferenceHandler(
            self.base_url, sign_in=sign_in, timeout_s=self.inference_timeout_s
        )

    @property
    def inference_handler(self) -> InferenceHandler:
        return self.chatgpt_handler


__all__ = [
    "ChatGPTInferenceHandler",
    "ChatGPTProxyRuntime",
    "ChatGPTRequestHeaders",
    "CodexSignIn",
    "backend_request_body",
]
