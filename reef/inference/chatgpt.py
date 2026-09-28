"""The ChatGPT backend as Reef's upstream: a person's own ChatGPT plan, signed in with ``reef login chatgpt``.

Selected with ``inference.upstream_api: chatgpt``. The backend speaks a
constrained Responses dialect at ``/codex/responses``: it takes only streamed,
unstored requests with the system text in ``instructions``, and with
``store: false`` its completed response carries no output items, which arrive
as ``response.output_item.done`` events instead. Clients and Reef's own calls
speak plain Responses to Reef; the handler adjusts each request for the
backend and folds a reply a caller asked for whole.

The sign-in is Reef's own, from ``reef login chatgpt``
(:mod:`reef.inference.chatgpt_sign_in`). A call the backend refuses as signed
out is sent once more on a renewed sign-in.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from reef.artifact.artifact import Artifact
from reef.inference.chatgpt_sign_in import SIGN_IN_HINT, ChatGPTSignIn
from reef.inference.http import HttpInferenceHandler, InferenceProxyRuntime, RequestHeadersFactory
from reef.runtime.interfaces import InferenceHandler, InferenceStream, UpstreamStatusError

#: The ``inference.upstream_api`` value that selects this upstream.
CHATGPT_UPSTREAM_API = "chatgpt"
RESPONSES_PATH = "/v1/responses"
BACKEND_PATH = "/codex/responses"
#: Request fields the backend refuses.
DROPPED_FIELDS = ("max_output_tokens", "prompt_cache_retention", "prompt_cache_options", "service_tier")


class ChatGPTRequestHeaders(RequestHeadersFactory):
    """The backend's sign-in headers, read from the sign-in for every call."""

    def __init__(self, sign_in: ChatGPTSignIn) -> None:
        self.sign_in = sign_in

    def headers(self, artifact: Artifact, path: str) -> Mapping[str, str]:
        token, account = self.sign_in.credentials()
        return {
            "Authorization": f"Bearer {token}",
            "chatgpt-account-id": account,
            "originator": "reef",
            "OpenAI-Beta": "responses=experimental",
        }


def backend_request_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    """``payload`` as the backend takes it: streamed and unstored, its system and developer items moved into
    ``instructions``, encrypted reasoning included, and without the fields the backend refuses."""
    source = payload.get("input")
    items = source if isinstance(source, list) else []
    instructions = [payload["instructions"]] if payload.get("instructions") else []
    kept = []
    for item in items:
        if isinstance(item, dict) and item.get("role") in ("system", "developer"):
            content = item.get("content")
            if isinstance(content, str):
                instructions.append(content)
            elif isinstance(content, list):
                instructions += [part["text"] for part in content if isinstance(part, dict) and "text" in part]
        else:
            kept.append(item)
    body = {key: value for key, value in payload.items() if key not in DROPPED_FIELDS}
    if isinstance(source, list):
        body["input"] = kept
    include = list(payload.get("include") or [])
    if "reasoning.encrypted_content" not in include:
        include.append("reasoning.encrypted_content")
    if instructions:
        body["instructions"] = "\n\n".join(instructions)
    body.update(stream=True, store=False, include=include)
    return body


class ChatGPTInferenceHandler(HttpInferenceHandler):
    """POST a caller's ``/v1/responses`` request to the backend, adjusted and signed in."""

    def __init__(self, upstream_url: str, *, sign_in: ChatGPTSignIn, timeout_s: float = 300.0) -> None:
        super().__init__(
            upstream_url,
            request_headers=ChatGPTRequestHeaders(sign_in),
            timeout_s=timeout_s,
            error_label="ChatGPT backend",
        )
        self.sign_in = sign_in

    def _post_arguments(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path != RESPONSES_PATH:
            raise UpstreamStatusError(f"the chatgpt upstream serves {RESPONSES_PATH} only, not {path}", status=404)
        return super()._post_arguments(artifact, BACKEND_PATH, backend_request_body(payload))

    def _status_error(self, status: int, body: str) -> UpstreamStatusError:
        error = super()._status_error(status, body)
        if status == 401:
            return UpstreamStatusError(f"{error}; {SIGN_IN_HINT}", status=status)
        return error

    async def inference_stream(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> InferenceStream:
        """The backend's events as it sends them, typed ``text/event-stream``. The backend sends no content type,
        and Reef's stream route and Codex both read a stream as events only by that type."""
        try:
            stream = await super().inference_stream(artifact, path, payload)
        except UpstreamStatusError as error:
            if error.status != 401:
                raise
            self.sign_in.renew()
            stream = await super().inference_stream(artifact, path, payload)
        headers = {name: value for name, value in stream.headers.items() if name.lower() != "content-type"}
        return InferenceStream(
            status=stream.status,
            headers={**headers, "Content-Type": "text/event-stream"},
            chunks=stream.chunks,
            close=stream.close,
        )

    async def inference(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """A reply the caller asked for whole: the backend streams every reply, so it is folded here."""
        try:
            return await self.folded_reply(artifact, path, payload)
        except UpstreamStatusError as error:
            if error.status != 401:
                raise
            self.sign_in.renew()
            return await self.folded_reply(artifact, path, payload)

    async def folded_reply(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
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
        self, *, model_path: str, base_url: str, sign_in: ChatGPTSignIn, inference_timeout_s: float = 300.0
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
    "CHATGPT_UPSTREAM_API",
    "ChatGPTInferenceHandler",
    "ChatGPTProxyRuntime",
    "ChatGPTRequestHeaders",
    "backend_request_body",
]
