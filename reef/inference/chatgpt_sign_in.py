"""Reef's own ChatGPT sign-in, for the ``chatgpt`` upstream.

``reef login chatgpt`` signs a person in with the device-code flow OpenAI's
Codex clients use: Reef shows a code, the person approves it in any browser,
and Reef keeps the resulting tokens in ``~/.reef/credentials/chatgpt.json``,
readable by its owner only. The upstream reads that file for every call,
renews the access token shortly before it expires, and keeps the renewed
tokens. OpenAI publishes no OAuth client for other applications, so Reef signs
in as the Codex CLI's public client, as pi does.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlencode

from reef.core import ReefError
from reef.core.version import __version__
from reef.runtime.interfaces import UpstreamStatusError

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTH_BASE_URL = "https://auth.openai.com"
#: Where the device-code flow sends an approved sign-in; the token exchange names it again.
DEVICE_REDIRECT_URL = f"{AUTH_BASE_URL}/deviceauth/callback"
DEVICE_CODE_TIMEOUT_SECONDS = 15 * 60
#: Renew an access token this long before it expires.
RENEWAL_MARGIN_SECONDS = 300
ACCOUNT_CLAIM = "https://api.openai.com/auth"
SIGN_IN_HINT = "sign in with `reef login chatgpt`"


class ChatGPTSignInError(ReefError):
    """The ChatGPT sign-in could not be completed."""


@dataclass(frozen=True)
class ChatGPTTokens:
    """A signed-in ChatGPT account's tokens; ``expires_at`` is the access token's expiry in epoch seconds."""

    access_token: str
    refresh_token: str
    expires_at: float
    account_id: str


def post(url: str, fields: Mapping[str, str], *, form: bool) -> tuple[int, dict[str, Any]]:
    """POST ``fields`` as a form or as JSON; the status and the JSON reply, an error status included."""
    if form:
        data, content_type = urlencode(fields).encode(), "application/x-www-form-urlencoded"
    else:
        data, content_type = json.dumps(fields).encode(), "application/json"
    # The sign-in server refuses Python's default user agent.
    headers = {"Content-Type": content_type, "User-Agent": f"reef/{__version__}"}
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        try:
            reply = json.loads(error.read() or b"{}")
        except ValueError:
            reply = {}
        return error.code, reply if isinstance(reply, dict) else {}


def tokens_from(reply: Mapping[str, Any]) -> ChatGPTTokens:
    """The tokens a token-endpoint reply grants; the account id is read from the access token's claims."""
    access_token, refresh_token, expires_in = (
        reply.get("access_token"),
        reply.get("refresh_token"),
        reply.get("expires_in"),
    )
    if (
        not isinstance(access_token, str)
        or not isinstance(refresh_token, str)
        or isinstance(expires_in, bool)
        or not isinstance(expires_in, (int, float))
    ):
        raise ChatGPTSignInError(f"the sign-in server's reply lacks tokens: {sorted(reply)}")
    payload = access_token.split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    account_id = claims.get(ACCOUNT_CLAIM, {}).get("chatgpt_account_id")
    if not isinstance(account_id, str) or not account_id:
        raise ChatGPTSignInError("the access token names no ChatGPT account")
    return ChatGPTTokens(access_token, refresh_token, time.time() + expires_in, account_id)


class ChatGPTSignIn:
    """The ChatGPT sign-in Reef keeps at ``path``."""

    def __init__(self, path: Path, *, auth_base_url: str = AUTH_BASE_URL) -> None:
        self.path = path
        self.auth_base_url = auth_base_url

    @classmethod
    def default(cls) -> ChatGPTSignIn:
        return cls(Path.home() / ".reef" / "credentials" / "chatgpt.json")

    def credentials(self) -> tuple[str, str]:
        """The access token and account id, renewed first when the token is about to expire."""
        tokens = self.load()
        if tokens.expires_at - time.time() < RENEWAL_MARGIN_SECONDS:
            tokens = self.renew()
        return tokens.access_token, tokens.account_id

    def load(self) -> ChatGPTTokens:
        try:
            return ChatGPTTokens(**json.loads(self.path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            raise UpstreamStatusError(f"no ChatGPT sign-in at {self.path}: {SIGN_IN_HINT}", status=401) from None

    def renew(self) -> ChatGPTTokens:
        """Trade the refresh token for new tokens and keep them."""
        status, reply = post(
            f"{self.auth_base_url}/oauth/token",
            {"grant_type": "refresh_token", "refresh_token": self.load().refresh_token, "client_id": CLIENT_ID},
            form=True,
        )
        if status != 200:
            raise UpstreamStatusError(
                f"the ChatGPT sign-in could not be renewed ({status}): {SIGN_IN_HINT}", status=401
            )
        tokens = tokens_from(reply)
        self.save(tokens)
        return tokens

    def save(self, tokens: ChatGPTTokens) -> None:
        """Write ``tokens`` readable by their owner only, replacing the file in one step."""
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        staged = self.path.with_suffix(".tmp")
        descriptor = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(asdict(tokens), file)
        os.replace(staged, self.path)

    def sign_out(self) -> bool:
        """Remove the sign-in; whether there was one."""
        try:
            self.path.unlink()
        except FileNotFoundError:
            return False
        return True

    def sign_in_with_device_code(self, output: TextIO) -> None:
        """Show a code for the person to approve in a browser, wait for the approval, and keep the tokens."""
        status, device = post(
            f"{self.auth_base_url}/api/accounts/deviceauth/usercode", {"client_id": CLIENT_ID}, form=False
        )
        if status == 404:
            raise ChatGPTSignInError("the sign-in server does not offer device code sign-in for this account (404)")
        if status != 200:
            raise ChatGPTSignInError(f"the sign-in server refused a device code ({status}): {device}")
        interval = float(device["interval"])
        output.write(
            f"Open {self.auth_base_url}/codex/device in a browser and enter the code {device['user_code']}.\n"
            "Waiting for the approval...\n"
        )
        output.flush()
        deadline = time.monotonic() + DEVICE_CODE_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            status, approval = post(
                f"{self.auth_base_url}/api/accounts/deviceauth/token",
                {"device_auth_id": device["device_auth_id"], "user_code": device["user_code"]},
                form=False,
            )
            if status == 200:
                break
            error = approval.get("error")
            code = error.get("code") if isinstance(error, dict) else error
            if code == "slow_down":
                interval += 5
            elif status not in (403, 404) and code != "deviceauth_authorization_pending":
                raise ChatGPTSignInError(f"the sign-in was not approved ({status}): {approval}")
            time.sleep(interval)
        else:
            raise ChatGPTSignInError("the code was not approved within 15 minutes")
        status, reply = post(
            f"{self.auth_base_url}/oauth/token",
            {
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": approval["authorization_code"],
                "code_verifier": approval["code_verifier"],
                "redirect_uri": DEVICE_REDIRECT_URL,
            },
            form=True,
        )
        if status != 200:
            raise ChatGPTSignInError(f"the sign-in server refused the approved code ({status}): {reply}")
        self.save(tokens_from(reply))


def main(command: str, arguments: Sequence[str]) -> None:
    """``reef login chatgpt`` and ``reef logout chatgpt``."""
    if list(arguments) != ["chatgpt"]:
        print(f"usage: reef {command} chatgpt", file=sys.stderr)
        sys.exit(2)
    sign_in = ChatGPTSignIn.default()
    if command == "logout":
        print("Signed out of ChatGPT." if sign_in.sign_out() else "No ChatGPT sign-in to remove.")
        return
    try:
        sign_in.sign_in_with_device_code(sys.stdout)
    except ChatGPTSignInError as error:
        print(f"reef: {error}", file=sys.stderr)
        sys.exit(1)
    print(f"Signed in to ChatGPT; Reef keeps the sign-in in {sign_in.path}.")


__all__ = ["ChatGPTSignIn", "ChatGPTSignInError", "ChatGPTTokens", "main"]
