"""Validate and normalize training-request bodies without constructing queued requests."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import NotRequired, TypedDict

from reef.core.requirements import parse_requires

ClientReport = dict[str, str | dict[str, bool]]


class TrainingRequestPayload(TypedDict):
    """Normalized fields stored in a TRAIN record; identity comes from its envelope."""

    text: str
    session: str
    release_id: str
    requires: list[dict[str, str]]
    client: NotRequired[ClientReport]


#: At most this many commands in one report.
MAX_CLIENT_COMMANDS = 64
CLIENT_WORD = re.compile(r"[A-Za-z0-9._+\- ()]{1,64}")
COMMAND_NAME = re.compile(r"[A-Za-z0-9._+\-]{1,40}")


def parse_client(value: object) -> ClientReport:
    """A client's report of its machine: ``platform``, ``arch`` and ``release`` as short words, and ``commands``
    mapping a command name to whether it is on the PATH, at most ``MAX_CLIENT_COMMANDS``. The report only
    informs a proposer, so what does not fit that shape is dropped rather than refusing the request; it is the
    client's word, data a proposer reads, never an instruction."""
    if not isinstance(value, Mapping):
        return {}
    parsed: ClientReport = {}
    for key in ("platform", "arch", "release"):
        word = value.get(key)
        if isinstance(word, str) and CLIENT_WORD.fullmatch(word.strip()):
            parsed[key] = word.strip()
    commands = value.get("commands")
    if isinstance(commands, Mapping):
        kept = {
            name: present
            for name, present in commands.items()
            if isinstance(name, str) and COMMAND_NAME.fullmatch(name) and isinstance(present, bool)
        }
        if kept:
            parsed["commands"] = dict(list(kept.items())[:MAX_CLIENT_COMMANDS])
    return parsed


def parse_training_request_fields(payload: Mapping[str, object]) -> tuple[str, str, str]:
    """Read the required text, session and release strings for normalization or construction."""
    fields: dict[str, str] = {}
    for key in ("text", "session", "release_id"):
        value = payload.get(key)
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        fields[key] = value
    return fields["text"], fields["session"], fields["release_id"]


def normalize_training_request_payload(payload: Mapping[str, object]) -> TrainingRequestPayload:
    """Validate a request and return its wire fields, dropping unknown keys.

    Invalid instruction fields or requirements raise ValueError. Client machine
    information is advisory: malformed entries are dropped. Text is preserved,
    absent requirements become an empty list, and an empty client is omitted.
    """
    text, session, release_id = parse_training_request_fields(payload)
    if not text.strip():
        raise ValueError("text must be a non-empty string")
    if len(text) > 4000:
        raise ValueError("text must not exceed 4000 characters")
    normalized: TrainingRequestPayload = {
        "text": text,
        "session": session,
        "release_id": release_id,
        "requires": parse_requires(payload.get("requires")),
    }
    client = parse_client(payload.get("client"))
    if client:
        normalized["client"] = client
    return normalized
