"""The bridge-side check of the sampler's recorded top-K, vectorized with torch.

``ScoreCenteringAlgorithm.prepare_rollout`` imports this module on the
bridge, which tensorizes the payload next anyway, so the spec module stays
importable without torch. The check costs a few tensor operations per sample
instead of Python work per recorded entry, which grows with
``positions * top_k``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

#: Log-prob given to the placeholder head of an untrained position; its
#: probability underflows to exactly zero in float32.
PADDING_LOG_PROB = -1.0e4
#: Slack for float rounding when a recorded head is checked against a distribution.
PROBABILITY_TOLERANCE = 1e-3


def recorded_rows(label: str, name: str, rows: object, response_length: int, top_k: int) -> Sequence[object]:
    """One sample's recorded top-K column as a sequence with a row per response token."""
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
        raise ValueError(f"{label} {name} must be a sequence of rows")
    if not rows:
        raise ValueError(
            f"{label} carries no sampler top-k log-probs: score centering needs a token-native inference "
            f"handler with capture_topk >= {top_k}"
        )
    if len(rows) != response_length:
        raise ValueError(f"{label} {name} has {len(rows)} rows for a {response_length}-token response")
    return rows


def sampler_head(
    label: str,
    indices: object,
    log_probs: object,
    *,
    response_tokens: Sequence[int],
    loss_mask: Sequence[int],
    rollout_log_probs: Sequence[float],
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate one sample's recorded top-K and return its ``[R, top_k]`` head ids and log-probs.

    Every trained position must carry at least ``top_k`` distinct
    non-negative ids with finite log-probs whose probabilities sum to at most
    one; the first ``top_k`` are kept. When the sampled token is in the head
    its recorded log-prob must equal the sampled log-prob, which catches a
    head shifted against the response. Untrained positions (the masked
    context multi-turn assembly inserts) get a placeholder head of
    probability zero.
    """
    response_length = len(loss_mask)
    index_rows = recorded_rows(label, "topk_indices", indices, response_length, top_k)
    log_prob_rows = recorded_rows(label, "topk_log_probs", log_probs, response_length, top_k)
    trained = [position for position, flag in enumerate(loss_mask) if flag]
    trained_ids: list[Sequence[object]] = []
    trained_log_probs: list[Sequence[object]] = []
    for position in trained:
        id_row, log_prob_row = index_rows[position], log_prob_rows[position]
        if (
            not isinstance(id_row, Sequence)
            or not isinstance(log_prob_row, Sequence)
            or isinstance(id_row, str | bytes)
            or isinstance(log_prob_row, str | bytes)
        ):
            raise ValueError(f"{label} position {position} top-k rows must be sequences")
        if len(id_row) < top_k or len(log_prob_row) < top_k:
            raise ValueError(
                f"{label} position {position} records {min(len(id_row), len(log_prob_row))} top-k entries, "
                f"fewer than top_k={top_k}; raise the inference handler's capture_topk"
            )
        trained_ids.append(id_row[:top_k])
        trained_log_probs.append(log_prob_row[:top_k])

    head_ids = torch.zeros((response_length, top_k), dtype=torch.long)
    head_log_probs = torch.full((response_length, top_k), PADDING_LOG_PROB, dtype=torch.float32)
    if not trained:
        return head_ids, head_log_probs
    try:
        ids = torch.tensor(trained_ids)
        values = torch.tensor(trained_log_probs, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{label} top-k rows must hold numbers: {error}") from error
    sampled = torch.tensor([response_tokens[position] for position in trained])
    sampled_log_probs = torch.tensor([rollout_log_probs[position] for position in trained], dtype=torch.float64)

    def _first_position(bad_rows: torch.Tensor) -> int:
        return trained[int(bad_rows.nonzero()[0, 0])]

    if ids.dtype != torch.long:
        raise ValueError(f"{label} topk_indices must be non-negative integers")
    negative = (ids < 0).any(-1)
    if bool(negative.any()):
        raise ValueError(f"{label} position {_first_position(negative)} topk_indices must be non-negative integers")
    ordered = ids.sort(dim=-1).values
    repeated = (ordered[:, 1:] == ordered[:, :-1]).any(-1)
    if bool(repeated.any()):
        raise ValueError(f"{label} position {_first_position(repeated)} topk_indices must be distinct")
    invalid = ~torch.isfinite(values).all(-1) | (values > PROBABILITY_TOLERANCE).any(-1)
    if bool(invalid.any()):
        raise ValueError(
            f"{label} position {_first_position(invalid)} topk_log_probs must be finite log-probabilities"
        )
    overfull = values.exp().sum(-1) > 1.0 + PROBABILITY_TOLERANCE
    if bool(overfull.any()):
        raise ValueError(f"{label} position {_first_position(overfull)} top-k probabilities sum to more than one")
    in_head = ids == sampled[:, None]
    recorded = (values * in_head).sum(-1)
    misaligned = in_head.any(-1) & ((recorded - sampled_log_probs).abs() > PROBABILITY_TOLERANCE)
    if bool(misaligned.any()):
        row = int(misaligned.nonzero()[0, 0])
        raise ValueError(
            f"{label} position {trained[row]} records the sampled token {int(sampled[row])} at log-prob "
            f"{float(recorded[row])} but samples it at {float(sampled_log_probs[row])}: the top-k rows are not "
            "aligned with the response"
        )
    rows = torch.tensor(trained)
    head_ids[rows] = ids
    head_log_probs[rows] = values.float()
    return head_ids, head_log_probs


__all__ = ["PADDING_LOG_PROB", "PROBABILITY_TOLERANCE", "recorded_rows", "sampler_head"]
