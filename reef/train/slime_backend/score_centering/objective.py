"""The worker side of Score Centering: the centered per-token loss and its ``@objective`` hook.

:func:`centered_token_loss` is the paper's equation 14 on replicated
log-probs; :func:`score_centering_loss` gathers the trainer's log-probs at
the sampler's head ids and the sampled token across the tensor-parallel
vocab shards (the distillation package's differentiable gather) and reduces
the per-token loss with Slime's per-sample mean. The kernel takes plain
tensors, so the CPU tests check it against a full-vocabulary reference.
Megatron and Slime are imported where the hook runs.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from reef.train.algos.score_centering import ScoreCenteringSettings
from reef.train.slime_backend.algorithm import objective
from reef.train.slime_backend.distill.objective import gather_log_probs_at_ids
from reef.train.slime_backend.score_centering import settings_from_args


@dataclass(frozen=True)
class CenteredTokenLoss:
    """Per-token loss and the per-token quantities the step reports, all ``[R]``.

    ``correction`` is the L1 norm of the centering coefficients
    ``q_v w_v - alpha p_v``; ``tail_clipped`` marks positions where either
    tail mass fell below ``min_tail_mass``.
    """

    loss: torch.Tensor
    sampled_weight: torch.Tensor
    correction: torch.Tensor
    sampler_head_mass: torch.Tensor
    trainer_head_mass: torch.Tensor
    tail_ratio: torch.Tensor
    tail_clipped: torch.Tensor


def importance_weight(ratio: torch.Tensor, settings: ScoreCenteringSettings) -> torch.Tensor:
    """``f(r)`` of the configured importance weight, elementwise."""
    if settings.importance_weight == "tis":
        return ratio.clamp(max=settings.tis_cap)
    if settings.importance_weight == "mis":
        inside = (ratio >= settings.mis_lower) & (ratio <= settings.mis_upper)
        return torch.where(inside, ratio, torch.zeros_like(ratio))
    return torch.ones_like(ratio)


def centered_token_loss(
    trainer_head_log_probs: torch.Tensor,
    trainer_sampled_log_prob: torch.Tensor,
    sampler_head_log_probs: torch.Tensor,
    sampler_sampled_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    settings: ScoreCenteringSettings,
) -> CenteredTokenLoss:
    """``-A (sg[w_y] log p_y - sum_head sg[q_v w_v - alpha p_v] log p_v)`` per position.

    ``trainer_head_log_probs`` (``[R, K]``) and ``trainer_sampled_log_prob``
    (``[R]``) are the trainer's full-vocabulary log-probs at the sampler's
    head ids and at the sampled token, differentiable; the sampler's are the
    recorded ones. Every coefficient is detached, so the gradient on the
    trainer's log-probs is ``-A (w_y s_y - sum_head (q_v w_v - alpha p_v) s_v)``.
    The sampled token's weight uses its own recorded sampler log-prob, also
    when it lies outside the head.
    """
    dtype = torch.promote_types(trainer_head_log_probs.dtype, torch.float32)
    trainer_head = trainer_head_log_probs.to(dtype)
    trainer_sampled = trainer_sampled_log_prob.to(dtype)
    with torch.no_grad():
        sampler_head_probs = sampler_head_log_probs.to(dtype).exp()
        trainer_head_probs = trainer_head.detach().exp()
        sampler_head_mass = sampler_head_probs.sum(dim=-1)
        trainer_head_mass = trainer_head_probs.sum(dim=-1)
        sampler_tail = 1.0 - sampler_head_mass
        trainer_tail = 1.0 - trainer_head_mass
        tail_clipped = (sampler_tail < settings.min_tail_mass) | (trainer_tail < settings.min_tail_mass)
        tail_ratio = sampler_tail.clamp(min=settings.min_tail_mass) / trainer_tail.clamp(min=settings.min_tail_mass)
        # A tail token's ratio p / q is 1 / rho under the q = rho p approximation.
        alpha = tail_ratio * importance_weight(1.0 / tail_ratio, settings)
        head_weight = importance_weight((trainer_head.detach() - sampler_head_log_probs.to(dtype)).exp(), settings)
        coefficients = sampler_head_probs * head_weight - alpha[:, None] * trainer_head_probs
        sampled_weight = importance_weight(
            (trainer_sampled.detach() - sampler_sampled_log_prob.to(dtype)).exp(), settings
        )
    centered = sampled_weight * trainer_sampled - (coefficients * trainer_head).sum(dim=-1)
    return CenteredTokenLoss(
        loss=-advantages.to(dtype) * centered,
        sampled_weight=sampled_weight,
        correction=coefficients.abs().sum(dim=-1),
        sampler_head_mass=sampler_head_mass,
        trainer_head_mass=trainer_head_mass,
        tail_ratio=tail_ratio,
        tail_clipped=tail_clipped.to(dtype),
    )


@objective("custom_loss_function_path")
def score_centering_loss(
    args: Namespace,
    batch: dict[str, Any],
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``--custom-loss-function-path`` entry point: the centered policy gradient's per-sample mean.

    Slime's outer ``loss_function`` divides the returned sum of per-sample
    means by the step's global batch size, which yields the batch mean.
    """
    from megatron.core import mpu
    from slime.backends.megatron_utils.loss import get_responses

    if mpu.get_context_parallel_world_size() > 1:
        raise NotImplementedError("the score centering loss supports context parallel = 1 only")
    settings = settings_from_args(args)
    for key in ("advantages", "rollout_log_probs", "topk_indices", "topk_log_probs"):
        if batch.get(key) is None:
            raise ValueError(f"the score centering loss needs {key} in the batch")
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_world = dist.get_world_size(group=tp_group) if dist.is_initialized() else 1
    tp_rank = dist.get_rank(group=tp_group) if dist.is_initialized() else 0

    per_sample: list[CenteredTokenLoss] = []
    responses = get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
    )
    for index, ((rows, sampled_tokens), head_ids, head_log_probs, rollout_log_probs, advantages) in enumerate(
        zip(
            responses,
            batch["topk_indices"],
            batch["topk_log_probs"],
            batch["rollout_log_probs"],
            batch["advantages"],
            strict=True,
        )
    ):
        device = rows.device
        head_ids = head_ids.to(device=device, dtype=torch.long)
        if head_ids.size(0) != rows.size(0) or head_ids.size(-1) != settings.top_k:
            raise ValueError(
                f"score centering sample {index} has a {tuple(head_ids.shape)} head for a {rows.size(0)}-token "
                f"response and top_k={settings.top_k}"
            )
        vocab_size = rows.size(-1) * tp_world
        if head_ids.numel() and int(head_ids.max()) >= vocab_size:
            raise ValueError(f"score centering sample {index} names a token id outside the {vocab_size}-token vocab")
        sampled = sampled_tokens.to(device=device, dtype=torch.long)[:, None]
        trainer_at = gather_log_probs_at_ids(rows, torch.cat([head_ids, sampled], dim=-1), tp_group, tp_world, tp_rank)
        per_sample.append(
            centered_token_loss(
                trainer_at[:, :-1],
                trainer_at[:, -1],
                head_log_probs.to(device=device),
                rollout_log_probs.to(device=device),
                advantages.to(device=device),
                settings,
            )
        )

    token_loss = torch.cat([sample.loss for sample in per_sample], dim=0)
    loss = sum_of_sample_mean(token_loss)
    if token_loss.numel() == 0:
        loss = loss + 0 * logits.sum()
    # Slime sums a micro-batch's metrics and divides the step's total by the
    # global batch size, so every value is a sum of per-sample means.
    reported = {
        "sampled_weight": [sample.sampled_weight for sample in per_sample],
        "correction_l1": [sample.correction for sample in per_sample],
        "sampler_head_mass": [sample.sampler_head_mass for sample in per_sample],
        "trainer_head_mass": [sample.trainer_head_mass for sample in per_sample],
        "tail_ratio": [sample.tail_ratio for sample in per_sample],
        "tail_clipped": [sample.tail_clipped for sample in per_sample],
    }
    metrics = {"loss": loss.detach().clone(), "pg_loss": loss.detach().clone()}
    for name, values in reported.items():
        metrics[f"score_centering_{name}"] = sum_of_sample_mean(torch.cat(values, dim=0))
    return loss, metrics


__all__ = ["CenteredTokenLoss", "centered_token_loss", "importance_weight", "score_centering_loss"]
