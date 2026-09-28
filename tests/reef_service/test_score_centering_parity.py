"""Score centering's per-token loss on CPU torch against a full-vocabulary reference.

The reference is the paper's estimator written out on the whole vocabulary:
the gradient on the trainer's logits of one position is
``-A (w_y s_y - sum_v q_v w_v s_v)`` with ``s_v = e_v - p`` (the score of
token ``v``). With the head covering the vocabulary the kernel must match it
exactly; with a smaller head it must match it exactly when the sampler's tail
is proportional to the trainer's (the approximation's assumption), which
separates approximation error from implementation error. The trainer's
log-probs come from the tensor-parallel gather at world size one.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reef.train.algos.score_centering import ScoreCenteringSettings
from reef.train.slime_backend.distill.objective import gather_log_probs_at_ids
from reef.train.slime_backend.score_centering.heads import PADDING_LOG_PROB
from reef.train.slime_backend.score_centering.objective import centered_token_loss, importance_weight

VOCAB = 7
WEIGHTS = (
    ScoreCenteringSettings(importance_weight="none"),
    ScoreCenteringSettings(importance_weight="tis", tis_cap=1.2),
    ScoreCenteringSettings(importance_weight="mis", mis_lower=0.7, mis_upper=1.4),
)


def distributions(seed: int, rows: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    trainer_logits = torch.randn(rows, VOCAB, generator=generator, dtype=torch.float64)
    sampler_log_probs = torch.log_softmax(
        trainer_logits + 0.5 * torch.randn(rows, VOCAB, generator=generator, dtype=torch.float64), dim=-1
    )
    return trainer_logits, sampler_log_probs


def kernel_gradient(
    trainer_logits: torch.Tensor,
    sampler_log_probs: torch.Tensor,
    head_ids: torch.Tensor,
    sampled: torch.Tensor,
    advantages: torch.Tensor,
    settings: ScoreCenteringSettings,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = trainer_logits.detach().clone().requires_grad_(True)
    trainer_at = gather_log_probs_at_ids(logits, torch.cat([head_ids, sampled[:, None]], dim=-1), None, 1, 0)
    result = centered_token_loss(
        trainer_at[:, :-1],
        trainer_at[:, -1],
        torch.gather(sampler_log_probs, -1, head_ids),
        torch.gather(sampler_log_probs, -1, sampled[:, None])[:, 0],
        advantages,
        settings,
    )
    result.loss.sum().backward()
    return logits.grad, result.correction


def reference_gradient(
    trainer_logits: torch.Tensor,
    sampler_log_probs: torch.Tensor,
    sampled: torch.Tensor,
    advantages: torch.Tensor,
    settings: ScoreCenteringSettings,
) -> torch.Tensor:
    """``-A (w_y s_y - sum_v q_v w_v s_v)`` per row, over the whole vocabulary."""
    probabilities = torch.softmax(trainer_logits, dim=-1)
    sampler = sampler_log_probs.exp()
    weights = importance_weight(probabilities / sampler, settings)
    scores = torch.eye(VOCAB, dtype=torch.float64)[None, :, :] - probabilities[:, None, :]  # [R, v, logit]
    rows = torch.arange(trainer_logits.size(0))
    sampled_term = weights[rows, sampled][:, None] * scores[rows, sampled]
    expected_term = ((sampler * weights)[:, :, None] * scores).sum(dim=1)
    return -advantages[:, None] * (sampled_term - expected_term)


@pytest.mark.unit
@pytest.mark.parametrize("settings", WEIGHTS, ids=lambda s: s.importance_weight)
def test_full_vocabulary_head_matches_the_exact_centered_gradient(settings: ScoreCenteringSettings) -> None:
    trainer_logits, sampler_log_probs = distributions(0)
    rows = trainer_logits.size(0)
    head_ids = torch.arange(VOCAB).expand(rows, VOCAB).contiguous()
    sampled = torch.tensor([0, 3, 6, 2])
    advantages = torch.tensor([1.0, -0.5, 2.0, 0.25], dtype=torch.float64)

    gradient, _ = kernel_gradient(trainer_logits, sampler_log_probs, head_ids, sampled, advantages, settings)

    torch.testing.assert_close(
        gradient, reference_gradient(trainer_logits, sampler_log_probs, sampled, advantages, settings)
    )


@pytest.mark.unit
@pytest.mark.parametrize("settings", WEIGHTS, ids=lambda s: s.importance_weight)
def test_constant_reward_has_zero_expected_update_under_the_exact_sampler(settings: ScoreCenteringSettings) -> None:
    # The drift the correction removes: E_{y ~ q}[gradient] is zero for any
    # sampler once the weighted score is centered, whatever the mismatch.
    trainer_logits, sampler_log_probs = distributions(1, rows=1)
    head_ids = torch.arange(VOCAB)[None, :]
    expected = torch.zeros_like(trainer_logits)
    for token in range(VOCAB):
        gradient, _ = kernel_gradient(
            trainer_logits,
            sampler_log_probs,
            head_ids,
            torch.tensor([token]),
            torch.ones(1, dtype=torch.float64),
            settings,
        )
        expected += sampler_log_probs[0, token].exp() * gradient
    torch.testing.assert_close(expected, torch.zeros_like(expected), atol=1e-12, rtol=0)


@pytest.mark.unit
@pytest.mark.parametrize("settings", WEIGHTS, ids=lambda s: s.importance_weight)
def test_matched_distributions_need_no_correction(settings: ScoreCenteringSettings) -> None:
    trainer_logits, _ = distributions(2)
    sampler_log_probs = torch.log_softmax(trainer_logits, dim=-1)
    head_ids = torch.topk(sampler_log_probs, k=3, dim=-1).indices
    sampled = head_ids[:, 0]
    advantages = torch.ones(trainer_logits.size(0), dtype=torch.float64)

    gradient, correction = kernel_gradient(trainer_logits, sampler_log_probs, head_ids, sampled, advantages, settings)

    torch.testing.assert_close(correction, torch.zeros_like(correction), atol=1e-12, rtol=0)
    # Without a correction the gradient is the plain on-policy -A s_y.
    scores = torch.eye(VOCAB, dtype=torch.float64)[sampled] - torch.softmax(trainer_logits, dim=-1)
    torch.testing.assert_close(gradient, -scores)


def proportional_tail_sampler(trainer_logits: torch.Tensor, head_ids: torch.Tensor, seed: int) -> torch.Tensor:
    """A sampler that differs from the trainer on the head and is ``rho p`` on the tail."""
    probabilities = torch.softmax(trainer_logits, dim=-1)
    generator = torch.Generator().manual_seed(seed)
    head_probabilities = torch.rand(head_ids.shape, generator=generator, dtype=torch.float64) + 0.1
    head_probabilities = 0.8 * head_probabilities / head_probabilities.sum(dim=-1, keepdim=True)
    in_head = torch.zeros_like(probabilities, dtype=torch.bool).scatter(-1, head_ids, True)
    tail = torch.where(in_head, torch.zeros_like(probabilities), probabilities)
    sampler = 0.2 * tail / tail.sum(dim=-1, keepdim=True)
    return sampler.scatter(-1, head_ids, head_probabilities).log()


@pytest.mark.unit
@pytest.mark.parametrize("settings", WEIGHTS, ids=lambda s: s.importance_weight)
def test_top_k_head_is_exact_when_the_sampler_tail_is_proportional(settings: ScoreCenteringSettings) -> None:
    trainer_logits, _ = distributions(3)
    head_ids = torch.tensor([[0, 1, 2], [4, 5, 6], [1, 3, 5], [6, 0, 2]])
    sampler_log_probs = proportional_tail_sampler(trainer_logits, head_ids, seed=3)
    # Sampled tokens inside and outside the head: the sampled weight uses the
    # token's own sampler probability either way.
    sampled = torch.tensor([0, 3, 6, 2])
    advantages = torch.tensor([1.0, -1.0, 0.5, 2.0], dtype=torch.float64)

    gradient, _ = kernel_gradient(trainer_logits, sampler_log_probs, head_ids, sampled, advantages, settings)

    torch.testing.assert_close(
        gradient, reference_gradient(trainer_logits, sampler_log_probs, sampled, advantages, settings)
    )


@pytest.mark.unit
def test_top_k_head_approximates_an_arbitrary_tail() -> None:
    # An arbitrary sampler tail is not rho p: the head-only estimate differs
    # from the exact one, but only by the tail's share.
    settings = ScoreCenteringSettings()
    trainer_logits, sampler_log_probs = distributions(4)
    head_ids = torch.topk(sampler_log_probs, k=5, dim=-1).indices
    sampled = head_ids[:, 0]
    advantages = torch.ones(trainer_logits.size(0), dtype=torch.float64)

    approximate, _ = kernel_gradient(trainer_logits, sampler_log_probs, head_ids, sampled, advantages, settings)
    exact = reference_gradient(trainer_logits, sampler_log_probs, sampled, advantages, settings)

    assert not torch.allclose(approximate, exact)
    tail_mass = 1.0 - torch.gather(sampler_log_probs.exp(), -1, head_ids).sum(dim=-1)
    assert torch.all((approximate - exact).abs().sum(dim=-1) <= 4 * tail_mass)


@pytest.mark.unit
def test_coefficients_are_detached() -> None:
    trainer_logits, sampler_log_probs = distributions(5, rows=1)
    head = torch.log_softmax(trainer_logits, dim=-1)[:, :3].clone().requires_grad_(True)
    sampled = torch.log_softmax(trainer_logits, dim=-1)[:, 4].clone().requires_grad_(True)
    advantages = torch.tensor([2.0], dtype=torch.float64)
    settings = ScoreCenteringSettings(importance_weight="tis", tis_cap=5.0)

    result = centered_token_loss(
        head, sampled, sampler_log_probs[:, :3], sampler_log_probs[:, 4], advantages, settings
    )
    result.loss.sum().backward()

    # d loss / d log p_y = -A w_y and d loss / d log p_v = A (q_v w_v - alpha p_v), coefficients held fixed.
    torch.testing.assert_close(sampled.grad, -advantages * result.sampled_weight)
    probabilities = head.detach().exp()
    sampler = sampler_log_probs[:, :3].exp()
    tail_ratio = (1 - sampler.sum(-1)) / (1 - probabilities.sum(-1))
    alpha = tail_ratio * importance_weight(1 / tail_ratio, settings)
    expected = advantages[:, None] * (
        sampler * importance_weight(probabilities / sampler, settings) - alpha[:, None] * probabilities
    )
    torch.testing.assert_close(head.grad, expected)


@pytest.mark.unit
def test_mis_drops_tokens_outside_the_ratio_band() -> None:
    settings = ScoreCenteringSettings(importance_weight="mis", mis_lower=0.5, mis_upper=2.0)
    ratios = torch.tensor([0.4, 0.5, 1.0, 2.0, 2.5], dtype=torch.float64)
    assert importance_weight(ratios, settings).tolist() == [0.0, 0.5, 1.0, 2.0, 0.0]
    tis = ScoreCenteringSettings(importance_weight="tis", tis_cap=2.0)
    assert importance_weight(ratios, tis).tolist() == [0.4, 0.5, 1.0, 2.0, 2.0]


@pytest.mark.unit
def test_nearly_empty_tails_are_clipped_and_counted() -> None:
    settings = ScoreCenteringSettings(importance_weight="tis", min_tail_mass=1e-6)
    trainer_logits = torch.tensor([[8.0, 7.0, -30.0, -30.0]], dtype=torch.float64)
    trainer_log_probs = torch.log_softmax(trainer_logits, dim=-1)
    sampler_log_probs = torch.log(torch.tensor([[0.6, 0.4 - 1e-12, 5e-13, 5e-13]], dtype=torch.float64))

    result = centered_token_loss(
        trainer_log_probs[:, :2].requires_grad_(True),
        trainer_log_probs[:, 0].requires_grad_(True),
        sampler_log_probs[:, :2],
        sampler_log_probs[:, 0],
        torch.ones(1, dtype=torch.float64),
        settings,
    )

    assert result.tail_clipped.tolist() == [1.0]
    assert torch.isfinite(result.loss).all()
    assert torch.isfinite(result.tail_ratio).all()


@pytest.mark.unit
def test_padded_positions_stay_finite() -> None:
    # The builder's placeholder head for an untrained position: probability zero everywhere.
    trainer_logits, _ = distributions(6, rows=1)
    head_ids = torch.zeros(1, 3, dtype=torch.long)
    logits = trainer_logits.clone().requires_grad_(True)
    trainer_at = gather_log_probs_at_ids(logits, torch.cat([head_ids, torch.tensor([[2]])], dim=-1), None, 1, 0)
    for settings in WEIGHTS:
        result = centered_token_loss(
            trainer_at[:, :-1],
            trainer_at[:, -1],
            torch.full((1, 3), PADDING_LOG_PROB, dtype=torch.float32),
            torch.zeros(1, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            settings,
        )
        assert torch.isfinite(result.loss).all()
        (result.loss * 0).sum().backward(retain_graph=True)
        assert torch.isfinite(logits.grad).all()


def sharded_worker(rank: int, world: int, port: int) -> None:
    """One tensor-parallel rank: the gradient on its vocab shard must be the dense gradient's slice."""
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    try:
        vocab = 24
        generator = torch.Generator().manual_seed(7)
        trainer_logits = torch.randn(3, vocab, generator=generator, dtype=torch.float64) * 2
        sampler_log_probs = torch.log_softmax(
            trainer_logits + torch.randn(3, vocab, generator=generator, dtype=torch.float64), dim=-1
        )
        head_ids = torch.topk(sampler_log_probs, k=5, dim=-1).indices
        ids = torch.cat([head_ids, torch.tensor([[1], [23], [head_ids[2, 0]]])], dim=-1)
        advantages = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
        settings = ScoreCenteringSettings(importance_weight="tis")
        shard = slice(rank * vocab // world, (rank + 1) * vocab // world)

        def _centered_loss(logits: torch.Tensor, group: object, tp_world: int, tp_rank: int) -> torch.Tensor:
            trainer_at = gather_log_probs_at_ids(logits, ids, group, tp_world, tp_rank)
            return centered_token_loss(
                trainer_at[:, :-1],
                trainer_at[:, -1],
                torch.gather(sampler_log_probs, -1, ids[:, :-1]),
                torch.gather(sampler_log_probs, -1, ids[:, -1:])[:, 0],
                advantages,
                settings,
            ).loss

        dense = trainer_logits.clone().requires_grad_(True)
        dense_loss = _centered_loss(dense, None, 1, 0)
        dense_loss.sum().backward()
        local = trainer_logits[:, shard].clone().requires_grad_(True)
        local_loss = _centered_loss(local, dist.group.WORLD, world, rank)
        local_loss.sum().backward()
        assert torch.allclose(local_loss, dense_loss.detach(), atol=1e-9), (rank, local_loss, dense_loss)
        assert torch.allclose(local.grad, dense.grad[:, shard], atol=1e-9), (rank, local.grad, dense.grad[:, shard])
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
def test_sharded_loss_matches_the_dense_computation_across_vocab_shards() -> None:
    import socket

    import torch.multiprocessing as multiprocessing

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    multiprocessing.spawn(sharded_worker, args=(4, port), nprocs=4, join=True)
