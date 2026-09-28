"""Backend-neutral contract of score centering, a correction added to off-policy policy-gradient losses.

Score centering (Marek and Ryabinin, arXiv:2609.20807) removes the drift an
off-policy policy gradient accumulates when the sampler ``q`` differs from
the trained policy ``p`` (quantized inference, stale weights). A loss whose
per-token gradient is ``A_t * f(p_t / q_t) * grad log p_t`` pulls ``p``
toward ``q`` by ``A_t * E_q[f(p / q) * score]`` at every prefix; score
centering adds the term that subtracts it (the paper's Appendix A, equations
9-14)::

    A_t * sum_{v in head} sg[q_v * f(p_v / q_v) - alpha * p_v] * log p_v

The head is the sampler's recorded top-K ids; the sampler's tail is
approximated as ``rho * p`` with ``rho = (1 - q(head)) / (1 - p(head))`` and
``alpha = rho * f(1 / rho)``. The term is zero when ``q = p``.

The term must mirror its loss: the same advantages, the same weight ``f``,
the same mask and reduction. A loss therefore declares its weight as a
:class:`PolicyGradientWeight`; :class:`ScoreCenteringSettings` holds the two
settings of the term itself. Torch-free, so the service validates both without
a training stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

#: ``none`` is ``f(r) = 1``; ``truncated`` is ``min(r, upper)``; ``masked`` is
#: ``r`` strictly inside ``(lower, upper)`` and 0 outside.
WEIGHT_KINDS = ("none", "truncated", "masked")


def is_finite_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class PolicyGradientWeight:
    """The weight ``f(r)``, ``r = p / q``, a policy-gradient loss puts on the sampled token's score.

    A loss of the form ``-A_t * sg[f(p_t / q_t)] * log p_t`` declares its
    ``f`` with this value: plain off-policy REINFORCE is ``none``, truncated
    importance sampling ``truncated`` with its cap as ``upper``, and masked
    importance sampling ``masked`` with its trust region as ``(lower,
    upper)``.
    """

    kind: str
    lower: float = 0.0
    upper: float = math.inf

    def __post_init__(self) -> None:
        if self.kind not in WEIGHT_KINDS:
            raise ValueError(f"policy-gradient weight kind must be one of: {', '.join(WEIGHT_KINDS)}")
        if self.kind == "truncated" and (not is_finite_number(self.upper) or self.upper <= 0):
            raise ValueError("a truncated policy-gradient weight needs a finite cap upper > 0")
        if self.kind == "masked" and (
            not is_finite_number(self.lower) or not is_finite_number(self.upper) or not 0 <= self.lower < self.upper
        ):
            raise ValueError("a masked policy-gradient weight needs finite bounds 0 <= lower < upper")


@dataclass(frozen=True)
class ScoreCenteringSettings:
    """The term's own settings.

    ``top_k`` is the number of sampler log-probs per position the term reads;
    a record must carry at least that many (the inference handler's
    ``capture_topk``). The paper's default is 128; a smaller head leaves more
    of the drift uncorrected when the sampler's tail differs from the
    trainer's. ``min_tail_mass`` is the floor both tail masses are clipped to
    before their ratio is taken (the paper uses ``1e-6``).
    """

    top_k: int = 128
    min_tail_mass: float = 1e-6

    def __post_init__(self) -> None:
        if not isinstance(self.top_k, Integral) or isinstance(self.top_k, bool) or self.top_k <= 0:
            raise ValueError("score centering top_k must be a positive integer")
        if not is_finite_number(self.min_tail_mass) or not 0 < self.min_tail_mass < 1:
            raise ValueError("score centering min_tail_mass must be a number in (0, 1)")


__all__ = ["WEIGHT_KINDS", "PolicyGradientWeight", "ScoreCenteringSettings", "is_finite_number"]
