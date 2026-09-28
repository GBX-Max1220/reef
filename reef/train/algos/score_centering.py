"""Backend-neutral settings of Score Centering, an additive correction to off-policy policy gradients.

Score Centering (Marek and Ryabinin, arXiv:2609.20807) removes the drift an
off-policy policy gradient accumulates when the sampler ``q`` differs from
the trained policy ``p`` (quantized inference, stale weights): at every
response position it subtracts the sampler's expected score from the sampled
token's score. Only the sampler's top-K log-probs are recorded, so the
sampler's tail is approximated as ``rho * p`` with
``rho = (1 - q(head)) / (1 - p(head))`` (the paper's Appendix A, equations
9-14). The correction composes with a truncated (``tis``) or masked
(``mis``) importance weight ``w = f(p / q)``; the weighted score is
centered, tail included.

A backend implements the loss; these settings are what a recipe configures.
Torch-free, so the service validates them without a training stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

#: ``none`` centers the plain score, ``tis`` weights by ``min(p / q, tis_cap)``,
#: ``mis`` by ``p / q`` inside ``[mis_lower, mis_upper]`` and 0 outside.
IMPORTANCE_WEIGHTS = ("none", "tis", "mis")


@dataclass(frozen=True)
class ScoreCenteringSettings:
    """What the correction does at every trained response position.

    ``top_k`` is the number of sampler log-probs per position the correction
    reads; a record must carry at least that many (the inference handler's
    ``capture_topk``). The paper's default is 128; 32 was also effective.
    ``min_tail_mass`` is the floor both tail masses are clipped to before
    their ratio is taken, so a head holding nearly all the mass cannot divide
    by zero (the paper uses ``1e-6``).
    """

    top_k: int = 128
    importance_weight: str = "none"
    tis_cap: float = 2.0
    mis_lower: float = 0.5
    mis_upper: float = 5.0
    min_tail_mass: float = 1e-6

    def __post_init__(self) -> None:
        if not isinstance(self.top_k, Integral) or isinstance(self.top_k, bool) or self.top_k <= 0:
            raise ValueError("score centering top_k must be a positive integer")
        if self.importance_weight not in IMPORTANCE_WEIGHTS:
            raise ValueError(f"score centering importance_weight must be one of: {', '.join(IMPORTANCE_WEIGHTS)}")
        if not is_finite_number(self.tis_cap) or self.tis_cap <= 0:
            raise ValueError("score centering tis_cap must be a finite number > 0")
        if (
            not is_finite_number(self.mis_lower)
            or not is_finite_number(self.mis_upper)
            or not 0 <= self.mis_lower < self.mis_upper
        ):
            raise ValueError("score centering needs finite 0 <= mis_lower < mis_upper")
        if not is_finite_number(self.min_tail_mass) or not 0 < self.min_tail_mass < 1:
            raise ValueError("score centering min_tail_mass must be a number in (0, 1)")


def is_finite_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


__all__ = ["IMPORTANCE_WEIGHTS", "ScoreCenteringSettings"]
