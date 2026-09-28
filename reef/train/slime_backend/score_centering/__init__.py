"""Score Centering on the Slime backend: a policy gradient whose sampled score is centered under the sampler.

A recipe opts in by giving its objective ``loss_family = "score_centering"``
and supplying one advantage per sample, the way it would for any policy
family; a recipe that wants its own flag prefix subclasses
:class:`ScoreCenteringAlgorithm` and forwards its ``objective.py`` hooks to
:mod:`.objective`. The per-token loss (the paper's equation 14) is

    -A * (sg[w_y] log p_y - sum_{v in head} sg[q_v w_v - alpha p_v] log p_v)

with ``p`` the trainer's distribution, ``q`` the sampler's, the head the
sampler's recorded top-K ids, ``w = f(p / q)`` the importance weight of
:class:`~reef.train.algos.score_centering.ScoreCenteringSettings` and
``alpha = rho f(1 / rho)`` the tail's weighted share. It is reduced with
Slime's per-sample mean over the trained response tokens.

This module is the driver side, torch-free: the flags and the seven-column
wire row (the policy row plus the sampler's ``topk_indices`` and
``topk_log_probs`` per response token). The bridge checks the rows with
:mod:`.heads` and the workers compute the loss in :mod:`.objective`, both
torch. The sampler's log-probs are those at the point the trainer reads
(after temperature, before top-k, top-p and min-p filters; see the
configuration reference), so records must come from a token-native handler
with ``capture_topk`` at least the configured ``top_k``.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from collections.abc import Mapping, Sequence
from typing import Any

from reef.core.trajectories import source_record_id, trajectory_reward
from reef.train.algos.score_centering import IMPORTANCE_WEIGHTS, ScoreCenteringSettings
from reef.train.slime_backend.algorithm import SlimeAlgorithm, register_loss_family
from reef.train.slime_backend.data_builder import build_policy_rollout_data
from reef.train.types import TrajectoryItem

ROW_SHAPE = "[source_id, tokens, loss_mask, rollout_log_probs, reward, topk_indices, topk_log_probs]"


def score_centering_sample_row(sample: TrajectoryItem) -> list[Any]:
    """Shape one Reef sample into the family's seven-column wire row."""
    return [
        source_record_id(sample),
        list(sample.training.get("tokens", [])),
        list(sample.training.get("loss_mask", [])),
        list(sample.training.get("rollout_log_probs", [])),
        trajectory_reward(sample),
        [list(row) for row in sample.training.get("topk_indices", [])],
        [list(row) for row in sample.training.get("topk_log_probs", [])],
    ]


@register_loss_family
class ScoreCenteringAlgorithm(SlimeAlgorithm):
    """A policy gradient on the recipe's advantages, with the sampler's expected score subtracted.

    The settings travel on ``args`` under ``score_centering_*`` names whatever
    the family's flag prefix, so the worker hooks read one contract.
    """

    loss_family = "score_centering"
    loss_type = "custom_loss"
    # The sampled token's importance weight and sampler probability come from
    # the rollout engine's log-probs.
    requires_rollout_logprobs = True
    advantages = "required"
    rollout_data_keys = ("topk_indices", "topk_log_probs")
    rollout_tensor_dtypes: Mapping[str, str] = {"topk_indices": "long", "topk_log_probs": "float32"}
    external_batch_keys = ("rollout_log_probs", "topk_indices", "topk_log_probs")
    rollout_log_skip_keys = ("topk_indices", "topk_log_probs")
    required_objective_hooks = ("custom_loss_function_path",)

    def __init__(self, settings: ScoreCenteringSettings | None = None) -> None:
        self.settings = settings or ScoreCenteringSettings()

    # --- stage 1: configure ---

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        if int(args.context_parallel_size or 1) != 1:
            raise RuntimeError(f"{source} supports --context-parallel-size 1 only: the top-k rows are not CP-sliced")
        # Slime applies these inside its stock policy loss, which this family
        # replaces; accepting them would silently train without them.
        if args.use_tis:
            raise RuntimeError(
                f"{source} does not read --use-tis; select the weight with "
                f"--{self.flag_prefix()}-importance-weight tis or mis"
            )
        if args.use_kl_loss or args.entropy_coef:
            raise RuntimeError(f"{source} does not apply --use-kl-loss or --entropy-coef")

    def flag_prefix(self) -> str:
        """The family's flag prefix, ``score-centering`` for the built-in family."""
        return self.loss_family.replace("_", "-")

    def parse_specific_options(self, arguments: Sequence[str]) -> tuple[ScoreCenteringSettings, list[str]]:
        prefix = f"--{self.flag_prefix()}-"
        defaults = ScoreCenteringSettings()
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, argument_default=argparse.SUPPRESS)
        parser.add_argument(
            f"{prefix}top-k",
            dest="top_k",
            type=int,
            help=(
                "Sampler log-probs read per response position; records need capture_topk at least this large. "
                f"Default {defaults.top_k}."
            ),
        )
        parser.add_argument(
            f"{prefix}importance-weight",
            dest="importance_weight",
            choices=list(IMPORTANCE_WEIGHTS),
            help=(
                "The weight of the centered score: 'none', 'tis' (min(p/q, tis-cap)) or 'mis' (p/q inside "
                f"[mis-lower, mis-upper], else 0). Default {defaults.importance_weight}."
            ),
        )
        parser.add_argument(
            f"{prefix}tis-cap", dest="tis_cap", type=float, help=f"TIS cap. Default {defaults.tis_cap}."
        )
        parser.add_argument(
            f"{prefix}mis-lower", dest="mis_lower", type=float, help=f"MIS lower bound. Default {defaults.mis_lower}."
        )
        parser.add_argument(
            f"{prefix}mis-upper", dest="mis_upper", type=float, help=f"MIS upper bound. Default {defaults.mis_upper}."
        )
        parser.add_argument(
            f"{prefix}min-tail-mass",
            dest="min_tail_mass",
            type=float,
            help=(
                "Floor of the sampler's and trainer's tail masses before their ratio is taken. "
                f"Default {defaults.min_tail_mass}."
            ),
        )
        options, remaining = parser.parse_known_args(list(arguments))
        return ScoreCenteringSettings(**vars(options)), remaining

    def apply_driver_options(self, args: Namespace, options: object | None) -> None:
        super().apply_driver_options(args, options)
        settings = options if isinstance(options, ScoreCenteringSettings) else ScoreCenteringSettings()
        args.score_centering_top_k = settings.top_k
        args.score_centering_importance_weight = settings.importance_weight
        args.score_centering_tis_cap = settings.tis_cap
        args.score_centering_mis_lower = settings.mis_lower
        args.score_centering_mis_upper = settings.mis_upper
        args.score_centering_min_tail_mass = settings.min_tail_mass

    def bind(
        self,
        config: object | None = None,
        *,
        critic_steps_per_actor: int | None = None,
        critic_only_steps: int = 0,
    ) -> ScoreCenteringAlgorithm:
        # The worker reads the settings off args; the bound instance keeps
        # them for the bridge-side check of each payload's top-k rows.
        if config is None:
            return self.__class__(ScoreCenteringSettings())
        if not isinstance(config, ScoreCenteringSettings):
            raise TypeError(f"{self.loss_family} bridge algorithm config must be ScoreCenteringSettings")
        return self.__class__(config)

    # --- stage 2: shape row ---

    def shape_sample_row(self, sample: TrajectoryItem) -> list[Any]:
        return score_centering_sample_row(sample)

    # --- stage 3: build batch ---

    def build_rollout_data(self, payload: Mapping[str, Any], samples: Sequence) -> dict:
        name = self.loss_family
        base_rows: list[list[Any]] = []
        indices: list[Any] = []
        log_probs: list[Any] = []
        for index, row in enumerate(samples):
            if not isinstance(row, Sequence) or isinstance(row, str | bytes) or len(row) != 7:
                raise ValueError(f"{name} sample {index} must be {ROW_SHAPE}")
            base_rows.append(list(row[:5]))
            indices.append(row[5])
            log_probs.append(row[6])
        data = build_policy_rollout_data({**dict(payload), "samples": base_rows}, base_rows, self)
        # Checked against the configured top_k in prepare_rollout, where the
        # bound settings are known.
        data["topk_indices"] = indices
        data["topk_log_probs"] = log_probs
        return data

    # --- stage 4: prepare rollout ---

    def prepare_rollout(self, rollout_data: dict[str, Any]) -> dict[str, Any]:
        """Replace the recorded top-k rows with validated ``[R, top_k]`` head tensors."""
        # The bridge tensorizes the payload next; the check runs there, in torch.
        from reef.train.slime_backend.score_centering.heads import sampler_head

        name = self.loss_family
        rollout_log_probs = rollout_data.get("rollout_log_probs")
        if not rollout_log_probs:
            raise ValueError(f"{name} requires one rollout_log_prob per response token")
        heads = [
            sampler_head(
                f"{name} sample {index}",
                row_indices,
                row_log_probs,
                response_tokens=tokens[len(tokens) - len(loss_mask) :],
                loss_mask=loss_mask,
                rollout_log_probs=sampled_log_probs,
                top_k=self.settings.top_k,
            )
            for index, (row_indices, row_log_probs, tokens, loss_mask, sampled_log_probs) in enumerate(
                zip(
                    rollout_data["topk_indices"],
                    rollout_data["topk_log_probs"],
                    rollout_data["tokens"],
                    rollout_data["loss_masks"],
                    rollout_log_probs,
                    strict=True,
                )
            )
        ]
        rollout_data["topk_indices"] = [ids for ids, _ in heads]
        rollout_data["topk_log_probs"] = [values for _, values in heads]
        return {}


def settings_from_args(args: Namespace) -> ScoreCenteringSettings:
    """The family's settings as the driver stamped them on ``args`` (worker side)."""
    return ScoreCenteringSettings(
        top_k=args.score_centering_top_k,
        importance_weight=args.score_centering_importance_weight,
        tis_cap=args.score_centering_tis_cap,
        mis_lower=args.score_centering_mis_lower,
        mis_upper=args.score_centering_mis_upper,
        min_tail_mass=args.score_centering_min_tail_mass,
    )


__all__ = ["ROW_SHAPE", "ScoreCenteringAlgorithm", "score_centering_sample_row", "settings_from_args"]
