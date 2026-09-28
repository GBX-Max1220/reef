"""The driver side of the built-in score-centering family: settings, flags, wire row and top-k validation."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from reef_service._trajectories import policy_trajectory

from reef.train.algos.score_centering import ScoreCenteringSettings
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.slime_backend.score_centering import (
    ScoreCenteringAlgorithm,
    score_centering_sample_row,
    settings_from_args,
)


def log_probs_fixture(*probabilities: float) -> list[float]:
    return [math.log(value) for value in probabilities]


def row_fixture(
    *,
    loss_mask: list[int] | None = None,
    indices: list[list[int]] | None = None,
    log_probs: list[list[float]] | None = None,
) -> list:
    # Prompt [1, 2], response [5, 6]; the sampler recorded three ids per position.
    return [
        "record-1",
        [1, 2, 5, 6],
        loss_mask if loss_mask is not None else [1, 1],
        log_probs_fixture(0.5, 0.3),
        1.0,
        indices if indices is not None else [[5, 7, 8], [9, 6, 3]],
        log_probs if log_probs is not None else [log_probs_fixture(0.5, 0.2, 0.1), log_probs_fixture(0.4, 0.3, 0.2)],
    ]


def payload_fixture(*rows: list) -> dict:
    return {
        "samples": list(rows),
        "rollout_ids": list(range(len(rows))),
        "loss": "score_centering",
        "advantages": [1.0] * len(rows),
    }


def prepared_fixture(top_k: int, *rows: list) -> dict:
    # The bridge checks the rows in torch.
    pytest.importorskip("torch")
    data = to_slime_rollout_data(payload_fixture(*rows))
    ScoreCenteringAlgorithm(ScoreCenteringSettings(top_k=top_k)).prepare_rollout(data)
    return data


@pytest.mark.unit
def test_family_is_built_in_and_needs_advantages_and_rollout_log_probs() -> None:
    spec = resolve_loss_family("score_centering")
    assert isinstance(spec, ScoreCenteringAlgorithm)
    assert spec.loss_type == "custom_loss"
    assert spec.advantages == "required"
    assert spec.requires_rollout_logprobs
    assert set(spec.rollout_data_keys) == {"topk_indices", "topk_log_probs"}
    assert spec.rollout_tensor_dtypes == {"topk_indices": "long", "topk_log_probs": "float32"}

    payload = payload_fixture(row_fixture())
    del payload["advantages"]
    with pytest.raises(ValueError, match="requires one Reef advantage per sample"):
        to_slime_rollout_data(payload)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_k", 0),
        ("top_k", True),
        ("importance_weight", "ppo"),
        ("tis_cap", 0.0),
        ("tis_cap", math.inf),
        ("mis_lower", 6.0),
        ("mis_upper", math.nan),
        ("min_tail_mass", 0.0),
        ("min_tail_mass", 1.0),
    ],
)
def test_settings_reject_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="score centering"):
        ScoreCenteringSettings(**{field: value})


@pytest.mark.unit
def test_flags_travel_from_argv_onto_args_and_back() -> None:
    spec = resolve_loss_family("score_centering")
    settings, remaining = spec.parse_driver_options(
        [
            "--score-centering-top-k=32",
            "--score-centering-importance-weight=mis",
            "--score-centering-mis-lower=0.25",
            "--score-centering-mis-upper=4",
            "--score-centering-tis-cap=3",
            "--score-centering-min-tail-mass=1e-5",
            "--loss-type",
            "custom_loss",
        ]
    )
    assert settings == ScoreCenteringSettings(
        top_k=32, importance_weight="mis", tis_cap=3.0, mis_lower=0.25, mis_upper=4.0, min_tail_mass=1e-5
    )
    assert remaining == ["--loss-type", "custom_loss"]

    args = SimpleNamespace()
    spec.apply_driver_options(args, settings)
    assert args.loss_family == "score_centering"
    assert settings_from_args(args) == settings
    assert spec.bind(settings).settings == settings
    assert spec.bind().settings == ScoreCenteringSettings()
    with pytest.raises(TypeError, match="ScoreCenteringSettings"):
        spec.bind(object())
    with pytest.raises(ValueError, match="top_k"):
        spec.parse_driver_options(["--score-centering-top-k=0"])


@pytest.mark.unit
def test_backend_args_refuse_what_the_loss_would_silently_ignore() -> None:
    spec = resolve_loss_family("score_centering")
    accepted = {
        "loss_type": "custom_loss",
        "use_rollout_logprobs": True,
        "context_parallel_size": 1,
        "use_tis": False,
        "use_kl_loss": False,
        "entropy_coef": 0.0,
    }
    spec.validate_backend_args(SimpleNamespace(**accepted))

    with pytest.raises(RuntimeError, match="use-rollout-logprobs"):
        spec.validate_backend_args(SimpleNamespace(**{**accepted, "use_rollout_logprobs": False}))
    with pytest.raises(RuntimeError, match="context-parallel-size 1"):
        spec.validate_backend_args(SimpleNamespace(**{**accepted, "context_parallel_size": 2}))
    with pytest.raises(RuntimeError, match="--score-centering-importance-weight"):
        spec.validate_backend_args(SimpleNamespace(**{**accepted, "use_tis": True}))
    with pytest.raises(RuntimeError, match="use-kl-loss or --entropy-coef"):
        spec.validate_backend_args(SimpleNamespace(**{**accepted, "use_kl_loss": True}))
    with pytest.raises(RuntimeError, match="use-kl-loss or --entropy-coef"):
        spec.validate_backend_args(SimpleNamespace(**{**accepted, "entropy_coef": 0.01}))


@pytest.mark.unit
def test_sample_row_carries_the_sampler_head() -> None:
    sample = policy_trajectory("i1", [1, 2, 5], [1], [-0.5], 1.0, "v0").with_training(
        topk_indices=[[5, 7]], topk_log_probs=[[-0.5, -1.5]]
    )
    row = score_centering_sample_row(sample)
    assert row == ["i1", [1, 2, 5], [1], [-0.5], 1.0, [[5, 7]], [[-0.5, -1.5]]]


@pytest.mark.unit
def test_payload_keeps_the_first_top_k_entries_and_pads_untrained_positions() -> None:
    from reef.train.slime_backend.score_centering.heads import PADDING_LOG_PROB

    data = prepared_fixture(2, row_fixture(loss_mask=[0, 1]))
    assert [ids.tolist() for ids in data["topk_indices"]] == [[[0, 0], [9, 6]]]
    assert [values.tolist() for values in data["topk_log_probs"]] == [
        [[PADDING_LOG_PROB] * 2, pytest.approx(log_probs_fixture(0.4, 0.3))]
    ]
    assert data["advantages"] == [[1.0, 1.0]]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("row", "top_k", "message"),
    [
        (row_fixture(indices=[], log_probs=[]), 3, "capture_topk >= 3"),
        (
            row_fixture(indices=[[5, 7, 8]], log_probs=[log_probs_fixture(0.5, 0.2, 0.1)]),
            3,
            "1 rows for a 2-token response",
        ),
        (row_fixture(), 4, "fewer than top_k=4"),
        (row_fixture(indices=[[5, 7, 7], [9, 6, 3]]), 3, "distinct"),
        (row_fixture(indices=[[5, -1, 8], [9, 6, 3]]), 3, "position 0 topk_indices must be non-negative"),
        (row_fixture(indices=[[5, 7.5, 8], [9, 6, 3]]), 3, "non-negative integers"),
        (row_fixture(log_probs=[["x", -1.0, -2.0], log_probs_fixture(0.4, 0.3, 0.2)]), 3, "must hold numbers"),
        (
            row_fixture(log_probs=[[-0.1, -0.2, math.nan], log_probs_fixture(0.4, 0.3, 0.2)]),
            3,
            "finite log-probabilities",
        ),
        (
            row_fixture(log_probs=[log_probs_fixture(0.5, 0.4, 0.3), log_probs_fixture(0.4, 0.3, 0.2)]),
            3,
            "sum to more than one",
        ),
        # Position 1 (after an untrained position 0) repeats an id.
        (row_fixture(loss_mask=[0, 1], indices=[[], [9, 9, 3]], log_probs=[[], [-1, -2, -3]]), 3, "position 1"),
        # Position 1's head records the sampled token 6 at 0.2, not the 0.3 it was sampled at.
        (
            row_fixture(log_probs=[log_probs_fixture(0.5, 0.2, 0.1), log_probs_fixture(0.4, 0.2, 0.3)]),
            3,
            "not aligned",
        ),
    ],
)
def test_payload_refuses_incomplete_or_misaligned_top_k(row: list, top_k: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        prepared_fixture(top_k, row)


@pytest.mark.unit
def test_payload_refuses_a_policy_row_without_the_top_k_columns() -> None:
    with pytest.raises(ValueError, match="must be \\[source_id"):
        to_slime_rollout_data(payload_fixture(row_fixture()[:5]))


@pytest.mark.unit
def test_worker_resolves_the_loss_hook_and_the_adapters_partition_the_top_k() -> None:
    pytest.importorskip("torch")
    from reef.train.slime_backend.algorithm import resolve_objective_paths
    from reef.train.slime_backend.reef_adapters.slime_arguments import configure_reef_loss_args

    args = SimpleNamespace(loss_family="score_centering", custom_rollout_data_keys=None)
    resolve_objective_paths(args)
    assert args.custom_loss_function_path == (
        "reef.train.slime_backend.score_centering.objective.score_centering_loss"
    )

    configure_reef_loss_args(args)
    assert set(args.custom_rollout_data_keys) == {"topk_indices", "topk_log_probs"}
    assert args.reef_rollout_tensor_dtypes == {"topk_indices": "long", "topk_log_probs": "float32"}
    assert set(args.reef_external_batch_keys) == {"rollout_log_probs", "topk_indices", "topk_log_probs"}
