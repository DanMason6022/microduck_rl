"""CPU-only tests for the walking eval battery (no live MuJoCo / GPU)."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from mjlab_microduck.eval.compare import compare_paths, decide, format_table
from mjlab_microduck.eval.walk_battery import (
    WalkCase,
    aggregate_trials,
    default_battery,
    prepare_eval_env_cfg,
    write_outputs,
)
from mjlab_microduck.eval.walk_metrics import (
    FALL_TILT_RAD,
    action_rate_rms,
    ang_vel_rmse,
    first_true_index,
    is_fallen,
    lin_vel_rmse,
    summarize_trial,
    tilt_from_vertical_rad,
)


def test_default_battery_in_distribution_and_ood_tag():
    core = default_battery(extra=False)
    ids = [c.case_id for c in core]
    assert ids == [
        "stand",
        "vx_0.2",
        "vx_0.4",
        "vx_0.3_vy_p0.1",
        "vx_0.3_vy_m0.1",
        "wz_p0.5",
        "wz_m0.5",
    ]
    assert all(not c.ood for c in core)
    extra = default_battery(extra=True)
    assert extra[-1] == WalkCase("vx_0.6", 0.6, 0.0, 0.0, ood=True)
    assert extra[-1].ood


def test_lin_vel_rmse_zero_on_perfect_track():
    vel = torch.tensor([[0.2, 0.0], [0.2, 0.0], [0.2, 0.0]])
    cmd = torch.tensor([[0.2, 0.0], [0.2, 0.0], [0.2, 0.0]])
    assert float(lin_vel_rmse(vel, cmd).item()) == 0.0


def test_lin_vel_rmse_known_error():
    vel = torch.tensor([[0.3, 0.0], [0.3, 0.0]])
    cmd = torch.tensor([[0.2, 0.0], [0.2, 0.0]])
    assert abs(float(lin_vel_rmse(vel, cmd).item()) - 0.1) < 1e-6


def test_ang_vel_rmse():
    wz = torch.tensor([0.5, 0.7])
    cmd = torch.tensor([0.5, 0.5])
    expected = math.sqrt(((0.0**2) + (0.2**2)) / 2)
    assert abs(float(ang_vel_rmse(wz, cmd).item()) - expected) < 1e-6


def test_action_rate_rms_constant_actions_is_zero():
    a = torch.ones(8, 14)
    assert float(action_rate_rms(a).item()) == 0.0


def test_action_rate_rms_single_step_is_zero():
    a = torch.randn(1, 14)
    assert float(action_rate_rms(a).item()) == 0.0


def test_action_rate_rms_unit_delta():
    a = torch.zeros(3, 2)
    a[1] = torch.tensor([3.0, 4.0])  # ||delta|| = 5
    a[2] = torch.tensor([3.0, 4.0])  # ||delta|| = 0
    # RMS of {5, 0} = sqrt(25/2)
    assert abs(float(action_rate_rms(a).item()) - math.sqrt(12.5)) < 1e-5


def test_tilt_upright_is_zero():
    g = torch.tensor([[0.0, 0.0, -1.0]])
    tilt = tilt_from_vertical_rad(g)
    assert float(tilt.item()) < 1e-6


def test_fall_at_45_deg_and_low_z():
    g_ok = torch.tensor([[0.0, 0.0, -1.0]])
    z_ok = torch.tensor([0.12])
    assert not bool(is_fallen(z_ok, g_ok).item())

    s = math.sin(FALL_TILT_RAD + 1e-3)
    c = math.cos(FALL_TILT_RAD + 1e-3)
    g_tilt = torch.tensor([[s, 0.0, -c]])
    assert bool(is_fallen(z_ok, g_tilt).item())

    z_low = torch.tensor([0.07])
    assert bool(is_fallen(z_low, g_ok).item())


def test_first_true_index_latches():
    m = torch.tensor([False, False, True, True])
    assert first_true_index(m) == 2
    assert first_true_index(torch.tensor([False, False])) is None


def test_summarize_trial_fall_truncates_metrics():
    t = 10
    dt = 0.02
    z = torch.full((t,), 0.12)
    z[4:] = 0.05
    g = torch.tensor([[0.0, 0.0, -1.0]]).expand(t, 3).clone()
    vel = torch.zeros(t, 2)
    wz = torch.zeros(t)
    cmd_xy = torch.zeros(t, 2)
    cmd_wz = torch.zeros(t)
    actions = torch.zeros(t, 4)
    pos = torch.zeros(t, 2)
    heading = torch.zeros(t)
    out = summarize_trial(
        dt=dt,
        horizon_s=1.0,
        trunk_z=z,
        projected_gravity_b=g,
        vel_xy=vel,
        ang_vel_z=wz,
        cmd_xy=cmd_xy,
        cmd_wz=cmd_wz,
        actions=actions,
        pos_xy=pos,
        heading_w=heading,
    )
    assert out["fell"] is True
    assert out["termination"] == "fall"
    assert abs(out["survival_s"] - 4 * dt) < 1e-9


def test_summarize_trial_env_termination_counts_as_fall():
    t = 6
    dt = 0.02
    z = torch.full((t,), 0.12)
    g = torch.tensor([[0.0, 0.0, -1.0]]).expand(t, 3).clone()
    term = torch.tensor([False, False, True, False, False, False])
    out = summarize_trial(
        dt=dt,
        horizon_s=1.0,
        trunk_z=z,
        projected_gravity_b=g,
        vel_xy=torch.zeros(t, 2),
        ang_vel_z=torch.zeros(t),
        cmd_xy=torch.zeros(t, 2),
        cmd_wz=torch.zeros(t),
        actions=torch.zeros(t, 4),
        pos_xy=torch.zeros(t, 2),
        heading_w=torch.zeros(t),
        terminated_mask=term,
    )
    assert out["termination"] == "fall"
    assert abs(out["survival_s"] - 2 * dt) < 1e-9


def test_prepare_eval_env_cfg_drops_pushes():
    events = {"push_robot": object(), "reset_base": object()}
    cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=1),
        seed=0,
        episode_length_s=20.0,
        events=events,
    )
    prepare_eval_env_cfg(cfg, num_envs=5, seed=7, horizon_s=10.0)
    assert cfg.scene.num_envs == 5
    assert cfg.seed == 7
    assert cfg.episode_length_s == 12.0
    assert "push_robot" not in cfg.events
    assert "reset_base" in cfg.events


def _fake_eval(label, *, action_rate, fall_rate, lin_rmse, extra_case=None):
    def case(cid, ood=False):
        return {
            "action_rate_rms": {"mean": action_rate, "std": 0.0, "n": 5},
            "lin_vel_rmse": {"mean": lin_rmse, "std": 0.0, "n": 5},
            "ang_vel_rmse": {"mean": 0.1, "std": 0.0, "n": 5},
            "survival_s": {"mean": 10.0, "std": 0.0, "n": 5},
            "tilt_rms": {"mean": 0.1, "std": 0.0, "n": 5},
            "xy_drift": {"mean": 0.01, "std": 0.0, "n": 5},
            "mean_trunk_z": {"mean": 0.12, "std": 0.0, "n": 5},
            "fall_rate": {"mean": fall_rate, "std": 0.0, "n": 5},
            "ood": ood,
            "n": 5,
        }

    by_case = {
        "stand": case("stand"),
        "vx_0.2": case("vx_0.2"),
        "vx_0.4": case("vx_0.4"),
    }
    if extra_case:
        by_case["vx_0.6"] = case("vx_0.6", ood=True)
        by_case["vx_0.6"]["action_rate_rms"]["mean"] = extra_case
    trials_n = 15
    payload = {
        "policy": {"policy_label": label},
        "aggregates": {
            "by_case": by_case,
            "overall_in_distribution": {
                "action_rate_rms": {"mean": action_rate},
                "lin_vel_rmse": {"mean": lin_rmse},
                "ang_vel_rmse": {"mean": 0.1},
                "survival_s": {"mean": 10.0},
                "tilt_rms": {"mean": 0.1},
                "xy_drift": {"mean": 0.01},
                "mean_trunk_z": {"mean": 0.12},
                "fall_rate": {"mean": fall_rate},
                "n": trials_n,
            },
        },
    }
    return payload


def test_compare_b_wins_smoothness_when_safer():
    a = _fake_eval("A", action_rate=1.0, fall_rate=0.0, lin_rmse=0.05)
    b = _fake_eval("B", action_rate=0.8, fall_rate=0.0, lin_rmse=0.05)
    v = decide(a, b)
    assert v["decision"] == "B_wins_smoothness"
    assert v["champion"] == "B"
    table = format_table(a, b, v)
    assert "B_wins_smoothness" in table


def test_compare_ood_only_smoothness_does_not_win():
    a = _fake_eval("A", action_rate=1.0, fall_rate=0.0, lin_rmse=0.05, extra_case=0.1)
    b = _fake_eval("B", action_rate=1.0, fall_rate=0.0, lin_rmse=0.05, extra_case=0.01)
    v = decide(a, b)
    assert v["smoothness_win"] is False
    assert v["decision"] == "inconclusive"


def test_compare_more_falls_keeps_a():
    a = _fake_eval("A", action_rate=1.0, fall_rate=0.0, lin_rmse=0.05)
    b = _fake_eval("B", action_rate=0.5, fall_rate=0.2, lin_rmse=0.05)
    v = decide(a, b)
    assert v["decision"] == "A_remains_champion"
    assert v["fall_ok"] is False


def test_compare_paths_roundtrip(tmp_path: Path):
    a = _fake_eval("A", action_rate=1.0, fall_rate=0.0, lin_rmse=0.05)
    b = _fake_eval("B", action_rate=0.7, fall_rate=0.0, lin_rmse=0.05)
    da = tmp_path / "A"
    db = tmp_path / "B"
    da.mkdir()
    db.mkdir()
    (da / "eval.json").write_text(json.dumps(a))
    (db / "eval.json").write_text(json.dumps(b))
    verdict, table = compare_paths(da, db, out_dir=tmp_path / "AB")
    assert verdict["champion"] == "B"
    assert "action_rate_rms" in table
    assert (tmp_path / "AB" / "compare.json").exists()
    assert (tmp_path / "AB" / "compare.txt").exists()


def test_aggregate_and_write_outputs(tmp_path: Path):
    trials = [
        {
            "policy_label": "A",
            "case_id": "stand",
            "ood": False,
            "fell": False,
            "survival_s": 10.0,
            "lin_vel_rmse": 0.1,
            "ang_vel_rmse": 0.0,
            "action_rate_rms": 1.0,
            "mean_trunk_z": 0.12,
            "tilt_rms": 0.05,
            "xy_drift": 0.01,
        },
        {
            "policy_label": "A",
            "case_id": "stand",
            "ood": False,
            "fell": True,
            "survival_s": 4.0,
            "lin_vel_rmse": 0.2,
            "ang_vel_rmse": 0.0,
            "action_rate_rms": 1.2,
            "mean_trunk_z": 0.10,
            "tilt_rms": 0.2,
            "xy_drift": 0.02,
        },
        {
            "policy_label": "A",
            "case_id": "vx_0.6",
            "ood": True,
            "fell": False,
            "survival_s": 10.0,
            "lin_vel_rmse": 0.5,
            "ang_vel_rmse": 0.0,
            "action_rate_rms": 2.0,
            "mean_trunk_z": 0.12,
            "tilt_rms": 0.05,
            "xy_drift": 0.3,
        },
    ]
    agg = aggregate_trials(trials)
    assert agg["by_case"]["vx_0.6"]["ood"] is True
    assert agg["overall_in_distribution"]["n"] == 2
    assert abs(agg["overall_in_distribution"]["fall_rate"]["mean"] - 0.5) < 1e-9
    payload = {"trials": trials, "aggregates": agg}
    write_outputs(tmp_path, payload)
    assert (tmp_path / "eval.json").exists()
    assert (tmp_path / "trials.csv").exists()
    loaded = json.loads((tmp_path / "eval.json").read_text())
    assert loaded["aggregates"]["overall_in_distribution"]["n"] == 2
