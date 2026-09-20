"""Fixed walking command battery, eval-env overrides, and headless rollout."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends

from mjlab_microduck.eval.walk_metrics import summarize_trial

DEFAULT_TASK = "Mjlab-Velocity-Flat-MicroDuck"
DEFAULT_HORIZON_S = 10.0
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
CONTROL_HZ_HINT = 50.0


@dataclass(frozen=True)
class WalkCase:
    case_id: str
    vx: float
    vy: float
    wz: float
    ood: bool = False


def default_battery(*, extra: bool = False) -> tuple[WalkCase, ...]:
    cases = (
        WalkCase("stand", 0.0, 0.0, 0.0),
        WalkCase("vx_0.2", 0.2, 0.0, 0.0),
        WalkCase("vx_0.4", 0.4, 0.0, 0.0),
        WalkCase("vx_0.3_vy_p0.1", 0.3, 0.1, 0.0),
        WalkCase("vx_0.3_vy_m0.1", 0.3, -0.1, 0.0),
        WalkCase("wz_p0.5", 0.0, 0.0, 0.5),
        WalkCase("wz_m0.5", 0.0, 0.0, -0.5),
    )
    if extra:
        cases = (*cases, WalkCase("vx_0.6", 0.6, 0.0, 0.0, ood=True))
    return cases


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[3],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def prepare_eval_env_cfg(env_cfg, *, num_envs: int, seed: int, horizon_s: float):
    """Mutate the cfg *before* env construction. Disable pushes; match trial length."""
    env_cfg.scene.num_envs = num_envs
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = seed
    env_cfg.episode_length_s = float(horizon_s) + 2.0
    events = getattr(env_cfg, "events", None)
    if events is not None and "push_robot" in events:
        del events["push_robot"]
    return env_cfg


def freeze_commands(env: ManagerBasedRlEnv, vx: float, vy: float, wz: float) -> None:
    """Hold twist constant and zero head/body pose. Call after reset, via managers."""
    cmd_mgr = env.command_manager
    huge = 1.0e9

    twist = cmd_mgr.get_term("twist")
    twist.time_left[:] = huge
    for flag in (
        "is_standing_env",
        "is_heading_env",
        "is_world_env",
        "is_forward_env",
    ):
        tensor = getattr(twist, flag, None)
        if tensor is not None:
            tensor[:] = False
    twist.vel_command_b[:, 0] = vx
    twist.vel_command_b[:, 1] = vy
    twist.vel_command_b[:, 2] = wz
    twist.vel_command_w[:] = twist.vel_command_b

    active = list(cmd_mgr.active_terms)
    for name in ("head_pose", "body_pose"):
        if name not in active:
            continue
        term = cmd_mgr.get_term(name)
        term.time_left[:] = huge
        term._command.zero_()


def resolve_checkpoint(
    *,
    agent_cfg,
    checkpoint: int | None,
    checkpoint_file: str | None,
    wandb_run_path: str | None,
) -> Path:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if checkpoint_file is not None:
        resume_path = Path(checkpoint_file)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        return resume_path

    if checkpoint is not None:
        checkpoint_filename = f"model_{checkpoint}.pt"
        if wandb_run_path is not None:
            import wandb

            api = wandb.Api()
            wandb_run = api.run(str(wandb_run_path))
            run_id = wandb_run_path.split("/")[-1]
            download_dir = log_root_path / "wandb_checkpoints" / run_id
            resume_path = download_dir / checkpoint_filename
            if resume_path.exists():
                return resume_path
            available = [f.name for f in wandb_run.files() if "model" in f.name]
            if checkpoint_filename not in available:
                raise FileNotFoundError(
                    f"Checkpoint '{checkpoint_filename}' not found in wandb run. "
                    f"Available: {sorted(available)}"
                )
            download_dir.mkdir(parents=True, exist_ok=True)
            wandb_run.file(checkpoint_filename).download(str(download_dir), replace=True)
            return resume_path
        resume_path = Path(
            get_checkpoint_path(log_root_path, checkpoint=checkpoint_filename)
        )
        return resume_path

    if wandb_run_path is None:
        raise ValueError(
            "Provide --checkpoint-file, or --checkpoint plus --wandb-run-path."
        )
    resume_path, _was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(wandb_run_path)
    )
    return Path(resume_path)


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": math.nan, "std": math.nan, "n": 0}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "n": 1}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values),
        "n": len(values),
    }


def aggregate_trials(trials: list[dict]) -> dict:
    by_case: dict[str, dict] = {}
    metric_keys = (
        "survival_s",
        "lin_vel_rmse",
        "ang_vel_rmse",
        "action_rate_rms",
        "mean_trunk_z",
        "tilt_rms",
        "xy_drift",
    )
    case_ids = []
    for row in trials:
        cid = row["case_id"]
        if cid not in by_case:
            case_ids.append(cid)
            by_case[cid] = {"trials": [], "ood": bool(row.get("ood", False))}
        by_case[cid]["trials"].append(row)

    out_cases = {}
    for cid in case_ids:
        rows = by_case[cid]["trials"]
        falls = [1.0 if r["fell"] else 0.0 for r in rows]
        stats = {k: _mean_std([float(r[k]) for r in rows]) for k in metric_keys}
        stats["fall_rate"] = _mean_std(falls)
        stats["ood"] = by_case[cid]["ood"]
        stats["n"] = len(rows)
        out_cases[cid] = stats

    in_dist = [r for r in trials if not r.get("ood")]
    overall = {k: _mean_std([float(r[k]) for r in in_dist]) for k in metric_keys}
    overall["fall_rate"] = _mean_std(
        [1.0 if r["fell"] else 0.0 for r in in_dist]
    )
    overall["n"] = len(in_dist)
    return {"by_case": out_cases, "overall_in_distribution": overall}


def write_outputs(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "eval.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    trials = payload.get("trials", [])
    if not trials:
        return
    csv_path = out_dir / "trials.csv"
    fieldnames = list(trials[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trials)


def _robot(env: ManagerBasedRlEnv):
    return env.scene["robot"]


@torch.inference_mode()
def _rollout_case(
    *,
    env: RslRlVecEnvWrapper,
    policy,
    case: WalkCase,
    horizon_s: float,
    seeds: tuple[int, ...],
    policy_meta: dict,
) -> list[dict]:
    raw = env.unwrapped
    freeze_commands(raw, case.vx, case.vy, case.wz)
    obs, _extras = env.reset()
    freeze_commands(raw, case.vx, case.vy, case.wz)
    obs = env.get_observations()

    dt = float(raw.step_dt)
    n_steps = max(1, int(round(horizon_s / dt)))
    n = raw.num_envs
    robot = _robot(raw)

    hist = {
        "trunk_z": [],
        "grav": [],
        "vel_xy": [],
        "ang_z": [],
        "actions": [],
        "pos_xy": [],
        "heading": [],
        "nan": [],
        "terminated": [],
    }

    cmd_xy = torch.tensor([case.vx, case.vy], device=raw.device, dtype=torch.float32)
    cmd_wz = torch.tensor(case.wz, device=raw.device, dtype=torch.float32)

    for _step in range(n_steps):
        freeze_commands(raw, case.vx, case.vy, case.wz)
        actions = policy(obs)
        data = robot.data
        hist["trunk_z"].append(data.root_link_pos_w[:, 2].clone())
        hist["grav"].append(data.projected_gravity_b.clone())
        hist["vel_xy"].append(data.root_link_lin_vel_b[:, :2].clone())
        hist["ang_z"].append(data.root_link_ang_vel_b[:, 2].clone())
        hist["actions"].append(actions.clone())
        hist["pos_xy"].append(data.root_link_pos_w[:, :2].clone())
        heading = getattr(data, "heading_w", None)
        if heading is None:
            quat = data.root_link_quat_w
            w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            heading = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        hist["heading"].append(heading.clone())
        nan_now = ~torch.isfinite(data.root_link_pos_w).all(dim=-1)
        hist["nan"].append(nan_now)

        obs, _rew, dones, extras = env.step(actions)
        dones_b = dones.bool() if dones.dtype != torch.bool else dones
        timeouts = extras.get("time_outs")
        if timeouts is None:
            non_timeout_term = dones_b
        else:
            non_timeout_term = dones_b & ~timeouts.bool()
        hist["terminated"].append((non_timeout_term & ~nan_now).clone())
        freeze_commands(raw, case.vx, case.vy, case.wz)

    stacked = {k: torch.stack(v, dim=0) for k, v in hist.items()}
    trials = []
    for env_i, seed in enumerate(seeds):
        sl = {k: v[:, env_i] for k, v in stacked.items()}
        tlen = sl["trunk_z"].shape[0]
        metrics = summarize_trial(
            dt=dt,
            horizon_s=horizon_s,
            trunk_z=sl["trunk_z"],
            projected_gravity_b=sl["grav"],
            vel_xy=sl["vel_xy"],
            ang_vel_z=sl["ang_z"],
            cmd_xy=cmd_xy.expand(tlen, -1),
            cmd_wz=cmd_wz.expand(tlen),
            actions=sl["actions"],
            pos_xy=sl["pos_xy"],
            heading_w=sl["heading"],
            nan_mask=sl["nan"],
            terminated_mask=sl["terminated"],
        )
        row = {
            **policy_meta,
            "seed": int(seed),
            "case_id": case.case_id,
            "cmd_vx": case.vx,
            "cmd_vy": case.vy,
            "cmd_wz": case.wz,
            "ood": case.ood,
            **metrics,
        }
        trials.append(row)
    return trials


def _cfg_asdict(agent_cfg):
    if hasattr(agent_cfg, "__dataclass_fields__"):
        return asdict(agent_cfg)
    if isinstance(agent_cfg, dict):
        return agent_cfg
    return dict(agent_cfg)


def run_walk_eval(
    *,
    task: str,
    label: str,
    out_dir: Path,
    wandb_run_path: str | None,
    checkpoint: int | None,
    checkpoint_file: str | None,
    device: str,
    horizon_s: float,
    seeds: tuple[int, ...],
    extra: bool,
) -> dict:
    configure_torch_backends()
    import mjlab_microduck.tasks  # noqa: F401 — register task family

    agent_cfg = load_rl_cfg(task)
    resume_path = resolve_checkpoint(
        agent_cfg=agent_cfg,
        checkpoint=checkpoint,
        checkpoint_file=checkpoint_file,
        wandb_run_path=wandb_run_path,
    )
    env_cfg = load_env_cfg(task, play=True)
    prepare_eval_env_cfg(
        env_cfg, num_envs=len(seeds), seed=seeds[0], horizon_s=horizon_s
    )

    torch.manual_seed(seeds[0])
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task) or OnPolicyRunner
    runner = runner_cls(wrapped, _cfg_asdict(agent_cfg), device=device)
    runner.load(str(resume_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

    protocol = {
        "task": task,
        "horizon_s": horizon_s,
        "seeds": list(seeds),
        "num_envs": len(seeds),
        "device": device,
        "pushes": "disabled",
        "commands": "frozen_twist_zero_pose",
        "domain_randomization": "reset_time_on_fixed_seeds",
        "control_hz_hint": CONTROL_HZ_HINT,
        "step_dt": float(env.step_dt),
        "extra_ood": extra,
        "git_sha": git_sha(),
    }
    policy_meta = {
        "policy_label": label,
        "wandb_run": wandb_run_path or "",
        "checkpoint": str(resume_path),
        "git_sha": protocol["git_sha"],
    }

    all_trials: list[dict] = []
    for case in default_battery(extra=extra):
        print(f"[eval_walk] case {case.case_id}  cmd=({case.vx}, {case.vy}, {case.wz})")
        all_trials.extend(
            _rollout_case(
                env=wrapped,
                policy=policy,
                case=case,
                horizon_s=horizon_s,
                seeds=seeds,
                policy_meta=policy_meta,
            )
        )

    payload = {
        "protocol": protocol,
        "policy": policy_meta,
        "trials": all_trials,
        "aggregates": aggregate_trials(all_trials),
    }
    write_outputs(out_dir, payload)
    env.close()
    return payload
