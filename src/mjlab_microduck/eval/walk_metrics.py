"""Pure walking metrics. No environment, no rewards — behavior only."""

from __future__ import annotations

import math
from typing import Literal

import torch

FALL_Z_M = 0.08
# Tilt from vertical at which hypot(gx, gy) == |gz| for a unit gravity vector.
FALL_TILT_RAD = math.pi / 4.0

Termination = Literal["fall", "nan", "timeout"]


def tilt_from_vertical_rad(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    """Angle from upright, radians.

    ``projected_gravity_b`` is (..., 3). Upright is roughly (0, 0, -1).
    """
    gx = projected_gravity_b[..., 0]
    gy = projected_gravity_b[..., 1]
    gz = projected_gravity_b[..., 2]
    return torch.atan2(torch.hypot(gx, gy), gz.abs().clamp(min=1e-8))


def is_fallen(
    trunk_z: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    z_min: float = FALL_Z_M,
    tilt_max_rad: float = FALL_TILT_RAD,
) -> torch.Tensor:
    """True where trunk is too low or tilt exceeds ``tilt_max_rad``."""
    tilt = tilt_from_vertical_rad(projected_gravity_b)
    return (trunk_z < z_min) | (tilt > tilt_max_rad)


def first_true_index(mask: torch.Tensor) -> int | None:
    """Index of the first True along a 1-D bool mask, or None."""
    if mask.ndim != 1:
        raise ValueError(f"expected 1-D mask, got shape {tuple(mask.shape)}")
    hits = mask.nonzero(as_tuple=False).flatten()
    if hits.numel() == 0:
        return None
    return int(hits[0].item())


def rmse(error: torch.Tensor) -> torch.Tensor:
    """Root-mean-square of ``error`` over all dimensions except batch.

    ``error`` is (T, ...) or (T,). Empty input returns 0.
    """
    if error.numel() == 0:
        return error.new_zeros(())
    return torch.sqrt(torch.mean(error.float().square()))


def lin_vel_rmse(vel_xy: torch.Tensor, cmd_xy: torch.Tensor) -> torch.Tensor:
    """RMSE of body-frame horizontal velocity vs command. Shapes (T, 2).

    Per-step error is the L2 vector residual; RMSE is over time.
    """
    err = torch.linalg.vector_norm((vel_xy - cmd_xy).float(), dim=-1)
    return rmse(err)


def ang_vel_rmse(ang_vel_z: torch.Tensor, cmd_wz: torch.Tensor) -> torch.Tensor:
    """RMSE of body-frame yaw rate vs command. Shapes (T,) or (T, 1)."""
    return rmse(ang_vel_z.reshape(-1) - cmd_wz.reshape(-1))


def action_rate_rms(actions: torch.Tensor) -> torch.Tensor:
    """RMS of per-step L2 action change. ``actions`` is (T, A).

    T < 2 returns 0 (no rate).
    """
    if actions.shape[0] < 2:
        return actions.new_zeros(())
    delta = actions[1:] - actions[:-1]
    per_step = torch.linalg.vector_norm(delta.float(), dim=-1)
    return torch.sqrt(torch.mean(per_step.square()))


def xy_drift(actual_xy: torch.Tensor, expected_xy: torch.Tensor) -> torch.Tensor:
    """||actual - expected|| at a single (2,) pose, or last row of (T, 2)."""
    a = actual_xy[-1] if actual_xy.ndim == 2 else actual_xy
    e = expected_xy[-1] if expected_xy.ndim == 2 else expected_xy
    return torch.linalg.vector_norm((a - e).float())


def body_cmd_to_world_xy(cmd_xy: torch.Tensor, heading_w: torch.Tensor) -> torch.Tensor:
    """Rotate body-frame (vx, vy) into world XY using yaw ``heading_w``."""
    c = torch.cos(heading_w)
    s = torch.sin(heading_w)
    vx = cmd_xy[..., 0]
    vy = cmd_xy[..., 1]
    world_x = c * vx - s * vy
    world_y = s * vx + c * vy
    return torch.stack((world_x, world_y), dim=-1)


def summarize_trial(
    *,
    dt: float,
    horizon_s: float,
    trunk_z: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    vel_xy: torch.Tensor,
    ang_vel_z: torch.Tensor,
    cmd_xy: torch.Tensor,
    cmd_wz: torch.Tensor,
    actions: torch.Tensor,
    pos_xy: torch.Tensor,
    heading_w: torch.Tensor,
    nan_mask: torch.Tensor | None = None,
    terminated_mask: torch.Tensor | None = None,
) -> dict:
    """Compute one-trial scalars from time series (all length T unless noted).

    Metrics use samples up to and including the first fall/NaN (exclusive of
    the failing sample for rates that need a delta). Survival is that time,
    or ``horizon_s`` if neither occurs.
    """
    t = trunk_z.shape[0]
    fallen = is_fallen(trunk_z, projected_gravity_b)
    if nan_mask is None:
        nan_mask = ~torch.isfinite(trunk_z)
        nan_mask = nan_mask | ~torch.isfinite(projected_gravity_b).all(dim=-1)
        nan_mask = nan_mask | ~torch.isfinite(vel_xy).all(dim=-1)

    if terminated_mask is None:
        terminated_mask = torch.zeros_like(fallen)
    else:
        terminated_mask = terminated_mask.to(dtype=torch.bool)

    fall_i = first_true_index(fallen | terminated_mask)
    nan_i = first_true_index(nan_mask.to(dtype=torch.bool))

    termination: Termination = "timeout"
    end_i = t
    if nan_i is not None and (fall_i is None or nan_i <= fall_i):
        termination = "nan"
        end_i = nan_i
    elif fall_i is not None:
        termination = "fall"
        end_i = fall_i

    n = max(end_i, 1)
    sl = slice(0, n)
    if termination == "timeout":
        survival_s = float(min(t * dt, horizon_s))
    else:
        survival_s = float(end_i * dt)

    cmd_xy_t = cmd_xy[sl] if cmd_xy.ndim == 2 else cmd_xy.expand(n, -1)
    cmd_wz_t = cmd_wz[sl] if cmd_wz.ndim == 1 else cmd_wz.reshape(-1)[sl]

    world_cmd = body_cmd_to_world_xy(cmd_xy_t, heading_w[sl])
    expected_delta = world_cmd.sum(dim=0) * dt
    actual_delta = pos_xy[sl][-1] - pos_xy[0]

    tilt = tilt_from_vertical_rad(projected_gravity_b[sl])
    return {
        "horizon_s": float(horizon_s),
        "survival_s": float(survival_s),
        "fell": termination == "fall",
        "termination": termination,
        "lin_vel_rmse": float(lin_vel_rmse(vel_xy[sl], cmd_xy_t).item()),
        "ang_vel_rmse": float(ang_vel_rmse(ang_vel_z[sl], cmd_wz_t).item()),
        "action_rate_rms": float(action_rate_rms(actions[sl]).item()),
        "mean_trunk_z": float(trunk_z[sl].float().mean().item()),
        "tilt_rms": float(torch.sqrt(torch.mean(tilt.float().square())).item()),
        "xy_drift": float(xy_drift(actual_delta, expected_delta).item()),
    }
