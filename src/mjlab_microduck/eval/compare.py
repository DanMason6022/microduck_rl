"""Compare two eval.json walking batteries. No promotion."""

from __future__ import annotations

import json
from pathlib import Path

SMOOTHNESS_CASES = ("stand", "vx_0.2")
FALL_RATE_TOL = 0.05
LIN_RMSE_REL_TOL = 0.10
LIN_RMSE_ABS_EPS = 1e-4


def load_eval(path: Path) -> dict:
    json_path = path / "eval.json" if path.is_dir() else path
    if not json_path.exists():
        raise FileNotFoundError(f"No eval.json at {json_path}")
    return json.loads(json_path.read_text())


def _case_mean(payload: dict, case_id: str, metric: str) -> float | None:
    cases = payload.get("aggregates", {}).get("by_case", {})
    row = cases.get(case_id)
    if row is None:
        return None
    if metric == "fall_rate":
        return float(row["fall_rate"]["mean"])
    return float(row[metric]["mean"])


def _overall(payload: dict, metric: str) -> float:
    overall = payload["aggregates"]["overall_in_distribution"]
    if metric == "fall_rate":
        return float(overall["fall_rate"]["mean"])
    return float(overall[metric]["mean"])


def decide(a: dict, b: dict) -> dict:
    """Apply the Phase 1C decision rule. Does not copy checkpoints."""
    smoothness_drops = []
    for cid in SMOOTHNESS_CASES:
        va = _case_mean(a, cid, "action_rate_rms")
        vb = _case_mean(b, cid, "action_rate_rms")
        if va is None or vb is None:
            smoothness_drops.append((cid, None))
            continue
        smoothness_drops.append((cid, vb < va))

    known = [x for x in smoothness_drops if x[1] is not None]
    smoothness_win = bool(known) and all(x[1] for x in known)

    fall_a = _overall(a, "fall_rate")
    fall_b = _overall(b, "fall_rate")
    fall_ok = (fall_b - fall_a) <= FALL_RATE_TOL

    rmse_a = _overall(a, "lin_vel_rmse")
    rmse_b = _overall(b, "lin_vel_rmse")
    if rmse_a <= LIN_RMSE_ABS_EPS:
        tracking_ok = rmse_b <= LIN_RMSE_ABS_EPS
    else:
        tracking_ok = rmse_b <= rmse_a * (1.0 + LIN_RMSE_REL_TOL)

    acceptable = fall_ok and tracking_ok
    if smoothness_win and acceptable:
        decision = "B_wins_smoothness"
        champion = "B"
    elif not acceptable:
        decision = "A_remains_champion"
        champion = "A"
    else:
        decision = "inconclusive"
        champion = "A"

    return {
        "decision": decision,
        "champion": champion,
        "smoothness_win": smoothness_win,
        "smoothness_by_case": {cid: dropped for cid, dropped in smoothness_drops},
        "fall_ok": fall_ok,
        "tracking_ok": tracking_ok,
        "fall_rate_A": fall_a,
        "fall_rate_B": fall_b,
        "lin_vel_rmse_A": rmse_a,
        "lin_vel_rmse_B": rmse_b,
        "thresholds": {
            "smoothness_cases": list(SMOOTHNESS_CASES),
            "fall_rate_tol_pp": FALL_RATE_TOL,
            "lin_rmse_rel_tol": LIN_RMSE_REL_TOL,
        },
        "note": (
            "Yaw RMSE is secondary. A W&B track_angular_velocity bump without "
            "ang_vel_rmse improvement is not a win. No auto-promotion."
        ),
    }


def format_table(a: dict, b: dict, verdict: dict) -> str:
    cases = sorted(
        set(a["aggregates"]["by_case"]) | set(b["aggregates"]["by_case"]),
        key=lambda c: (
            a["aggregates"]["by_case"].get(c, {}).get("ood", False),
            c,
        ),
    )
    metrics = (
        "fall_rate",
        "lin_vel_rmse",
        "ang_vel_rmse",
        "action_rate_rms",
        "survival_s",
        "tilt_rms",
        "xy_drift",
    )
    lines = []
    la = a.get("policy", {}).get("policy_label", "A")
    lb = b.get("policy", {}).get("policy_label", "B")
    lines.append(f"Walking A/B  {la} vs {lb}")
    header = f"{'case':<22} {'metric':<16} {la:>12} {lb:>12} {'delta':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for cid in cases:
        tag = " ood" if a["aggregates"]["by_case"].get(cid, {}).get("ood") or b[
            "aggregates"
        ]["by_case"].get(cid, {}).get("ood") else ""
        for metric in metrics:
            va = _case_mean(a, cid, metric)
            vb = _case_mean(b, cid, metric)
            if va is None or vb is None:
                continue
            delta = vb - va
            lines.append(
                f"{cid + tag:<22} {metric:<16} {va:12.4f} {vb:12.4f} {delta:+12.4f}"
            )
        lines.append("")
    lines.append(
        f"decision: {verdict['decision']}  champion: {verdict['champion']}"
    )
    lines.append(
        f"smoothness_win={verdict['smoothness_win']}  "
        f"fall_ok={verdict['fall_ok']}  tracking_ok={verdict['tracking_ok']}"
    )
    lines.append(verdict["note"])
    return "\n".join(lines) + "\n"


def reading_notes(a: dict, b: dict) -> dict:
    """Human/agent footnotes. Survival-only reads miss freeze-vs-walk."""

    def m(payload: dict, case: str, metric: str) -> float | None:
        return _case_mean(payload, case, metric)

    notes = []
    vx = m(b, "vx_0.2", "lin_vel_rmse")
    drift = m(b, "vx_0.2", "xy_drift")
    if vx is not None and vx >= 0.15:
        notes.append(
            "B vx_0.2 lin_vel_rmse is close to the 0.2 m/s command; "
            "that is 'barely walking / standing', not a calmer gait."
        )
    if drift is not None and drift >= 1.5:
        notes.append(
            "B vx_0.2 xy_drift ≈ 2 m over 10 s is the missing integrated walk "
            "(commanded 0.2 m/s × 10 s). Action-rate drop there is freeze."
        )
    wz = m(b, "wz_p0.5", "ang_vel_rmse")
    if wz is not None and wz >= 0.4:
        notes.append(
            "B turn-in-place ang_vel_rmse ≈ |cmd|; it is not spinning."
        )
    notes.append(
        "Zero falls is expected on this battery: 10 s, flat, no pushes, "
        "in-distribution commands. Survival is a floor, not the 1C score."
    )
    notes.append(
        "Do not trust overall lin_vel_rmse alone: stand/turn cases with "
        "cmd_xy=0 pull the mean down if B refuses to walk."
    )
    return {
        "notes": notes,
        "vx_0.2_lin_vel_rmse": {"A": m(a, "vx_0.2", "lin_vel_rmse"), "B": vx},
        "vx_0.2_xy_drift": {"A": m(a, "vx_0.2", "xy_drift"), "B": drift},
        "vx_0.2_action_rate_rms": {
            "A": m(a, "vx_0.2", "action_rate_rms"),
            "B": m(b, "vx_0.2", "action_rate_rms"),
        },
    }


def write_compare(
    out_dir: Path,
    *,
    path_a: Path,
    path_b: Path,
    a: dict,
    b: dict,
    verdict: dict,
    table: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "eval_A": str(path_a),
        "eval_B": str(path_b),
        "policy_A": a.get("policy"),
        "policy_B": b.get("policy"),
        "protocol_A": a.get("protocol"),
        "protocol_B": b.get("protocol"),
        "verdict": verdict,
        "reading": reading_notes(a, b),
        "aggregates": {"A": a.get("aggregates"), "B": b.get("aggregates")},
    }
    (out_dir / "compare.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out_dir / "compare.txt").write_text(table)


def compare_paths(
    path_a: Path, path_b: Path, *, out_dir: Path | None = None
) -> tuple[dict, str]:
    a = load_eval(path_a)
    b = load_eval(path_b)
    verdict = decide(a, b)
    table = format_table(a, b, verdict)
    if out_dir is not None:
        write_compare(
            out_dir, path_a=path_a, path_b=path_b, a=a, b=b, verdict=verdict, table=table
        )
    return verdict, table
