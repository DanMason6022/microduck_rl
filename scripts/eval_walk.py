"""Headless walking A/B evaluator (Phase 1E).

Scoreboard is behavior (falls, velocity RMSE, action-rate RMS), not W&B
Episode_Reward. Do not use scripts/eval_and_promote.py.

Examples:

  uv run python scripts/eval_walk.py --dry-run

  uv run python scripts/eval_walk.py \\
    --task Mjlab-Velocity-Flat-MicroDuck \\
    --wandb-run-path danielmason2206-personal/mjlab_microduck/mdnezvfu \\
    --checkpoint 3000 --label A --out results/eval_A

  uv run python scripts/eval_walk.py \\
    --task Mjlab-Velocity-Flat-MicroDuck \\
    --wandb-run-path danielmason2206-personal/mjlab_microduck/<POLICY_B> \\
    --checkpoint 3000 --label B --out results/eval_B

  uv run python scripts/eval_walk.py --compare results/eval_A results/eval_B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mjlab_microduck.eval.compare import compare_paths
from mjlab_microduck.eval.walk_battery import (
    DEFAULT_HORIZON_S,
    DEFAULT_SEEDS,
    DEFAULT_TASK,
    default_battery,
    run_walk_eval,
)


def _parse_seeds(text: str) -> tuple[int, ...]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("need at least one seed")
    return tuple(int(p) for p in parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Headless walking battery for Microduck A/B eval.",
        epilog=(
            "Owner-run: evaluate A then B with identical --seeds and --horizon-s, "
            "then --compare. Never mix horizons. King-of-the-hill promotion is out "
            "of scope until this CSV exists."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("A_DIR", "B_DIR"),
        help="Compare two eval.json directories and print the 1C decision table.",
    )
    parser.add_argument(
        "--compare-out",
        type=Path,
        default=None,
        help="Directory for compare.json/compare.txt (default: <parent-of-A>/eval_AB).",
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--label", default="policy")
    parser.add_argument("--out", type=Path, default=Path("results/eval"))
    parser.add_argument("--wandb-run-path", default=None)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--checkpoint-file", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizon-s", type=float, default=DEFAULT_HORIZON_S)
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=DEFAULT_SEEDS,
        help="Comma-separated trial seeds (default: 0,1,2,3,4).",
    )
    parser.add_argument(
        "--extra",
        action="store_true",
        help="Append OOD vx=0.6 (not used in the 1C win/loss call).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command battery and exit without building MuJoCo.",
    )
    args = parser.parse_args(argv)

    if args.compare:
        path_a = Path(args.compare[0])
        path_b = Path(args.compare[1])
        out = args.compare_out
        if out is None:
            parent = path_a.parent if path_a.is_dir() else path_a.parent.parent
            out = parent / "eval_AB"
        verdict, table = compare_paths(path_a, path_b, out_dir=out)
        sys.stdout.write(table)
        print(json.dumps(verdict, indent=2))
        print(f"Wrote {out / 'compare.json'} and {out / 'compare.txt'}")
        return 0

    if args.dry_run:
        battery = default_battery(extra=args.extra)
        payload = {
            "task": args.task,
            "horizon_s": args.horizon_s,
            "seeds": list(args.seeds),
            "device": args.device,
            "extra": args.extra,
            "cases": [c.__dict__ for c in battery],
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.checkpoint_file is None and args.checkpoint is None and args.wandb_run_path is None:
        parser.error(
            "need --checkpoint-file, or --wandb-run-path (optionally with --checkpoint)"
        )

    run_walk_eval(
        task=args.task,
        label=args.label,
        out_dir=args.out,
        wandb_run_path=args.wandb_run_path,
        checkpoint=args.checkpoint,
        checkpoint_file=args.checkpoint_file,
        device=args.device,
        horizon_s=args.horizon_s,
        seeds=args.seeds,
        extra=args.extra,
    )
    print(f"Wrote {args.out / 'eval.json'} and {args.out / 'trials.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
