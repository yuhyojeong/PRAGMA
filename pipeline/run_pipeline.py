"""
Run the PRAGMA data construction pipeline in order.

Inputs:
    - Scripts in pipeline/
    - Optionally, metrics/eval_criteria.py when --with-metrics is set

Outputs:
    - data/privasis.json
    - data/query_data.json
    - data/evid_sessions.json
    - data/filler_sessions.json
    - data/full_sessions.json
    - data/metadata.json
    - data/response_metrics.json, when --with-metrics is set
"""
import argparse
import subprocess
import sys
from pathlib import Path


RECBENCH_DIR = Path(__file__).resolve().parent.parent

PIPELINE_STEPS = [
    ("sample_users", Path("pipeline/sample_users.py")),
    ("extract_schema", Path("pipeline/extract_schema.py")),
    ("create_events", Path("pipeline/create_events.py")),
    ("event_query", Path("pipeline/event_query.py")),
    ("create_traj", Path("pipeline/create_traj.py")),
    ("traj_query", Path("pipeline/traj_query.py")),
    ("evid_sessions", Path("pipeline/evid_sessions.py")),
    ("filler_sessions", Path("pipeline/filler_sessions.py")),
    ("full_sessions", Path("pipeline/full_sessions.py")),
]

METRIC_STEPS = [
    ("eval_criteria", Path("metrics/eval_criteria.py")),
]


def parse_step_list(raw_steps: str) -> set[str]:
    return {step.strip() for step in raw_steps.split(",") if step.strip()}


def select_steps(args: argparse.Namespace) -> list[tuple[str, Path]]:
    steps = list(PIPELINE_STEPS)
    if args.with_metrics:
        steps.extend(METRIC_STEPS)

    valid_names = {name for name, _ in steps}
    unknown = args.only | args.skip
    if args.start:
        unknown.add(args.start)
    if args.end:
        unknown.add(args.end)
    unknown -= valid_names
    if unknown:
        raise SystemExit(f"Unknown step(s): {', '.join(sorted(unknown))}")

    if args.only:
        return [(name, path) for name, path in steps if name in args.only]

    start_idx = 0
    end_idx = len(steps) - 1
    if args.start:
        start_idx = next(idx for idx, (name, _) in enumerate(steps) if name == args.start)
    if args.end:
        end_idx = next(idx for idx, (name, _) in enumerate(steps) if name == args.end)
    if start_idx > end_idx:
        raise SystemExit("--start must come before --end")

    selected = steps[start_idx : end_idx + 1]
    return [(name, path) for name, path in selected if name not in args.skip]


def run_step(name: str, script_path: Path, dry_run: bool) -> None:
    command = [sys.executable, str(script_path)]
    print(f"\n==> {name}: {' '.join(command)}", flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=RECBENCH_DIR, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PRAGMA data construction scripts sequentially.",
    )
    parser.add_argument(
        "--with-metrics",
        action="store_true",
        help="Also generate evaluation rubrics after data construction.",
    )
    parser.add_argument(
        "--start",
        choices=[name for name, _ in PIPELINE_STEPS + METRIC_STEPS],
        help="First step to run.",
    )
    parser.add_argument(
        "--end",
        choices=[name for name, _ in PIPELINE_STEPS + METRIC_STEPS],
        help="Last step to run.",
    )
    parser.add_argument(
        "--only",
        type=parse_step_list,
        default=set(),
        help="Comma-separated step names to run, preserving pipeline order.",
    )
    parser.add_argument(
        "--skip",
        type=parse_step_list,
        default=set(),
        help="Comma-separated step names to skip.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected commands without running them.",
    )
    args = parser.parse_args()

    for name, path in select_steps(args):
        run_step(name, path, args.dry_run)


if __name__ == "__main__":
    main()
