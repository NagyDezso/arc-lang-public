"""Score saved attempt JSON against ARC ground-truth solutions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from src.models import GRID


class ChallengeSolution(BaseModel):
    attempt_1: GRID
    attempt_2: GRID


def evaluate_solutions(
    attempts_solutions_path: Path, truth_solutions_path: Path
) -> None:
    truth: dict[str, list[GRID]] = json.loads(
        truth_solutions_path.read_text(encoding="utf-8")
    )
    attempts: dict[str, list[ChallengeSolution]] = TypeAdapter(
        dict[str, list[ChallengeSolution]]
    ).validate_json(attempts_solutions_path.read_text(encoding="utf-8"))
    total_count = 0
    correct_count = 0
    for challenge_id, attempt_list in attempts.items():
        truth_grids: list[GRID] = truth[challenge_id]
        for i, truth_grid in enumerate(truth_grids):
            total_count = total_count + 1
            attempt_grids = attempt_list[i]
            if attempt_grids.attempt_1 == truth_grid:
                correct_count = correct_count + 1
            elif attempt_grids.attempt_2 == truth_grid:
                correct_count = correct_count + 1

    incorrect_count = total_count - correct_count
    accuracy = (correct_count / total_count * 100) if total_count else 0.0

    print("\n=== Final Evaluation Results ===")
    print(f"Total test cases : {total_count}")
    print(f"Correct          : {correct_count}")
    print(f"Incorrect        : {incorrect_count}")
    print(f"Accuracy         : {accuracy:.2f}%")


def _project_root_from_results_dir(results_dir: Path) -> Path:
    """results/<run_id> -> repo root."""
    resolved = results_dir.resolve()
    if resolved.name == "results" and resolved.parent.is_dir():
        return resolved.parent
    return resolved.parent.parent


def _default_truth_path(results_dir: Path) -> Path:
    root = _project_root_from_results_dir(results_dir)
    return root / "data" / "arc-prize-2025" / "arc-agi_evaluation_solutions.json"


def resolve_aggregate_attempts(results_dir: Path, attempts: Path | None) -> Path:
    if attempts is not None:
        p = attempts.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Attempts file not found: {p}")
        return p
    attempts_dir = results_dir.resolve() / "attempts"
    if not attempts_dir.is_dir():
        raise FileNotFoundError(f"No attempts directory: {attempts_dir}")
    candidates = sorted(attempts_dir.glob("*_attempts.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No *_attempts.json under {attempts_dir}; pass --attempts explicitly"
        )
    if len(candidates) > 1:
        names = ", ".join(c.name for c in candidates)
        raise ValueError(
            f"Multiple aggregate attempt files ({names}); use --attempts to pick one"
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate aggregate attempts JSON from a results run directory "
        "against ARC solutions."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Run folder (contains attempts/…)",
    )
    parser.add_argument(
        "--attempts",
        type=Path,
        default=None,
        help="Path to dict[task_id -> list[ChallengeSolution]] JSON "
        "(default: sole attempts/*_attempts.json under results_dir)",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        default=None,
        help="Ground-truth solutions JSON (default: data/arc-prize-2025/"
        "arc-agi_evaluation_solutions.json next to repo root inferred from results_dir)",
    )
    args = parser.parse_args()
    results_dir = args.results_dir
    if not results_dir.is_dir():
        raise SystemExit(f"Not a directory: {results_dir}")

    attempts_path = resolve_aggregate_attempts(results_dir, args.attempts)
    truth_path = (
        args.truth.expanduser().resolve()
        if args.truth is not None
        else _default_truth_path(results_dir)
    )
    if not truth_path.is_file():
        raise SystemExit(
            f"Truth solutions not found: {truth_path} (set --truth to a valid file)"
        )

    print(f"Attempts: {attempts_path}")
    print(f"Truth:    {truth_path}")
    evaluate_solutions(attempts_path, truth_path)


if __name__ == "__main__":
    main()
