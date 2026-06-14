"""Score saved attempt JSON against ARC ground-truth solutions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from src.models import GRID
from src.usage import print_run_usage

# Patterns matched against saved agy transcripts via re.search. Any hit means
# the model reached for tools / network / secrets despite the no-tools clamp.
_SUSPICIOUS_PATTERNS = [
    # Any non-empty tool_calls array in the transcript JSONL.
    r'"tool_calls":\s*\[\s*\{',
    # Specific high-risk tool families.
    r'"name":"search_web"',
    r'"name":"web_search"',
    r'"name":"read_url',
    r'"name":"browser_',
    r'"name":"antigravity_browser"',
    r'"name":"run_command"',
    r'"name":"view_file"',
    r'"name":"write_to_file"',
    r'"name":"edit_file"',
    r"vertexaisearch",
    # Secret / env access attempts.
    r"GEMINI_API_KEY",
    r"GOOGLE_API_KEY",
    r"ANTHROPIC_API_KEY",
    r"OPENAI_API_KEY",
    r"DEEPMIND_API_KEY",
    r"ANTIGRAVITY_OAUTH_REFRESH_TOKEN",
    r"os\.environ",
    r"/proc/self/environ",
    r"/proc/\d+/environ",
]
_SUSPICIOUS_REGEXES = [re.compile(p) for p in _SUSPICIOUS_PATTERNS]


def check_transcripts(transcripts_dir: Path) -> list[str]:
    """Scan saved agy transcripts for tool use, network access, or env probing.

    Returns one warning string per (file, first matching pattern). Empty list
    means clean.
    """
    warnings: list[str] = []
    if not transcripts_dir.is_dir():
        return warnings

    for transcript_file in sorted(transcripts_dir.rglob("*.jsonl")):
        rel = transcript_file.relative_to(transcripts_dir)
        try:
            text = transcript_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            warnings.append(f"[{rel}] unreadable transcript: {exc}")
            continue
        for regex in _SUSPICIOUS_REGEXES:
            match = regex.search(text)
            if match is not None:
                warnings.append(
                    f'[{rel}] suspicious pattern "{regex.pattern}" at index '
                    f"{match.start()}"
                )
                break
    return warnings


class ChallengeSolution(BaseModel):
    attempt_1: GRID
    attempt_2: GRID


def evaluate_solutions(
    attempts_solutions_path: Path,
    truth_solutions_path: Path,
    show_task_ids: bool = False,
) -> None:
    truth: dict[str, list[GRID]] = json.loads(
        truth_solutions_path.read_text(encoding="utf-8")
    )
    attempts = {}
    if attempts_solutions_path.is_file():
        attempts = TypeAdapter(dict[str, list[ChallengeSolution]]).validate_json(
            attempts_solutions_path.read_text(encoding="utf-8")
        )
    total_count = 0
    correct_count = 0
    successful_task_ids: list[str] = []
    failed_task_ids: list[str] = []
    for challenge_id, attempt_list in attempts.items():
        truth_grids: list[GRID] = truth[challenge_id]
        task_all_correct = True
        for i, truth_grid in enumerate(truth_grids):
            total_count = total_count + 1
            attempt_grids = attempt_list[i]
            if attempt_grids.attempt_1 == truth_grid:
                correct_count = correct_count + 1
            elif attempt_grids.attempt_2 == truth_grid:
                correct_count = correct_count + 1
            else:
                task_all_correct = False
        if task_all_correct:
            successful_task_ids.append(challenge_id)
        else:
            failed_task_ids.append(challenge_id)

    incorrect_count = total_count - correct_count
    accuracy = (correct_count / total_count * 100) if total_count else 0.0

    print("\n=== Final Evaluation Results ===")
    print(f"Total test cases : {total_count}")
    print(f"Correct          : {correct_count}")
    print(f"Incorrect        : {incorrect_count}")
    print(f"Accuracy         : {accuracy:.2f}%")
    print(f"Successful tasks : {len(successful_task_ids)}")
    print(f"Failed tasks     : {len(failed_task_ids)}")

    if show_task_ids:
        print("\n--- Successful task ids ---")
        print(", ".join(successful_task_ids))
        print("\n--- Failed task ids ---")
        print(", ".join(failed_task_ids))

    # If the run used the agy provider, the adapter saved each conversation
    # transcript under results/<run>/agy_transcripts. Scan them for tool use
    # or other rule-breaking before declaring the run clean.
    run_dir = attempts_solutions_path.resolve().parent.parent
    print_run_usage(run_dir)
    transcripts_dir = run_dir / "agy_transcripts"
    transcript_warnings = check_transcripts(transcripts_dir)
    if transcript_warnings:
        print(f"\n=== Suspicious transcripts ({len(transcript_warnings)}) ===")
        for w in transcript_warnings:
            print(w)
    elif transcripts_dir.is_dir():
        print("\n=== Transcript check: clean ===")


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
    parser.add_argument(
        "--show-task-ids",
        action="store_true",
        help="Also print the lists of successful and failed task ids",
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
    evaluate_solutions(attempts_path, truth_path, show_task_ids=args.show_task_ids)


if __name__ == "__main__":
    main()
