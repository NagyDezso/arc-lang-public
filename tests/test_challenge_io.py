from __future__ import annotations

import json
from pathlib import Path

from src.models import Challenge


def test_grid_to_str_roundtrip() -> None:
    grid = [
        [0, 0, 0, 5, 0],
        [0, 5, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 5, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    s = Challenge.grid_to_str(grid)
    assert "5" in s
    assert Challenge.grid_from_str(s) == grid


def test_grid_from_str_picks_last_block() -> None:
    input_str = """Some header text...
    0 0 0 5 0
    0 5 0 0 0
    0 0 0 0 0
    0 5 0 0 0
    0 0 0 0 0
    random-text-here
    0 0 0 5 0
    0 5 0 0 0
    0 0 0 0 0
    0 5 0 0 0
    0 0 0 0 0
    Some footer text."""
    last = [
        [0, 0, 0, 5, 0],
        [0, 5, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 5, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    assert Challenge.grid_from_str(input_str) == last


def test_load_challenges_json(arc_challenges_path: Path) -> None:
    raw = json.loads(arc_challenges_path.read_text(encoding="utf-8"))
    assert len(raw) >= 1
    task_id, payload = next(iter(raw.items()))
    challenge = Challenge.model_validate({"task_id": task_id, **payload})
    assert challenge.task_id == task_id
    assert challenge.train


def test_to_basic_prompt_contains_examples(arc_challenges_path: Path) -> None:
    raw = json.loads(arc_challenges_path.read_text(encoding="utf-8"))
    task_id, payload = next(iter(raw.items()))
    challenge = Challenge.model_validate({"task_id": task_id, **payload})
    prompts = challenge.to_basic_prompt(use_cot=False)
    assert prompts
    text = prompts[0]
    assert "Example 1" in text
    assert "Input:" in text
    assert "Output:" in text
