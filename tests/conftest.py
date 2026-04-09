from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _arc_challenges_path(repo_root: Path) -> Path | None:
    base = repo_root / "data" / "arc-prize-2024"
    for name in (
        "arc-agi_training_challenges.json",
        "arc-agi_test_challenges.json",
        "arc-agi_evaluation_challenges.json",
    ):
        p = base / name
        if p.is_file():
            return p
    return None


@pytest.fixture(scope="session")
def arc_challenges_path(repo_root: Path) -> Path:
    p = _arc_challenges_path(repo_root)
    if p is None:
        pytest.skip("No ARC challenges JSON under data/arc-prize-2024/")
    return p
