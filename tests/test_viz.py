from __future__ import annotations

import base64

import matplotlib

matplotlib.use("Agg", force=True)

from src.models import COLOR_MAP
from src.viz import base64_from_grid


def test_base64_from_grid_is_png() -> None:
    grid = [[1, 2, 1], [2, 0, 2], [1, 2, 1]]
    b64 = base64_from_grid(grid)
    assert len(b64) > 100
    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_color_map_has_expected_keys() -> None:
    assert COLOR_MAP[0] == "black"
    assert 9 in COLOR_MAP
