from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _require_integration_env() -> None:
    if os.environ.get("RUN_INTEGRATION") != "1":
        pytest.skip("Set RUN_INTEGRATION=1 to run integration tests (uses paid APIs).")
