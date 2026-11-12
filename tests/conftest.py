"""Shared pytest configuration for the test suite."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``slow`` marker where ``pyproject.toml`` is not in reach.

    The wheel tests run from a copy of ``tests/`` outside the repository, so
    the ``markers`` list of ``[tool.pytest.ini_options]`` is not read there.
    """
    config.addinivalue_line(
        "markers",
        "slow: exhaustive versions of invariance checks that also have a quick "
        "variant (pull requests run -m 'not slow')",
    )
