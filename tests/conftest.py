"""Shared pytest and Hypothesis configuration."""

from __future__ import annotations

import os

from hypothesis import settings

pytest_plugins = ("tests.pytest_policy",)

settings.register_profile("development", max_examples=25)
settings.register_profile("ci", max_examples=100, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))
