"""Bundled static assets (the app logo, etc.) and helpers to locate them."""

from __future__ import annotations

from pathlib import Path


def logo_path() -> str:
    """Absolute path to the CRYSTALLine logo (SVG)."""
    return str(Path(__file__).with_name("logo.svg"))


__all__ = ["logo_path"]
