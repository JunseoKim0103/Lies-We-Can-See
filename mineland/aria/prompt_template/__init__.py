"""Prompt template loader for Aria agent."""

from __future__ import annotations

import os
from typing import Optional

_DIR = os.path.dirname(os.path.abspath(__file__))


def load_prompt(style: str, filename: str) -> str:
    """Load a prompt template file.

    Parameters
    ----------
    style : "deterministic" or "minimal"
    filename : template name (with or without .txt extension)
    """
    if not filename.endswith(".txt"):
        filename += ".txt"
    path = os.path.join(_DIR, style, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""
