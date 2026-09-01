"""Base class for Aria memory modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class BaseMemory(ABC):
    """Interface for all memory variants."""

    @abstractmethod
    def update(self, tick: int, action_summary: str, state_summary: str,
               events: List[dict]) -> None:
        """Store new step info."""

    @abstractmethod
    def get_context(self, module_name: str, max_entries: int = 5) -> str:
        """Return text context for LLM prompts."""

    @abstractmethod
    def add_reflection(self, reflection_text: str) -> None:
        """Store reflection output."""

    @abstractmethod
    def get_module_history(self, module_name: str, k: int = 5) -> List[dict]:
        """Get recent decisions for a specific module."""

    @abstractmethod
    def add_module_entry(self, module_name: str, entry: dict) -> None:
        """Store a module's decision."""

    @abstractmethod
    def save_to_json(self, path: str) -> None:
        """Persist to JSON file."""

    @abstractmethod
    def load_from_json(self, path: str) -> bool:
        """Load from JSON file. Returns True on success."""
