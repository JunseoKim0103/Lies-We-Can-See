"""NoneMemory — no persistent memory (stateless baseline)."""

from __future__ import annotations

from typing import List

from .base import BaseMemory


class NoneMemory(BaseMemory):
    """All operations are no-ops. Agent sees only current state."""

    def update(self, tick, action_summary, state_summary, events):
        pass

    def get_context(self, module_name, max_entries=5):
        return ""

    def add_reflection(self, reflection_text):
        pass

    def get_module_history(self, module_name, k=5):
        return []

    def add_module_entry(self, module_name, entry):
        pass

    def save_to_json(self, path):
        pass

    def load_from_json(self, path):
        return False
