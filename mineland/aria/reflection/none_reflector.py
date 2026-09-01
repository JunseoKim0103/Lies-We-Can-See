"""NoneReflector — no reflection (baseline)."""

from .base import BaseReflector, ReflectionResult


class NoneReflector(BaseReflector):
    def __init__(self):
        self._history = []

    def should_reflect(self, game_state, prev_game_state, action_completed):
        return False

    def reflect(self, state_context, memory_context, game_state, role, prompt_style):
        return ReflectionResult()
