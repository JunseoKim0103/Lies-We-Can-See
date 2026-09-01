"""Base reflector for Aria agent."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReflectionResult:
    insights: str = ""
    belief_updates: Dict[str, Any] = field(default_factory=dict)
    skill_candidates: List[dict] = field(default_factory=list)


class BaseReflector(ABC):
    @abstractmethod
    def should_reflect(
        self, game_state, prev_game_state, action_completed: bool
    ) -> bool:
        """Whether reflection should trigger this step."""

    @abstractmethod
    def reflect(
        self, state_context: str, memory_context: str,
        game_state, role: str, prompt_style: str,
    ) -> ReflectionResult:
        """Produce reflection output."""

    def save_to_json(self, path: str) -> None:
        history = getattr(self, "_history", [])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {"reflector": self.__class__.__name__, "history": history},
                f, ensure_ascii=False, indent=2,
            )
