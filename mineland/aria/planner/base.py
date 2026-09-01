"""Base planner for Aria agent."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class PlanResult:
    mode: str               # "kill", "report", "vote", "meeting", "mission", "explore"
    reasoning: str          # why this mode was chosen
    short_term_plan: str    # what to do
    long_term_plan: Optional[str] = None  # hierarchical only
    target: Optional[str] = None          # kill target name (if known)


class BasePlanner(ABC):
    @abstractmethod
    def plan(
        self,
        state_context: str,
        memory_context: str,
        skill_context: str,
        game_state,
        state_builder,
        role: str,
        prompt_style: str,
        **kwargs,
    ) -> PlanResult:
        """Decide what mode/action to take."""

    def save_to_json(self, path: str) -> None:
        history = getattr(self, "_history", [])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {"planner": self.__class__.__name__, "history": history},
                f, ensure_ascii=False, indent=2,
            )
