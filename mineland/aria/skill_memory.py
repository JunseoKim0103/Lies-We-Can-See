"""Skill Memory — list-based storage of strategic patterns."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class SkillEntry:
    source: str                         # "self" or "observed"
    agent_name: str                     # who did it
    phase: str                          # "explore", "meeting", "vote"
    role_hint: Optional[str] = None     # "imposter", "crewmate", or None
    situation: str = ""                 # when/why
    behavior: str = ""                  # what was done
    outcome: str = ""                   # result
    tick: int = 0


class SkillMemory:
    """Recency-based list of strategic skill entries.

    Entries are added by the reflection module. Retrieved by planners
    and action modules for LLM context enrichment.
    """

    def __init__(self, max_size: int = 50, sources: Optional[List[str]] = None):
        self.max_size = max_size
        self.sources = sources or ["self"]
        self._entries: List[SkillEntry] = []

    def add(self, entry: SkillEntry) -> None:
        """Add a skill entry (filtered by allowed sources)."""
        if entry.source not in self.sources:
            return
        self._entries.append(entry)
        if len(self._entries) > self.max_size:
            self._entries.pop(0)

    def get_relevant(self, situation: str = "", k: int = 5) -> List[SkillEntry]:
        """Return most recent k entries (recency-based, no vector search)."""
        return self._entries[-k:]

    def get_context(self, k: int = 5) -> str:
        """Format recent skills as text for LLM context."""
        entries = self.get_relevant("", k)
        if not entries:
            return ""
        lines = ["[Skill Memory — recent entries]"]
        for e in entries:
            lines.append(
                f"  [{e.source}] {e.situation} → {e.behavior} → {e.outcome}"
            )
        return "\n".join(lines)

    def save_to_json(self, path: str) -> None:
        data = [asdict(e) for e in self._entries]
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[SkillMemory] save failed: {e}")

    def load_from_json(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._entries = [SkillEntry(**d) for d in data]
            return True
        except Exception:
            return False
