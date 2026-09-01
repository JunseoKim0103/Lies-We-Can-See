"""Egocentric state builder — only current step's observations."""

from __future__ import annotations

from typing import List, Optional

from .base import BaseStateBuilder
from ..env_adapter import AmongUsGameState


class EgocentricStateBuilder(BaseStateBuilder):
    """State built from current obs only — no position persistence.

    Applies a post-filter on distance so the LLM context truly reflects the
    bot's visibility radius (mineflayer's bot.entities is the server's
    view-distance, typically 6 chunks = 96 blocks, which is wider than the
    intended ego horizon).
    """

    def __init__(
        self,
        bot_name: str,
        role: str,
        teammate: Optional[str] = None,
        show_mission_progress: bool = True,
        visible_range: float = 36.0,
    ):
        super().__init__(bot_name, role, teammate=teammate,
                         show_mission_progress=show_mission_progress)
        self.visible_range = visible_range
        self._current_visible: List[dict] = []

    def _update_players(self, game_state: AmongUsGameState) -> None:
        """Store only currently visible players (vision-restricted, no persistence).

        Visibility criteria (all must hold):
        1. in_sight=True from observation_utils.js (LOS + frustum)
        2. distance < visible_range (additional defensive radius cap)
        """
        self._current_visible = [
            {"name": p.name, "position": p.position, "distance": p.distance}
            for p in game_state.visible_players
            if p.in_sight and (p.distance is None or p.distance < self.visible_range)
        ]

    def alive_player_names(self) -> List[str]:
        """Return names of currently visible players that aren't dead."""
        return [
            p["name"] for p in self._current_visible
            if not self.is_player_dead(p["name"])
        ]

    def _nearby_crewmates(self):
        # No distance cap; teammate INCLUDED with "(teammate)" tag so the
        # LLM can opt into team-kill deception. See privileged version for
        # rationale.
        out = []
        for p in self._current_visible:
            if self.is_player_dead(p["name"]):
                continue
            tag = " (your imposter teammate, not a crewmate)" if p["name"] == self.teammate else ""
            dist = f"dist={p['distance']:.0f}" if p["distance"] else ""
            out.append((p["name"], f"{dist}{tag}".strip()))
        return out

    def summarize_for_context(self, module_name: str, recent_chat_k: int = 10) -> str:
        lines = []
        if module_name == "reflection":
            lines.extend(self._authoritative_vote_outcome_lines())
        lines.extend(self._common_world_lines())

        # Ego: only show currently visible players
        visible_alive = [p for p in self._current_visible
                         if not self.is_player_dead(p["name"])]
        if visible_alive:
            lines.append("Currently visible players:")
            for p in visible_alive:
                pos_str = self._fmt_pos_with_room(p["position"])
                dist = f" dist={p['distance']:.0f}" if p["distance"] else ""
                lines.append(f"  {p['name']}: {pos_str}{dist}")
        else:
            lines.append("No players currently visible.")

        lines.extend(self._chat_context_lines(module_name, recent_chat_k))
        return "\n".join(lines)
