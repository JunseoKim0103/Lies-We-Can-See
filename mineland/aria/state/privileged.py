"""Privileged state builder — stores player positions across steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .base import BaseStateBuilder
from ..env_adapter import AmongUsGameState


@dataclass
class PlayerInfo:
    name: str
    position: Optional[Tuple[float, float, float]] = None
    last_seen_tick: int = -1
    is_dead: bool = False


class PrivilegedStateBuilder(BaseStateBuilder):
    """State with persistent player position tracking (like Steve's GlobalMemory)."""

    def __init__(
        self,
        bot_name: str,
        role: str,
        all_players: Optional[List[str]] = None,
        teammate: Optional[str] = None,
        show_mission_progress: bool = True,
        visible_range: Optional[float] = None,
    ):
        super().__init__(bot_name, role, teammate=teammate,
                         show_mission_progress=show_mission_progress)
        # Optional perception-radius cap on coordinate ingestion. None = unlimited
        # (server view ~96 blocks). When set, players farther than this are NOT
        # ingested this step, but their previously-stored last-known position is
        # kept (persistence is preserved) — this isolates the distance cheat.
        self.visible_range = visible_range
        self.players: Dict[str, PlayerInfo] = {
            name: PlayerInfo(name=name)
            for name in (all_players or [])
            if name != bot_name
        }
        self._current_visible: list = []

    def _update_players(self, game_state: AmongUsGameState) -> None:
        """Update stored positions from visible entities."""
        self._current_visible = []
        for p in game_state.visible_players:
            if p.name == self.bot_name:
                continue
            # Distance-cap: skip ingesting players beyond the perception radius.
            # Their prior last-known position stays in self.players (persistence).
            if (self.visible_range is not None and p.distance is not None
                    and p.distance >= self.visible_range):
                continue
            if p.name not in self.players:
                self.players[p.name] = PlayerInfo(name=p.name)
            self.players[p.name].position = p.position
            self.players[p.name].last_seen_tick = self.tick
            self._current_visible.append(
                {"name": p.name, "distance": p.distance}
            )

        # Sync death info
        for name in self._dead_players:
            if name in self.players:
                self.players[name].is_dead = True

    def alive_player_names(self) -> List[str]:
        return [n for n, p in self.players.items()
                if not p.is_dead and n != self.bot_name]

    def nearest_alive_player(self, verbose: bool = False) -> Optional[str]:
        """Find nearest alive player by stored position."""
        if self.own_position is None:
            if verbose:
                print(f"[nearest_alive_player:{self.bot_name}] own_position is None")
            return None
        best_name, best_dist = None, float("inf")
        ox, oy, oz = self.own_position
        skipped = []
        for p in self.players.values():
            if p.is_dead or p.position is None or p.name == self.teammate:
                skipped.append(f"{p.name}(dead={p.is_dead},pos={'None' if p.position is None else 'ok'},tm={p.name == self.teammate})")
                continue
            dx = p.position[0] - ox
            dy = p.position[1] - oy
            dz = p.position[2] - oz
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dist < best_dist:
                best_dist, best_name = dist, p.name
        if verbose and best_name is None:
            print(f"[nearest_alive_player:{self.bot_name}] no valid player found. skipped={skipped}")
        return best_name

    def _nearby_crewmates(self):
        # No distance cap: list every visible alive non-self player with
        # their distance. Teammate is INCLUDED but tagged "(teammate)" so
        # the LLM can choose to team-kill as a deception tactic if it
        # judges the situation worth it. Default personal_message guidance
        # still says treat teammate as ally; this just makes the option
        # visible to the LLM rather than hiding it.
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

        alive = [p for p in self.players.values() if not p.is_dead]
        if alive:
            lines.append("Alive players:")
            for p in alive:
                age = self.tick - p.last_seen_tick if p.last_seen_tick >= 0 else -1
                age_str = f"({age}t ago)" if age >= 0 else "(never seen)"
                lines.append(f"  {p.name}: {self._fmt_pos(p.position)}  {age_str}")

        lines.extend(self._chat_context_lines(module_name, recent_chat_k))
        return "\n".join(lines)

    def _extra_save_data(self) -> dict:
        return {
            "players": {n: {
                "name": p.name,
                "position": list(p.position) if p.position else None,
                "last_seen_tick": p.last_seen_tick,
                "is_dead": p.is_dead,
            } for n, p in self.players.items()}
        }

    def _load_extra_data(self, data: dict) -> None:
        for n, d in data.get("players", {}).items():
            pos = tuple(d["position"]) if d.get("position") else None
            self.players[n] = PlayerInfo(
                name=d["name"], position=pos,
                last_seen_tick=d.get("last_seen_tick", -1),
                is_dead=d.get("is_dead", False),
            )
