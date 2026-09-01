"""Surveillance Module — observe surroundings by looking around.

Multi-step scan: look behind (1 step) → look forward (1 step) → record.
Only triggers after action completion (new position arrival).

policy="rule": no observation (skip)
policy="vlm":  VLM analyzes behind + forward RGB, records notable observations

Observations stored in memory → used during meetings for discussion.
Both crewmates (detect imposter) and imposters (gather intel).
"""

from __future__ import annotations

import math
from typing import List, Optional, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from ..env_adapter import AmongUsGameState

if TYPE_CHECKING:
    from ..state.base import BaseStateBuilder
    from ..memory.base import BaseMemory


class DirectionObservation(BaseModel):
    players_seen: bool = Field(description="True if any player is visible.")
    detail: str = Field(description="What you see (1 sentence).")
    suspect: str = Field(description="Player name if suspicious, or 'none'.")


_VLM_SCAN_CREWMATE = """\
You are a CREWMATE observing your surroundings in an Among Us-style Minecraft game.

Look at the RGB screenshot. Report what you see in this direction:
- Are there any players? What color outfit? What are they doing?
- Is anyone lurking without doing tasks? Following someone?
- Any colored carpets on the ground (dead bodies)?
- Any suspicious behavior?

If nothing notable, say so honestly.

OUTPUT (JSON):
{{
  "players_seen": true/false,
  "detail": "<what you see — 1 sentence>",
  "suspect": "<player name if suspicious behavior, or 'none'>"
}}
"""

_VLM_SCAN_IMPOSTER = """\
You are the IMPOSTER observing your surroundings in an Among Us-style Minecraft game.

Look at the RGB screenshot. Gather intelligence:
- Are there any players? Where? Alone or grouped?
- Is anyone watching you?
- Any isolated areas good for a kill?
- What tasks are visible crewmates doing?

OUTPUT (JSON):
{{
  "players_seen": true/false,
  "detail": "<what you see — 1 sentence>",
  "suspect": "<player name to track, or 'none'>"
}}
"""

# Text-only (no-camera) surveillance prompts. The agent reasons purely from
# player positions/Euclidean distance — no walls/line-of-sight. Used when
# config.use_vision is False.
_TEXT_SCAN_CREWMATE = """\
You are a CREWMATE in an Among Us-style game. You have NO camera — reason ONLY
from the player positions below (Euclidean distance; walls/line-of-sight unknown).

Your position: {own}
Other alive players (position, distance from you):
{others}

Note anything worth flagging: someone lingering very close to you, players
clustered oddly, or an isolated player. If nothing notable, say so honestly.

OUTPUT (JSON):
{{"players_seen": true/false, "detail": "<1 sentence>", "suspect": "<name or 'none'>"}}
"""

_TEXT_SCAN_IMPOSTER = """\
You are the IMPOSTER in an Among Us-style game. You have NO camera — reason ONLY
from the player positions below (Euclidean distance; walls/line-of-sight unknown).

Your position: {own}
Other alive players (position, distance from you):
{others}

Gather intel: who is isolated (far from others = a good kill target), who is
close enough to watch you, and who to track.

OUTPUT (JSON):
{{"players_seen": true/false, "detail": "<1 sentence>", "suspect": "<name to track or 'none'>"}}
"""

_SCAN_DIRS = [
    ("behind",  +math.pi),
    ("forward",  0.0),
]


class SurveillanceModule:
    def __init__(self, bot_name: str, role: str = "crewmate",
                 policy: str = "rule", vlm=None, use_vision: bool = True):
        self.bot_name = bot_name
        self.role = role
        self.policy = policy
        self.use_vision = use_vision
        self._chain = None
        if vlm is not None:
            self._chain = vlm | JsonOutputParser(pydantic_object=DirectionObservation)

        # Scan state machine
        self._scan_phase: int = 0  # 0=idle, 1=behind sent, 2=forward sent
        self._scan_results: List[dict] = []
        self._active: bool = False  # True when scan in progress
        self._last_observation: Optional[str] = None  # preserved after finalize for emergency check

    def set_vlm(self, vlm) -> None:
        self._chain = vlm | JsonOutputParser(pydantic_object=DirectionObservation)

    @property
    def is_scanning(self) -> bool:
        """True if a multi-step scan is in progress."""
        return self._active

    def start_scan(self) -> None:
        """Called by aria_agent when action completes → start surveillance."""
        if self.policy == "vlm" and self._chain is not None:
            self._active = True
            self._scan_phase = 0
            self._scan_results = []

    def step(
        self,
        game_state: AmongUsGameState,
        state_builder: "BaseStateBuilder",
        memory: "BaseMemory",
        verbose: bool = False,
    ) -> Optional[dict]:
        """Execute one step of the surveillance scan.

        Returns:
          dict with JS code (rotation) → continue scanning next step
          None → scan complete or not active
        """
        if not self._active or self.policy != "vlm" or self._chain is None:
            return None

        if game_state.phase != 0:
            self._reset()
            return None

        # No-vision (image-only ablation): DO NOT branch. Fall through to the
        # normal scan — with rgb_base64=None the scan prompt is sent WITHOUT an
        # image (only the "what do you see?" text).
        # ── Evaluate previous direction (no-vision: runs imagelessly) ──
        if self._scan_phase > 0:
            prev_label = _SCAN_DIRS[self._scan_phase - 1][0]
            result = self._vlm_observe(game_state, prev_label, verbose)
            self._scan_results.append(result)

        # ── If all directions scanned → finalize ──
        if self._scan_phase >= len(_SCAN_DIRS):
            self._finalize(state_builder, memory, verbose)
            return None  # scan complete

        # ── Rotate to next direction ──
        label, yaw_offset = _SCAN_DIRS[self._scan_phase]
        js_code = (
            f"async function surveil{label.capitalize()}(bot) {{\n"
            f"  const yaw = bot.entity.yaw + {yaw_offset};\n"
            f"  await bot.look(yaw, 0);\n"
            f"  await bot.waitForTicks(3);\n"
            f"}}\nawait surveil{label.capitalize()}(bot);"
        )

        self._scan_phase += 1

        if verbose:
            print(f"[Surveillance:{self.bot_name}] scanning: {label} "
                  f"(phase {self._scan_phase}/{len(_SCAN_DIRS)})")

        return {"code": js_code, "metadata": {"action": f"surveil_{label}"}}

    def get_last_observation(self) -> Optional[str]:
        """Get the combined observation text from the last completed scan.
        Preserved after _finalize/_reset so emergency check can read it."""
        return self._last_observation

    def _text_observe(self, game_state, state_builder, verbose: bool) -> dict:
        """No-camera observation: reason from player positions/distance only."""
        own = getattr(state_builder, "own_position", None)
        players = getattr(state_builder, "players", None)

        def _d(a, b):
            return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5

        lines = []
        if players and own:
            for name, p in players.items():
                if getattr(p, "is_dead", False) or p.position is None or name == self.bot_name:
                    continue
                lines.append(
                    f"  {name}: ({p.position[0]:.0f}, {p.position[1]:.0f}, {p.position[2]:.0f}) "
                    f"dist={_d(own, p.position):.0f}b")
        others = "\n".join(lines) if lines else "  (none nearby)"
        own_s = (f"({own[0]:.0f}, {own[1]:.0f}, {own[2]:.0f})" if own else "unknown")

        tmpl = _TEXT_SCAN_IMPOSTER if self.role == "imposter" else _TEXT_SCAN_CREWMATE
        system_text = tmpl.format(own=own_s, others=others)

        if self._chain is None:
            return {"direction": "around", "players_seen": bool(lines),
                    "detail": "(no chain)", "suspect": "none"}
        try:
            result = self._chain.invoke([
                SystemMessage(content=system_text),
                HumanMessage(content="Report your observation. Output JSON only."),
            ])
            if verbose:
                print(f"[Surveillance:{self.bot_name}] text: {result.get('detail', '')}")
            return {"direction": "around",
                    "players_seen": result.get("players_seen", False),
                    "detail": result.get("detail", ""),
                    "suspect": result.get("suspect", "none")}
        except Exception as e:
            if verbose:
                print(f"[Surveillance:{self.bot_name}] text error: {e}")
            return {"direction": "around", "players_seen": False,
                    "detail": f"error: {e}", "suspect": "none"}

    def _vlm_observe(self, game_state, direction: str, verbose: bool) -> dict:
        """VLM evaluates one direction's RGB."""
        system_text = (_VLM_SCAN_IMPOSTER if self.role == "imposter"
                       else _VLM_SCAN_CREWMATE)

        human_parts = [
            {"type": "text", "text": f"Direction: {direction}. What do you see?"},
        ]
        if game_state.rgb_base64:
            human_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{game_state.rgb_base64}",
                    "detail": "low",
                },
            })

        try:
            result = self._chain.invoke([
                SystemMessage(content=system_text),
                HumanMessage(content=human_parts),
            ])
            players_seen = result.get("players_seen", False)
            detail = result.get("detail", "")
            suspect = result.get("suspect", "none")

            if verbose:
                print(f"[Surveillance:{self.bot_name}] {direction}: "
                      f"{'players seen' if players_seen else 'clear'} — {detail}"
                      + (f" (suspect: {suspect})" if suspect != "none" else ""))

            return {
                "direction": direction,
                "players_seen": players_seen,
                "detail": detail,
                "suspect": suspect,
            }

        except Exception as e:
            if verbose:
                print(f"[Surveillance:{self.bot_name}] {direction} error: {e}")
            return {"direction": direction, "players_seen": False,
                    "detail": f"error: {e}", "suspect": "none"}

    def _finalize(self, state_builder, memory, verbose):
        """Combine scan results and record in memory."""
        notable = [r for r in self._scan_results if r["players_seen"]]

        if notable:
            combined = " | ".join(
                f"{r['direction']}: {r['detail']}" for r in self._scan_results
            )
            suspects = [r["suspect"] for r in notable
                        if r["suspect"] and r["suspect"].lower() != "none"]

            memory.add_module_entry("surveillance", {
                "tick": state_builder.get_tick(),
                "observation": combined,
                "suspects": suspects,
                "summary": f"SAW: {combined[:80]}"
                          + (f" [suspects={suspects}]" if suspects else ""),
            })

            for suspect in suspects:
                state_builder.record_suspicion(suspect, +0.1)

            # Save for emergency check (survives _reset)
            self._last_observation = combined

            if verbose:
                print(f"[Surveillance:{self.bot_name}] recorded: {combined[:100]}")
        else:
            self._last_observation = None
            if verbose:
                print(f"[Surveillance:{self.bot_name}] nothing notable")

        self._reset()

    def _reset(self):
        self._active = False
        self._scan_phase = 0
        self._scan_results = []
