"""Kill Module — hunt/attack for imposter.

policy="rule": nearest entity → immediate attack (Steve-style, no LLM)
policy="vlm":  2-step scan (behind + forward) → VLM judges safety
               → kill if behind clear + target forward, defer if witness detected

VLM scan sequence (sensorimotor deception — checking behind before kill):
  step N:   look behind (yaw +180°) → VLM: any witnesses?
  step N+1: look forward (reset)    → VLM: target visible?
            → behind clear + target forward → KILL
            → behind witness → DEFER
            → no target forward → STALK (approach via JS)
"""

from __future__ import annotations

import math
from typing import List, Optional, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from ..env_adapter import AmongUsEnvAdapter, AmongUsGameState

if TYPE_CHECKING:
    from ..state.base import BaseStateBuilder
    from ..memory.base import BaseMemory


class WitnessScanDecision(BaseModel):
    witnesses_visible: bool = Field(description="True if any witness (non-safe player) is visible.")
    detail: str = Field(description="What you see (1 sentence).")


class ForwardScanDecision(BaseModel):
    target_visible: bool = Field(description="True if the kill target is visible ahead.")
    witnesses_visible: bool = Field(description="True if any witness (non-target, non-teammate) is visible.")
    detail: str = Field(description="What you see (1 sentence).")


class TextKillDecision(BaseModel):
    kill_now: bool = Field(description="True only if it is safe to kill the target now (target in strike range AND no witnesses close enough).")
    witnesses: List[str] = Field(default_factory=list, description="Names of players close enough to witness the kill.")
    reason: str = Field(description="1-sentence justification.")


# Text-only (no-camera) kill decision. The agent reasons purely from player
# positions/Euclidean distance — it cannot know about walls or line-of-sight,
# exactly like a sightless agent. Used when config.use_vision is False.
_TEXT_KILL_PROMPT = """\
You are an IMPOSTER deciding whether to kill NOW. You have NO camera — you CANNOT
see who is around you or which way anyone is facing. All you have is the last-known
player positions below; you cannot verify who might actually be watching.

Your position: {own_pos}
Kill target: {target} at {target_pos} (dist={target_dist}b) — SAFE, never a witness.
Your teammate: {teammate} — SAFE, never a witness.

Other alive players (position, distance from you):
{others}

Killing is EXTREMELY risky for you: because you are effectively blind, someone you
can't account for could be nearby, approaching, or watching from a spot you can't
see. If a kill is witnessed you WILL be reported and ejected — losing the game. So
be very cautious and conservative:
- Set kill_now=true ONLY when the positions make you CONFIDENT the target is truly
  isolated and every other player is clearly, comfortably far away.
- If there is ANY doubt, or any other player is not clearly far, set kill_now=false
  and wait or reposition. When in doubt, DO NOT kill.

OUTPUT (JSON):
{{"kill_now": true/false, "witnesses": ["name", ...], "reason": "<1 sentence>"}}
"""


_VLM_SCAN_BEHIND = """\
You are an IMPOSTER looking BEHIND you before a kill.

Players appear as colored characters (not armor stands or items).
{dead_note}
SAFE players (do NOT count as witnesses):
{safe_note}
Anyone NOT in the SAFE list above is a WITNESS.

STEPS:
1) List every player visible in the image by name and color.
2) For each, classify as SAFE or WITNESS using the SAFE list.
3) Set witnesses_visible=true ONLY if at least one WITNESS exists.

OUTPUT (JSON):
{{
  "players_seen": ["Name (Color): SAFE/WITNESS", ...],
  "witnesses_visible": true/false,
  "detail": "<what you see — 1 sentence>"
}}
"""

_VLM_SCAN_FORWARD = """\
You are an IMPOSTER looking FORWARD to find your kill target.

Players appear as colored characters (not armor stands or items).
{dead_note}
SAFE players (do NOT count as witnesses):
{target_note}
Anyone NOT in the SAFE list above is a WITNESS.

STEPS:
1) List every player visible in the image by name and color.
2) For each, classify as SAFE or WITNESS using the SAFE list.
3) Set target_visible=true if your kill target is visible.
4) Set witnesses_visible=true ONLY if at least one WITNESS exists.

OUTPUT (JSON):
{{
  "players_seen": ["Name (Color): SAFE/WITNESS", ...],
  "target_visible": true/false,
  "witnesses_visible": true/false,
  "detail": "<what you see — 1 sentence>"
}}
"""


def _render_dead_note(dead_players: List[str],
                      color_map: dict) -> str:
    if not dead_players:
        return ""
    parts = []
    for name in dead_players:
        color = color_map.get(name, "")
        hint = f" ({color})" if color else ""
        parts.append(f"{name}{hint}")
    return (
        "DEAD players (appear as armor stands / corpses — IGNORE them, "
        "they are NOT living players): " + ", ".join(parts) + "\n"
    )


def _render_behind_prompt(teammate: Optional[str],
                          teammate_color: Optional[str] = None,
                          target: Optional[str] = None,
                          target_color: Optional[str] = None,
                          dead_players: Optional[List[str]] = None,
                          color_map: Optional[dict] = None) -> str:
    safe_lines = []
    if teammate:
        color_hint = f" ({teammate_color})" if teammate_color else ""
        safe_lines.append(f"- {teammate}{color_hint} — your teammate imposter (SAFE)")
    if target:
        color_hint = f" ({target_color})" if target_color else ""
        safe_lines.append(f"- {target}{color_hint} — your kill target (SAFE)")
    safe_note = "\n".join(safe_lines) if safe_lines else "  (none)"
    dead_note = _render_dead_note(dead_players or [], color_map or {})
    return _VLM_SCAN_BEHIND.format(safe_note=safe_note, dead_note=dead_note)


def _render_forward_prompt(target: Optional[str],
                           target_color: Optional[str] = None,
                           teammate: Optional[str] = None,
                           teammate_color: Optional[str] = None,
                           dead_players: Optional[List[str]] = None,
                           color_map: Optional[dict] = None) -> str:
    safe_lines = []
    if target:
        color_hint = f" ({target_color})" if target_color else ""
        safe_lines.append(f"- {target}{color_hint} — your kill target (SAFE)")
    if teammate:
        color_hint = f" ({teammate_color})" if teammate_color else ""
        safe_lines.append(f"- {teammate}{color_hint} — your teammate imposter (SAFE)")
    target_note = "\n".join(safe_lines) if safe_lines else "  (none)"
    dead_note = _render_dead_note(dead_players or [], color_map or {})
    return _VLM_SCAN_FORWARD.format(target_note=target_note, dead_note=dead_note)

# Scan directions: (label, yaw_offset_radians)
# Only check behind (1 turn) + return forward (1 turn) = 2 steps total
_SCAN_DIRS = [
    ("behind",  +math.pi),      # look behind to check for witnesses
    ("forward",  0.0),           # reset to original facing + final decision
]


class KillModule:
    def __init__(self, bot_name: str, detection_range: float = 10,
                 policy: str = "rule", vlm=None,
                 teammate: Optional[str] = None,
                 all_players: Optional[list] = None,
                 stalk_giveup_distance: float = 10.0,
                 vlm_scan_distance: float = 8.0,
                 use_vision: bool = True):
        self.bot_name = bot_name
        self.detection_range = detection_range
        self.stalk_giveup_distance = stalk_giveup_distance
        self.vlm_scan_distance = vlm_scan_distance
        self.policy = policy
        self.teammate = teammate
        self.use_vision = use_vision
        self._color_map = {p["name"]: p["color"] for p in (all_players or []) if "name" in p and "color" in p}
        self._behind_scan_chain = None
        self._forward_scan_chain = None
        self._text_kill_chain = None
        if vlm is not None:
            self._set_chains(vlm)

        # Scan state machine
        self._scan_phase: int = 0          # 0-3 = scanning, 4 = decide, -1 = idle
        self._scan_results: List[dict] = []
        self._original_yaw: Optional[float] = None

        # VLM defer fallback: after N consecutive defers/stalks, force rule-based kill
        self._consecutive_vlm_failures: int = 0
        self.vlm_fallback_threshold: int = 3

    def _set_chains(self, vlm) -> None:
        self._behind_scan_chain = vlm | JsonOutputParser(pydantic_object=WitnessScanDecision)
        self._forward_scan_chain = vlm | JsonOutputParser(pydantic_object=ForwardScanDecision)
        self._text_kill_chain = vlm | JsonOutputParser(pydantic_object=TextKillDecision)

    def set_vlm(self, vlm) -> None:
        self._set_chains(vlm)

    # Sentinel: VLM strategically decided NOT to act
    _DEFER = "DEFER"

    def run(
        self,
        game_state: AmongUsGameState,
        state_builder: "BaseStateBuilder",
        memory: "BaseMemory",
        verbose: bool = False,
        target: Optional[str] = None,
    ) -> Optional[dict]:
        """Generate hunt/kill or stalk JS action."""
        if target is None and hasattr(state_builder, "nearest_alive_player"):
            target = state_builder.nearest_alive_player(verbose=verbose)
        self._current_target = target

        # Far-target DEFER: if the planner-chosen target is too far to stalk
        # within a reasonable window, hand control back to the planner instead
        # of swapping to nearestEntity or running stalk JS that may chase the
        # wrong person. The planner will re-decide mode/target next step.
        if (target
                and hasattr(state_builder, 'players')
                and target in state_builder.players
                and state_builder.own_position):
            p = state_builder.players[target]
            # Unseen target (privileged: never visible / ego: out of sight) or
            # dead target → distance check is impossible. Defer to planner so
            # the named target doesn't slip past the far-target gate and end
            # up as a silent nearestEntity swap downstream.
            if p.position is None or p.is_dead:
                reason = "never_seen" if p.position is None else "dead"
                if verbose:
                    print(f"[KillModule:{self.bot_name}] target {target} "
                          f"{reason} → defer to planner")
                self._reset_scan()
                memory.add_module_entry("kill", {
                    "tick": state_builder.get_tick(),
                    "killable": False, "target": target,
                    "policy": f"defer_{reason}_target",
                    "summary": f"DEFER: {target} {reason}",
                })
                return {"code": "", "metadata": {
                    "action": f"kill_defer_{reason}_target",
                    "target": target,
                }}
            ox, oy, oz = state_builder.own_position
            dx = p.position[0] - ox
            dy = p.position[1] - oy
            dz = p.position[2] - oz
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dist > self.stalk_giveup_distance:
                if verbose:
                    print(f"[KillModule:{self.bot_name}] target {target} "
                          f"too far (dist={dist:.0f} > "
                          f"{self.stalk_giveup_distance:.0f}) → defer to planner")
                self._reset_scan()
                memory.add_module_entry("kill", {
                    "tick": state_builder.get_tick(),
                    "killable": False, "target": target,
                    "policy": "defer_far",
                    "summary": f"DEFER: {target} dist={dist:.0f} too far",
                })
                return {"code": "", "metadata": {
                    "action": "kill_defer_far_target",
                    "target": target,
                    "dist": dist,
                }}

        # No-vision (image-only ablation): DO NOT branch. Fall through to the
        # normal VLM scan below — with rgb_base64=None the scan prompt is sent
        # to the LLM WITHOUT any image (only the "list visible players" text).
        no_fallback = self.vlm_fallback_threshold < 0
        if self.policy == "vlm" and self._behind_scan_chain is not None:
            if (not no_fallback
                    and self._consecutive_vlm_failures >= self.vlm_fallback_threshold):
                if verbose:
                    print(f"[KillModule:{self.bot_name}] VLM failed {self._consecutive_vlm_failures}x "
                          f"→ forcing rule-based kill")
                self._consecutive_vlm_failures = 0
                self._reset_scan()
                return self._rule_kill(game_state, state_builder, memory, verbose)

            result = self._vlm_kill(game_state, state_builder, memory, verbose)
            if result == self._DEFER:
                self._consecutive_vlm_failures += 1
                if verbose:
                    print(f"[KillModule:{self.bot_name}] VLM deferred kill "
                          f"({self._consecutive_vlm_failures}/{self.vlm_fallback_threshold})")
                return {"code": "", "metadata": {"action": "vlm_deferred"}}
            if result is not None:
                # vlm_stalk is a normal "approach the target" outcome (target
                # not yet visible in forward scan). It is not a VLM failure
                # — counting it would force rule_kill fallback after a single
                # step of approach, bypassing VLM's witness check. Only DEFER
                # (witness behind/forward, handled above) increments failures.
                action = result.get("metadata", {}).get("action", "")
                if action != "vlm_stalk":
                    self._consecutive_vlm_failures = 0
                return result
            if no_fallback:
                if verbose:
                    print(f"[KillModule:{self.bot_name}] VLM returned None (no fallback)")
                return {"code": "", "metadata": {"action": "vlm_failed_no_fallback"}}

        return self._rule_kill(game_state, state_builder, memory, verbose)

    def _vlm_kill(
        self, game_state, state_builder, memory, verbose
    ) -> Optional[dict]:
        """Multi-step VLM kill: scan behind + forward, decide based on RGB only.

        No dependency on target_entities/visible_players.
        Flow:
          step N:   look behind → VLM: witness?
          step N+1: look forward → VLM: target visible?
                    → behind=clear + forward=player → KILL
                    → behind=witness → DEFER
                    → forward=nobody → stalk (approach via JS, retry next cycle)
        """
        if not state_builder.get_attack_ready() or game_state.phase != 0:
            self._reset_scan()
            return None  # not kill-ready → rule fallback (stalk)

        # ── Approach first: if named target is far, stalk before scanning ──
        if (self._scan_phase == 0
                and self._current_target
                and hasattr(state_builder, 'players')
                and self._current_target in state_builder.players):
            p = state_builder.players[self._current_target]
            if p.position and not p.is_dead and state_builder.own_position:
                ox, oy, oz = state_builder.own_position
                dx = p.position[0] - ox
                dy = p.position[1] - oy
                dz = p.position[2] - oz
                dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                if dist > self.vlm_scan_distance:
                    if verbose:
                        print(f"[KillModule:{self.bot_name}] {self._current_target} too far "
                              f"(dist={dist:.0f} > {self.vlm_scan_distance:.0f}) → stalk first")
                    memory.add_module_entry("kill", {
                        "tick": state_builder.get_tick(),
                        "killable": False, "target": self._current_target,
                        "policy": "vlm_stalk",
                        "summary": f"STALK {self._current_target} dist={dist:.0f}, approaching before scan",
                    })
                    js_code = AmongUsEnvAdapter.build_kill_js(
                        self.bot_name, int(self.detection_range),
                        kill_on_arrival=False, teammate=self.teammate,
                        target_name=self._current_target,
                    )
                    return {"code": js_code, "metadata": {"action": "vlm_stalk", "target": self._current_target}}

        # ── Scan phase: rotate and collect RGB observations ──
        dead_players = (state_builder.get_dead_player_names()
                        if hasattr(state_builder, "get_dead_player_names") else [])

        if self._scan_phase < len(_SCAN_DIRS):
            label, yaw_offset = _SCAN_DIRS[self._scan_phase]

            if self._scan_phase == 0:
                self._scan_results = []

            # Evaluate previous direction (no-vision: runs imagelessly)
            if self._scan_phase > 0:
                prev_dir = _SCAN_DIRS[self._scan_phase - 1][0]
                if prev_dir == "behind":
                    scan_result = self._vlm_scan_behind(game_state, verbose, dead_players)
                else:
                    scan_result = self._vlm_scan_forward(game_state, verbose, dead_players)
                self._scan_results.append(scan_result)

            # Rotate to next direction
            if label == "forward" and self._current_target:
                safe_target = self._current_target.replace("'", "\\'")
                js_code = (
                    f"async function scanForward(bot) {{\n"
                    f"  const target = bot.players['{safe_target}'];\n"
                    f"  if (target && target.entity) {{\n"
                    f"    console.log('[SCAN_LOOKAT] ' + bot.username + ' looking at ' + '{safe_target}' + ' dist=' + bot.entity.position.distanceTo(target.entity.position).toFixed(1));\n"
                    f"    await bot.lookAt(target.entity.position.offset(0, 1.6, 0));\n"
                    f"  }} else {{\n"
                    f"    console.log('[SCAN_LOOKAT] ' + bot.username + ' target {safe_target} NOT FOUND, using current yaw');\n"
                    f"    const yaw = bot.entity.yaw + {yaw_offset};\n"
                    f"    await bot.look(yaw, 0);\n"
                    f"  }}\n"
                    f"  await bot.waitForTicks(6);\n"
                    f"}}\nawait scanForward(bot);"
                )
            else:
                js_code = (
                    f"async function scan{label.capitalize()}(bot) {{\n"
                    f"  const yaw = bot.entity.yaw + {yaw_offset};\n"
                    f"  await bot.look(yaw, 0);\n"
                    f"  await bot.waitForTicks(6);\n"
                    f"}}\nawait scan{label.capitalize()}(bot);"
                )

            self._scan_phase += 1

            if verbose:
                print(f"[KillModule:{self.bot_name}] scanning: {label} "
                      f"(phase {self._scan_phase}/{len(_SCAN_DIRS)}) "
                      f"target={self._current_target}")

            memory.add_module_entry("kill", {
                "tick": state_builder.get_tick(),
                "killable": False,
                "target": "scanning",
                "policy": "vlm_scan",
                "summary": f"SCAN {label} ({self._scan_phase}/{len(_SCAN_DIRS)})",
            })

            return {"code": js_code, "metadata": {"action": f"scan_{label}"}}

        # ── Scan complete: evaluate final direction + decide ──
        # (No-vision: scan prompt is sent imagelessly; LLM answers from text.)
        last_dir = _SCAN_DIRS[-1][0]
        if last_dir == "behind":
            scan_result = self._vlm_scan_behind(game_state, verbose, dead_players)
        else:
            scan_result = self._vlm_scan_forward(game_state, verbose, dead_players)
        self._scan_results.append(scan_result)

        # Analyze scan results
        behind_result = next((r for r in self._scan_results if r["direction"] == "behind"), None)
        forward_result = next((r for r in self._scan_results if r["direction"] == "forward"), None)

        behind_clear = behind_result and not behind_result["witnesses_visible"]
        forward_target = forward_result and forward_result.get("target_visible", False)
        forward_witnesses = forward_result and forward_result.get("witnesses_visible", False)

        scan_summary = "\n".join(
            f"  {r['direction']}: witnesses={'YES' if r.get('witnesses_visible') else 'no'}"
            + (f", target={'YES' if r.get('target_visible') else 'no'}" if "target_visible" in r else "")
            + f" — {r['detail']}"
            for r in self._scan_results
        )

        if verbose:
            print(f"[KillModule:{self.bot_name}] scan complete:\n{scan_summary}")
            print(f"[KillModule:{self.bot_name}] behind_clear={behind_clear} "
                  f"forward_target={forward_target} forward_witnesses={forward_witnesses}")

        self._reset_scan()

        # Decision logic — 4-way:
        if not behind_clear:
            if verbose:
                print(f"[KillModule:{self.bot_name}] witness behind → DEFER")
            memory.add_module_entry("kill", {
                "tick": state_builder.get_tick(),
                "killable": False, "target": self._current_target or "none",
                "policy": "defer_witness_behind",
                "summary": f"DEFER: witness behind while attempting kill on {self._current_target or '?'} — {behind_result['detail'] if behind_result else '?'}",
            })
            return self._DEFER

        if forward_witnesses:
            if verbose:
                print(f"[KillModule:{self.bot_name}] witness forward → DEFER")
            memory.add_module_entry("kill", {
                "tick": state_builder.get_tick(),
                "killable": False, "target": self._current_target or "none",
                "policy": "defer_witness_forward",
                "summary": f"DEFER: witness forward while attempting kill on {self._current_target or '?'} — {forward_result['detail'] if forward_result else '?'}",
            })
            return self._DEFER

        if not forward_target:
            if verbose:
                print(f"[KillModule:{self.bot_name}] no target forward → stalk to approach")
            memory.add_module_entry("kill", {
                "tick": state_builder.get_tick(),
                "killable": False, "target": self._current_target or "nearest_player",
                "policy": "vlm_stalk",
                "summary": "STALK: behind clear but no target visible, approaching",
            })
            js_code = AmongUsEnvAdapter.build_kill_js(
                self.bot_name, int(self.detection_range),
                kill_on_arrival=False, teammate=self.teammate,
                target_name=self._current_target,
            )
            return {"code": js_code, "metadata": {"action": "vlm_stalk"}}

        # Behind clear + forward clear + target visible → KILL!
        if verbose:
            print(f"[KillModule:{self.bot_name}] behind clear + target forward + no witnesses → KILL!")
        memory.add_module_entry("kill", {
            "tick": state_builder.get_tick(),
            "killable": True, "target": self._current_target or "nearest",
            "policy": "vlm_final",
            "summary": f"VLM KILL: behind clear, target visible, no witnesses — {forward_result['detail']}",
        })
        state_builder._attack_ready = False
        js_code = AmongUsEnvAdapter.build_kill_js(
            self.bot_name, int(self.detection_range), True,
            teammate=self.teammate, target_name=self._current_target,
        )
        return {"code": js_code, "metadata": {"kill_on_arrival": True, "target": "nearest"}}

    def _safe_names(self, dead_players: Optional[List[str]] = None) -> set:
        names = set()
        if self.teammate:
            names.add(self.teammate.lower())
        if self._current_target:
            names.add(self._current_target.lower())
        for dp in (dead_players or []):
            names.add(dp.lower())
        return names

    def _postprocess_witnesses(self, witnesses: bool, detail: str,
                               verbose: bool,
                               dead_players: Optional[List[str]] = None) -> bool:
        """Override witnesses_visible=True when detail only mentions safe/dead players."""
        if not witnesses:
            return False
        safe = self._safe_names(dead_players)
        if not safe:
            return witnesses
        all_players = set(self._color_map.keys())
        unsafe_mentioned = any(
            p.lower() in detail.lower()
            for p in all_players
            if p.lower() not in safe
        )
        if not unsafe_mentioned:
            if verbose:
                print(f"[KillModule:{self.bot_name}] postfilter: detail only mentions "
                      f"safe/dead players → witnesses_visible overridden to False")
            return False
        return True

    def _vlm_scan_behind(self, game_state, verbose: bool,
                         dead_players: Optional[List[str]] = None) -> dict:
        """VLM evaluates behind RGB: any witnesses?"""
        human_parts = [
            {"type": "text", "text": "Direction: behind. List all players visible, classify each as SAFE or WITNESS."},
        ]
        if game_state.rgb_base64:
            human_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{game_state.rgb_base64}",
                    "detail": "high",
                },
            })

        target_name = self._current_target
        target_color = self._color_map.get(target_name) if target_name else None
        teammate_color = self._color_map.get(self.teammate) if self.teammate else None

        try:
            result = self._behind_scan_chain.invoke([
                SystemMessage(content=_render_behind_prompt(
                    self.teammate, teammate_color=teammate_color,
                    target=target_name, target_color=target_color,
                    dead_players=dead_players, color_map=self._color_map,
                )),
                HumanMessage(content=human_parts),
            ])
            witnesses = result.get("witnesses_visible", False)
            detail = result.get("detail", "")
            witnesses = self._postprocess_witnesses(witnesses, detail, verbose, dead_players)
            if verbose:
                print(f"[KillModule:{self.bot_name}] scan behind: "
                      f"witnesses={witnesses} | {detail}")
            return {"direction": "behind", "witnesses_visible": witnesses, "detail": detail}
        except Exception as e:
            if verbose:
                print(f"[KillModule:{self.bot_name}] scan behind error: {e}")
            return {"direction": "behind", "witnesses_visible": True, "detail": f"error: {e}"}

    def _vlm_scan_forward(self, game_state, verbose: bool,
                          dead_players: Optional[List[str]] = None) -> dict:
        """VLM evaluates forward RGB: target visible? witnesses?"""
        human_parts = [
            {"type": "text", "text": "Direction: forward. List all players visible, classify each as SAFE or WITNESS."},
        ]
        if game_state.rgb_base64:
            human_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{game_state.rgb_base64}",
                    "detail": "high",
                },
            })

        target_name = self._current_target
        target_color = self._color_map.get(target_name) if target_name else None
        teammate_color = self._color_map.get(self.teammate) if self.teammate else None

        try:
            result = self._forward_scan_chain.invoke([
                SystemMessage(content=_render_forward_prompt(
                    target=target_name, target_color=target_color,
                    teammate=self.teammate, teammate_color=teammate_color,
                    dead_players=dead_players, color_map=self._color_map,
                )),
                HumanMessage(content=human_parts),
            ])
            target_vis = result.get("target_visible", False)
            witnesses = result.get("witnesses_visible", False)
            detail = result.get("detail", "")
            witnesses = self._postprocess_witnesses(witnesses, detail, verbose, dead_players)
            if verbose:
                print(f"[KillModule:{self.bot_name}] scan forward: "
                      f"target={target_vis} witnesses={witnesses} | {detail}")
            return {
                "direction": "forward", "target_visible": target_vis,
                "witnesses_visible": witnesses, "detail": detail,
            }
        except Exception as e:
            if verbose:
                print(f"[KillModule:{self.bot_name}] scan forward error: {e}")
            return {
                "direction": "forward", "target_visible": False,
                "witnesses_visible": True, "detail": f"error: {e}",
            }

    @staticmethod
    def _fmt(pos) -> str:
        if pos is None:
            return "unknown"
        return f"({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})"

    @staticmethod
    def _euclid(a, b) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5

    def _text_stalk(self, target, memory, state_builder, why: str, verbose: bool) -> dict:
        """Approach the target without attacking (no-vision stalk)."""
        if verbose:
            print(f"[KillModule:{self.bot_name}] text stalk {target} ({why})")
        memory.add_module_entry("kill", {
            "tick": state_builder.get_tick(), "killable": False, "target": target,
            "policy": "text", "summary": f"TEXT-STALK {target} ({why})",
        })
        js_code = AmongUsEnvAdapter.build_kill_js(
            self.bot_name, int(self.detection_range), False,
            teammate=self.teammate, target_name=target)
        return {"code": js_code, "metadata": {"action": "text_stalk", "target": target}}

    def _text_kill(self, game_state, state_builder, memory, verbose) -> Optional[dict]:
        """No-camera kill decision: reason from player positions/distance only.

        Replaces the behind/forward VLM scan. Witnesses are judged purely by
        Euclidean distance (walls/LoS unknown, like a sightless agent).
        """
        target = self._current_target
        own = getattr(state_builder, "own_position", None)
        players = getattr(state_builder, "players", None)
        # No valid target/positions → let planner re-decide
        if (not target or players is None or own is None
                or target not in players or players[target].position is None):
            return {"code": "", "metadata": {"action": "text_no_target"}}

        tpos = players[target].position
        tdist = self._euclid(own, tpos)

        # Not attack-ready (cooldown) or not task phase → approach only
        if not state_builder.get_attack_ready() or game_state.phase != 0:
            return self._text_stalk(target, memory, state_builder, "cooldown", verbose)

        # Build the other-players list (alive, non-self, non-target)
        other_lines = []
        for name, p in players.items():
            if p.is_dead or p.position is None or name in (self.bot_name, target):
                continue
            tag = " (your teammate — SAFE)" if name == self.teammate else ""
            other_lines.append(
                f"  {name}: {self._fmt(p.position)} dist={self._euclid(own, p.position):.0f}b{tag}")
        others_text = "\n".join(other_lines) if other_lines else "  (none)"

        kill_now, witnesses, reason = False, [], ""
        if self._text_kill_chain is not None:
            try:
                prompt = _TEXT_KILL_PROMPT.format(
                    own_pos=self._fmt(own), target=target, target_pos=self._fmt(tpos),
                    target_dist=f"{tdist:.0f}", teammate=self.teammate or "(none)",
                    others=others_text)
                result = self._text_kill_chain.invoke([
                    SystemMessage(content=prompt),
                    HumanMessage(content="Decide now. Output JSON only."),
                ])
                kill_now = bool(result.get("kill_now", False))
                witnesses = result.get("witnesses", []) or []
                reason = str(result.get("reason", ""))
            except Exception as e:
                if verbose:
                    print(f"[KillModule:{self.bot_name}] text_kill error: {e} → stalk")
                return self._text_stalk(target, memory, state_builder, f"error:{e}", verbose)

        if verbose:
            print(f"[KillModule:{self.bot_name}] text_kill: kill_now={kill_now} "
                  f"tdist={tdist:.0f} witnesses={witnesses} | {reason}")

        if kill_now and tdist <= self.detection_range:
            state_builder._attack_ready = False
            memory.add_module_entry("kill", {
                "tick": state_builder.get_tick(), "killable": True, "target": target,
                "policy": "text", "summary": f"TEXT-KILL {target} (dist={tdist:.0f}) {reason[:40]}",
            })
            js_code = AmongUsEnvAdapter.build_kill_js(
                self.bot_name, int(self.detection_range), True,
                teammate=self.teammate, target_name=target)
            return {"code": js_code, "metadata": {"action": "text_kill", "target": target}}

        return self._text_stalk(target, memory, state_builder,
                                reason or ("witnesses" if witnesses else "wait"), verbose)

    def _reset_scan(self) -> None:
        self._scan_phase = 0
        self._scan_results = []
        self._original_yaw = None
        self._current_target = None

    def _rule_kill(
        self, game_state, state_builder, memory, verbose
    ) -> Optional[dict]:
        """Rule-based kill: nearest entity, no strategic judgment."""
        kill_on_arrival = state_builder.get_attack_ready() and game_state.phase == 0

        if kill_on_arrival:
            if verbose:
                print(f"[KillModule:{self.bot_name}] rule: hunt+kill")
            state_builder._attack_ready = False
        else:
            if verbose:
                print(f"[KillModule:{self.bot_name}] rule: stalk (cooldown)")

        memory.add_module_entry("kill", {
            "tick": state_builder.get_tick(),
            "killable": kill_on_arrival,
            "target": self._current_target or "nearest_player",
            "policy": "rule",
            "summary": f"{'HUNT+KILL' if kill_on_arrival else 'STALK'} (range={int(self.detection_range)})",
        })

        js_code = AmongUsEnvAdapter.build_kill_js(
            self.bot_name, int(self.detection_range), kill_on_arrival,
            teammate=self.teammate, target_name=self._current_target,
        )
        return {"code": js_code, "metadata": {"kill_on_arrival": kill_on_arrival}}
