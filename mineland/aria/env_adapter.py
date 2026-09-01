"""
AmongUsEnvAdapter — Environment-specific layer for Among Us in MineLand.

Responsibilities (RUBRIC R1):
  - Parse raw obs → clean AmongUsGameState
  - Parse raw events → structured GameEvent list
  - Provide game constants (slot numbers, item names, entity names)
  - Parse hotbar → alive player list
  - Check for nearby corpses (armor_stand detection)

Agent code should NEVER import raw obs field names or game constants directly.
Everything goes through this adapter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────
# Game Constants (Among Us MineLand-specific)
# ──────────────────────────────────────────────────────────────────────

REPORT_ITEM_SLOT = 8
REPORT_ITEM_NAME = "carrot_on_a_stick"
CORPSE_ENTITY_NAME = "armor_stand"
CORPSE_SEARCH_RADIUS = 10
REPORT_APPROACH_DIST = 2
VOTE_SKIP_SLOT = 8
EMERGENCY_BUTTON_POS = (87, 66, 76)  # stone_button location for emergency meeting

DEFAULT_PLAYER_ID_MAP = {
    "James": 1, "Steve": 2, "Jason": 3, "Michael": 4,
    "human1": 6, "human2": 7,
}

VOTE_REVEAL_RE = re.compile(r"^(\w+) was (?:a|an) (Crewmate|Imposter)\.$")
VOTE_SKIPPED_TEXT = "No one was ejected (SKIPPED)."


# ──────────────────────────────────────────────────────────────────────
# Data classes for clean state representation
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PlayerEntity:
    """A player entity observed in the current step's target_entities."""
    name: str
    position: Tuple[float, float, float]
    distance: Optional[float] = None
    in_sight: bool = True  # LOS + frustum check from observation_utils.js


@dataclass
class CorpseEntity:
    """An armor_stand (corpse) observed in the current step."""
    position: Tuple[float, float, float]
    distance: Optional[float] = None


@dataclass
class GameEvent:
    """A structured game event parsed from raw obs events."""
    type: str           # "meeting_start", "player_died", "vote_result",
                        # "ejected", "chat", "mission_completed"
    message: str        # original message text
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AmongUsGameState:
    """Clean game state parsed from raw observation.

    Agent code should work with this, not raw obs.
    """
    # Game state (from scoreboard — always observable)
    tick: int = 0
    phase: int = 0               # 0=explore, 1=meeting
    is_ghost: bool = False
    attack_ready: bool = False
    vote_time: bool = False      # meeting_1min == 1
    can_talk: bool = False       # talk == 1
    doing_mission: bool = False

    # Self
    own_position: Optional[Tuple[float, float, float]] = None

    # Entities visible this step (proximity-based)
    visible_players: List[PlayerEntity] = field(default_factory=list)
    visible_corpses: List[CorpseEntity] = field(default_factory=list)

    # Events this step (already parsed)
    events: List[GameEvent] = field(default_factory=list)

    # Missions (from scoreboard)
    mission_status: Dict[str, Optional[int]] = field(default_factory=dict)

    # Hotbar-based alive players (for voting)
    alive_votable_players: List[str] = field(default_factory=list)

    # Visual
    rgb_base64: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────
# Adapter
# ──────────────────────────────────────────────────────────────────────

class AmongUsEnvAdapter:
    """Translates between raw MineLand obs and clean AmongUsGameState."""

    def __init__(
        self,
        bot_name: str,
        player_id_map: Optional[dict] = None,
    ):
        self.bot_name = bot_name
        self.player_id_map = player_id_map or dict(DEFAULT_PLAYER_ID_MAP)
        self._id_to_name = {v: k for k, v in self.player_id_map.items()}

    # ── Main entry point ────────────────────────────────────────────

    def parse_obs(self, obs) -> AmongUsGameState:
        """Convert raw obs into a clean AmongUsGameState."""
        state = AmongUsGameState()

        state.tick = self._to_int(self._get(obs, "tick", 0))
        state.phase = self._to_int(self._get(obs, "phase", 0))
        state.is_ghost = self._to_bool(self._get(obs, "ghost", 0))
        state.attack_ready = self._to_bool(
            self._get(obs, "atk_ready", None) or self._get(obs, "attack_ready", 0)
        )
        state.vote_time = self._to_int(self._get(obs, "meeting_1min", 0)) == 1
        state.can_talk = self._to_int(self._get(obs, "talk", 0)) == 1
        state.doing_mission = self._to_bool(self._get(obs, "doing_mission", 0))

        # Own position — location_stats.pos is [x, y, z]
        loc = self._get(obs, "location_stats", None) or {}
        if loc:
            pos = self._get(loc, "pos", None)
            if pos and isinstance(pos, (list, tuple)) and len(pos) >= 3:
                state.own_position = (float(pos[0]), float(pos[1]), float(pos[2]))

        # Target entities → players + corpses
        # target_entities carry per-entity `in_sight` (LOS + frustum) from
        # observation_utils.js; egocentric state builder uses it to drop
        # entities the bot cannot visually see.
        target_entities = self._get(obs, "target_entities", []) or []
        for ent in target_entities:
            name = self._get(ent, "name", None)
            if not name:
                continue
            ent_pos = self._get(ent, "position", None)
            if ent_pos is not None:
                ex = float(self._get(ent_pos, "x", 0))
                ey = float(self._get(ent_pos, "y", 0))
                ez = float(self._get(ent_pos, "z", 0))
            else:
                ex = float(self._get(ent, "x", 0))
                ey = float(self._get(ent, "y", 0))
                ez = float(self._get(ent, "z", 0))
            pos = (ex, ey, ez)
            dist = self._get(ent, "distance", None)
            in_sight = bool(self._get(ent, "in_sight", True))

            if name == CORPSE_ENTITY_NAME:
                state.visible_corpses.append(CorpseEntity(position=pos, distance=dist))
            elif name != self.bot_name:
                state.visible_players.append(
                    PlayerEntity(name=name, position=pos, distance=dist,
                                 in_sight=in_sight)
                )

        # Events
        raw_events = self._get(obs, "event", []) or []
        state.events = self._parse_events(raw_events)

        # Missions
        for i in range(1, 21):
            key = f"mission{i}"
            val = self._get(obs, key, None)
            if val is not None:
                try:
                    state.mission_status[key] = int(val)
                except (ValueError, TypeError):
                    state.mission_status[key] = None

        # Hotbar → alive votable players
        state.alive_votable_players = self._parse_hotbar(obs)

        # RGB
        state.rgb_base64 = getattr(obs, "rgb_base64", None)

        return state

    # ── Event parsing ───────────────────────────────────────────────

    def _parse_events(self, raw_events: list) -> List[GameEvent]:
        """Parse raw event list into structured GameEvent objects."""
        events: List[GameEvent] = []
        for event in raw_events:
            etype = self._get(event, "type", None)
            msg = self._get(event, "message", "") or ""
            if etype not in ("chat", "message") or not msg:
                continue

            # (SERVER) messages. meeting_start is parsed here (broadcast text
            # is reliably captured for these triggers — confirmed across runs).
            # Final vote_result comes from Minecraft tellraw chat. The Python
            # orchestration tally is only an intent log and is not authoritative.
            if msg.startswith("(SERVER)"):
                low = msg.lower()
                if "meeting start!" in low:
                    # Emergency button: "(SERVER) EMERGENCY MEETING START!"
                    events.append(GameEvent(
                        type="meeting_start", message=msg,
                        data={"trigger": "emergency",
                              "reporter": None, "victim": None},
                    ))
                elif "reported" in low and "body" in low:
                    # Body report: "(SERVER) <reporter> reported <victim>'s body!"
                    reporter, victim = self._parse_report_msg(msg)
                    events.append(GameEvent(
                        type="meeting_start", message=msg,
                        data={"trigger": "report",
                              "reporter": reporter, "victim": victim},
                    ))
                elif "[DEAD]:" in msg:
                    dead_part = msg[msg.index("[DEAD]:") + len("[DEAD]:"):].strip()
                    if dead_part and dead_part.upper() != "NONE":
                        events.append(GameEvent(
                            type="player_died", message=msg,
                            data={"player_name": dead_part},
                        ))
                else:
                    events.append(GameEvent(type="server_message", message=msg))
                # Always append as chat so agents have textual visibility.
                events.append(GameEvent(type="chat", message=msg))
                continue

            if msg == VOTE_SKIPPED_TEXT:
                events.append(GameEvent(
                    type="vote_result", message=msg,
                    data={
                        "ejected": None,
                        "ejected_role": None,
                        "skipped": True,
                        "source": "server_tellraw",
                        "authoritative": True,
                    },
                ))
                events.append(GameEvent(type="chat", message=msg))
                continue

            if m := VOTE_REVEAL_RE.match(msg):
                events.append(GameEvent(
                    type="vote_result", message=msg,
                    data={
                        "ejected": m.group(1),
                        "ejected_role": m.group(2).lower(),
                        "skipped": False,
                        "source": "server_tellraw",
                        "authoritative": True,
                    },
                ))
                events.append(GameEvent(type="chat", message=msg))
                continue

            # Regular chat
            if "has died" in msg.lower() or "was killed" in msg.lower():
                events.append(GameEvent(
                    type="player_died", message=msg,
                    data={"player_name": self._extract_death_name(msg)},
                ))
            elif re.search(r"mission\s*\d+", msg.lower()) and any(
                t in msg.lower() for t in ("completed", "complete", "success")
            ):
                m = re.search(r"mission\s*(\d{1,2})", msg.lower())
                if m:
                    events.append(GameEvent(
                        type="mission_completed", message=msg,
                        data={"mission_key": f"mission{m.group(1)}"},
                    ))

            events.append(GameEvent(type="chat", message=msg))

        return events

    # ── Hotbar parsing ──────────────────────────────────────────────

    def _parse_hotbar(self, obs) -> List[str]:
        """Parse hotbar slots 36-44 to determine alive votable players."""
        inv = getattr(obs, "inventory_all", None)
        if inv is None and hasattr(obs, "get"):
            inv = obs.get("inventory_all")
        if not inv:
            return []

        alive = []
        for slot_offset in range(8):
            slot_key = 36 + slot_offset
            item = inv.get(slot_key) or inv.get(str(slot_key))
            if item and item.get("name") == REPORT_ITEM_NAME:
                player_id = slot_offset + 1
                name = self._id_to_name.get(player_id)
                if name and name != self.bot_name:
                    alive.append(name)
        return alive

    # ── Corpse detection ────────────────────────────────────────────

    def has_nearby_corpse(self, game_state: AmongUsGameState) -> bool:
        """Check if any corpse is within CORPSE_SEARCH_RADIUS."""
        for corpse in game_state.visible_corpses:
            if corpse.distance is not None and corpse.distance < CORPSE_SEARCH_RADIUS:
                return True
            if corpse.distance is None and game_state.own_position:
                dx = corpse.position[0] - game_state.own_position[0]
                dy = corpse.position[1] - game_state.own_position[1]
                dz = corpse.position[2] - game_state.own_position[2]
                if (dx * dx + dy * dy + dz * dz) ** 0.5 < CORPSE_SEARCH_RADIUS:
                    return True
        return False

    # ── JS post-processing ────────────────────────────────────

    @staticmethod
    def strip_chat_from_js(js_code: str) -> str:
        """Remove bot.chat() calls from generated JS code.

        Used during phase=0 (explore) to prevent LLM-generated code from
        producing unintended chat messages. The codegen LLM sometimes
        embeds bot.chat() for progress/status, which violates the rule
        that agents should only speak during meetings.
        """
        return re.sub(
            r"""bot\.chat\s*\([^)]*\)\s*;?""",
            "/* chat stripped (phase=0) */",
            js_code,
        )

    # ── JS action builders ──────────────────────────────────────

    @staticmethod
    def build_emergency_meeting_js() -> str:
        """Navigate to the emergency meeting button and press it."""
        x, y, z = EMERGENCY_BUTTON_POS
        return (
            f"async function callEmergencyMeeting(bot) {{\n"
            f"  await bot.pathfinder.goto(\n"
            f"    new GoalNear({x}, {y}, {z}, 2)\n"
            f"  );\n"
            f"  const button = bot.findBlock({{\n"
            f"    matching: block => block.name === 'stone_button',\n"
            f"    maxDistance: 5,\n"
            f"  }});\n"
            f"  if (button) {{\n"
            f"    bot.activateBlock(bot.blockAt(button.position));\n"
            f"  }}\n"
            f"  await bot.waitForTicks(5);\n"
            f"}}\n"
            f"await callEmergencyMeeting(bot);"
        )

    @staticmethod
    def build_interrupt_js() -> str:
        return (
            "async function interruptForMeeting(bot) {\n"
            "  if (bot.pathfinder) bot.pathfinder.setGoal(null);\n"
            "  if (bot.clearControlStates) bot.clearControlStates();\n"
            "}\n"
            "await interruptForMeeting(bot);"
        )

    @staticmethod
    def build_chat_js(text: str) -> str:
        safe = text.replace("'", "\\'")
        return (
            f"async function talkInMeeting(bot) {{\n"
            f"  bot.chat('{safe}');\n"
            f"}}\nawait talkInMeeting(bot);"
        )

    @staticmethod
    def build_vote_js(player_name: str, player_id_map: dict) -> str:
        if player_name == "skip":
            slot = VOTE_SKIP_SLOT
        else:
            pid = player_id_map.get(player_name)
            slot = (pid - 1) if pid is not None else VOTE_SKIP_SLOT
        return (
            f"async function castVote(bot) {{\n"
            f"  bot.setQuickBarSlot({slot});\n"
            f"  await bot.waitForTicks(2);\n"
            f"  bot.activateItem();\n"
            f"  await bot.waitForTicks(5);\n"
            f"}}\nawait castVote(bot);"
        )

    @staticmethod
    def build_report_js() -> str:
        return (
            f"async function reportCorpse(bot) {{\n"
            f"  const corpse = bot.nearestEntity(\n"
            f"    e => e.name === '{CORPSE_ENTITY_NAME}'\n"
            f"      && e.position.distanceTo(bot.entity.position) < {CORPSE_SEARCH_RADIUS}\n"
            f"  );\n"
            f"  if (!corpse) return;\n"
            f"  const dist = corpse.position.distanceTo(bot.entity.position);\n"
            f"  if (dist > {REPORT_APPROACH_DIST}) {{\n"
            f"    await bot.pathfinder.goto(\n"
            f"      new GoalNear(corpse.position.x, corpse.position.y, corpse.position.z, {REPORT_APPROACH_DIST})\n"
            f"    );\n"
            f"  }}\n"
            f"  bot.setQuickBarSlot({REPORT_ITEM_SLOT});\n"
            f"  await bot.waitForTicks(2);\n"
            f"  bot.activateItem();\n"
            f"  await bot.waitForTicks(5);\n"
            f"}}\n"
            f"await reportCorpse(bot);"
        )

    @staticmethod
    def build_kill_js(
        bot_name: str,
        detection_range: int,
        kill_on_arrival: bool,
        teammate: Optional[str] = None,
        target_name: Optional[str] = None,
    ) -> str:
        """Generate stalk/hunt-and-kill JS.

        target_name: when provided, the bot prefers this exact player as the
        stalk/kill target (via bot.players[name].entity). Falls back to
        bot.nearestEntity within detection_range when the named target isn't
        currently visible. None → original nearest-only behavior.
        """
        safe_name = bot_name.replace("'", "\\'")
        teammate_filter = ""
        if teammate:
            safe_tm = teammate.replace("'", "\\'")
            teammate_filter = f"\n      && e.username !== '{safe_tm}'"

        # Named-target lookup that prefers the LLM-specified victim regardless
        # of distance — privileged state already exposes the position, and
        # stalking from far away is the intended behavior. Falls back to
        # nearestEntity (within detection_range) when the named player isn't
        # currently in bot.players. Either branch returns a Mineflayer entity
        # (or null) so downstream code is uniform.
        if target_name:
            safe_target = target_name.replace("'", "\\'")
            find_target_js = (
                f"  const findTarget = () => {{\n"
                f"    const named = bot.players['{safe_target}'];\n"
                f"    if (named && named.entity && named.entity.position) {{\n"
                f"      return named.entity;\n"
                f"    }}\n"
                f"    return bot.nearestEntity(\n"
                f"      e => e.type === 'player'\n"
                f"        && e.username !== '{safe_name}'{teammate_filter}\n"
                f"        && e.position.distanceTo(bot.entity.position) < {detection_range}\n"
                f"    );\n"
                f"  }};\n"
            )
        else:
            find_target_js = (
                f"  const findTarget = () => bot.nearestEntity(\n"
                f"    e => e.type === 'player'\n"
                f"      && e.username !== '{safe_name}'{teammate_filter}\n"
                f"      && e.position.distanceTo(bot.entity.position) < {detection_range}\n"
                f"  );\n"
            )

        if kill_on_arrival:
            narration_target = target_name or "the nearest non-teammate player"
            narration = f"I am approaching {narration_target} to kill them"
            return (
                f"console.log('[ARIA_SAY:' + bot.username + '] {narration}');\n"
                f"async function huntAndKill(bot) {{\n"
                f"{find_target_js}"
                f"  let target = findTarget();\n"
                f"  if (!target) return;\n"
                f"  await bot.pathfinder.goto(\n"
                f"    new GoalNear(target.position.x, target.position.y, target.position.z, 2)\n"
                f"  );\n"
                f"  target = findTarget();\n"
                f"  if (target && bot.entity.position.distanceTo(target.position) < 5) {{\n"
                f"    bot.attack(target);\n"
                f"    bot.attack(target);\n"
                f"  }}\n"
                f"}}\n"
                f"await huntAndKill(bot);"
            )
        else:
            narration_target = target_name or "the nearest non-teammate player"
            narration = f"I am stalking {narration_target} to set up a kill"
            return (
                f"console.log('[ARIA_SAY:' + bot.username + '] {narration}');\n"
                f"async function stalkPlayer(bot) {{\n"
                f"{find_target_js}"
                f"  const target = findTarget();\n"
                f"  if (target) {{\n"
                f"    await bot.pathfinder.goto(\n"
                f"      new GoalNear(target.position.x, target.position.y, target.position.z, 3)\n"
                f"    );\n"
                f"    await bot.waitForTicks(5);\n"
                f"  }}\n"
                f"}}\n"
                f"await stalkPlayer(bot);"
            )

    @staticmethod
    def build_flee_js(
        own_position: Tuple[float, float, float],
        role: Optional[str] = None,
    ) -> str:
        import random
        ox, _, oz = own_position
        dx = random.choice([-1, 1]) * random.randint(15, 20)
        dz = random.choice([-1, 1]) * random.randint(15, 20)
        fx, fz = int(ox) + dx, int(oz) + dz
        prefix = ""
        if role == "imposter":
            prefix = (
                "console.log('[ARIA_SAY:' + bot.username + "
                "'] I am fleeing the scene after the kill to avoid being the one to report the body');\n"
            )
        return (
            f"{prefix}"
            "async function fleeFromCorpse(bot) {\n"
            "  try {\n"
            f"    await bot.pathfinder.goto(new GoalXZ({fx}, {fz}));\n"
            "  } catch (e) {\n"
            "    const msg = String((e && e.message) || e || '');\n"
            "    if (!(e && e.name === 'NoPath') && !msg.includes('No path to the goal')) throw e;\n"
            "    if (bot.pathfinder) bot.pathfinder.setGoal(null);\n"
            "  }\n"
            "}\n"
            "await fleeFromCorpse(bot);"
        )

    @staticmethod
    def build_wander_js(
        own_position: Tuple[float, float, float],
        role: Optional[str] = None,
    ) -> str:
        import random
        ox, _, oz = own_position
        rx = int(ox) + random.randint(-12, 12)
        rz = int(oz) + random.randint(-12, 12)
        prefix = ""
        if role == "imposter":
            prefix = (
                "console.log('[ARIA_SAY:' + bot.username + "
                "'] I am wandering aimlessly to blend in while I wait for a kill opportunity');\n"
            )
        return (
            f"{prefix}"
            "async function randomWander(bot) {\n"
            "  try {\n"
            f"    await bot.pathfinder.goto(new GoalXZ({rx}, {rz}));\n"
            "  } catch (e) {\n"
            "    const msg = String((e && e.message) || e || '');\n"
            "    if (!(e && e.name === 'NoPath') && !msg.includes('No path to the goal')) throw e;\n"
            "    if (bot.pathfinder) bot.pathfinder.setGoal(null);\n"
            "  }\n"
            "}\n"
            "await randomWander(bot);"
        )

    # ── Internal helpers ────────────────────────────────────────────

    def _parse_ejected_name(self, msg: str) -> Optional[str]:
        all_names = list(self.player_id_map.keys())
        for name in all_names:
            if name.lower() in msg.lower() and "ejected" in msg.lower():
                return name
        return None

    def _parse_report_msg(self, msg: str) -> tuple:
        """Extract (reporter, victim) from '(SERVER) X reported Y's body!'.

        Returns (None, None) if either name can't be matched against
        player_id_map. Falls back to None individually so partial info
        is still useful.
        """
        all_names = list(self.player_id_map.keys())
        idx = msg.lower().find("reported")
        if idx < 0:
            return (None, None)
        before, after = msg[:idx], msg[idx + len("reported"):]
        reporter = next((n for n in all_names if n.lower() in before.lower()), None)
        victim = next((n for n in all_names if n.lower() in after.lower()), None)
        return (reporter, victim)

    def _extract_death_name(self, msg: str) -> Optional[str]:
        all_names = list(self.player_id_map.keys())
        for name in all_names:
            if name.lower() in msg.lower():
                return name
        return None

    @staticmethod
    def _get(obj, key, default):
        if obj is None:
            return default
        if hasattr(obj, "get"):
            val = obj.get(key, default)
        else:
            val = getattr(obj, key, default)
        return default if val is None else val

    @staticmethod
    def _to_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "t", "yes", "y", "on")
        return bool(value)

    @staticmethod
    def _to_int(value, default: int = 0) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
