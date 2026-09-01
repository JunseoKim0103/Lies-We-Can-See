"""Base class for Aria state builders."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from ..env_adapter import AmongUsGameState, GameEvent


class BaseStateBuilder(ABC):
    """Shared state tracking for all state modes.

    Scoreboard values, chat, events, missions, suspicion are common to both
    ego and privileged since they come from observable game state.
    """

    def __init__(
        self,
        bot_name: str,
        role: str,
        teammate: Optional[str] = None,
        show_mission_progress: bool = True,
    ):
        self.bot_name = bot_name
        self.role = role
        self.teammate = teammate
        self.show_mission_progress = show_mission_progress

        # Scoreboard-based (always observable)
        self.tick: int = 0
        self.phase: int = 0
        self._attack_ready: bool = False
        self.vote_time: bool = False
        self.can_talk: bool = False
        self.is_ghost: bool = False
        self.own_position: Optional[Tuple[float, float, float]] = None

        # Missions
        self.mission_status: Dict[str, Optional[int]] = {}
        self.mission_recently_completed: List[str] = []

        # Chat / meetings
        self.chat_log: List[str] = []
        self.meeting_start_index: int = -1
        self.last_meeting_chat: List[str] = []
        self.meeting_summaries: List[dict] = []
        self._meeting_count: int = 0
        self._last_vote_result_meeting: int = 0
        # Reporter/victim for the *current* meeting (cleared on vote_result).
        # None on emergency-button meetings.
        self.current_meeting_reporter: Optional[str] = None
        self.current_meeting_victim: Optional[str] = None
        self.current_meeting_new_dead: List[str] = []

        # Suspicion (agent's internal belief)
        self.suspicion: Dict[str, float] = {}

        # Speech tracking
        self.speech_counts: Dict[str, int] = {}
        self._force_all_spoke: bool = False

        # Death tracking (from events — sent at meeting start, not privileged)
        self._dead_players: set = set()

        # Global mission progress (imposter view — injected by orchestration layer).
        # Each agent only sees its own mission_status, so a correct global count
        # requires aggregating pending lists across all crewmates externally.
        self._global_mission_remaining: Optional[int] = None
        self._global_initial_crew: Optional[int] = None

        # Current plan from planner (set per step by Aria._dispatch).
        # Carries short_term_plan + long_term_plan into module prompts via
        # summarize_for_context. Lets planning_mode actually influence module
        # behavior; otherwise reactive/shortterm/hierarchical are cosmetic.
        self._current_plan = None

    # ── Update from game state ──────────────────────────────────────

    def update(self, game_state: AmongUsGameState) -> None:
        """Update common state from parsed observation."""
        self.tick = game_state.tick
        self.phase = game_state.phase
        self._attack_ready = game_state.attack_ready
        self.vote_time = game_state.vote_time
        self.can_talk = game_state.can_talk
        self.is_ghost = game_state.is_ghost
        self.own_position = game_state.own_position

        # Missions
        prev_status = dict(self.mission_status)
        self.mission_recently_completed = []
        for key, val in game_state.mission_status.items():
            self.mission_status[key] = val
            if prev_status.get(key) != 1 and val == 1:
                self.mission_recently_completed.append(key)

        # Process events
        for event in game_state.events:
            self._process_event(event)

        # Subclass-specific update
        self._update_players(game_state)

    def _process_event(self, event: GameEvent) -> None:
        """Process structured game events."""
        if event.type == "meeting_start":
            self._meeting_count += 1
            self.chat_log.append(event.message)
            self.meeting_start_index = len(self.chat_log)
            self.speech_counts = {}
            data = event.data or {}
            reporter = data.get("reporter")
            victim = data.get("victim")
            new_dead = list(data.get("new_dead_since_last_meeting") or [])
            if not new_dead and self.current_meeting_new_dead:
                # The orchestration layer may patch the full death list before
                # the report broadcast is later consumed by this StateBuilder.
                # Do not let a duplicate meeting_start with an empty payload
                # erase that richer meeting context.
                new_dead = list(self.current_meeting_new_dead)
            self.current_meeting_reporter = reporter
            self.current_meeting_victim = victim
            self.current_meeting_new_dead = new_dead
            for name in new_dead:
                self._dead_players.add(name)
            trig = data.get("trigger") or "unknown"
            if trig == "report":
                tag = f"reporter={reporter} victim={victim}"
            elif trig == "emergency":
                tag = "emergency button"
            else:
                tag = f"reporter={reporter} victim={victim}"
            print(f"[StateBuilder:{self.bot_name}] meeting_start received "
                  f"(meeting #{self._meeting_count}, tick={self.tick}, "
                  f"trigger={trig}): {tag}; new_dead={new_dead}")

        elif event.type == "player_died":
            name = event.data.get("player_name")
            if name:
                self._dead_players.add(name)

        elif event.type == "vote_result":
            ejected = event.data.get("ejected")
            ejected_role = event.data.get("ejected_role")
            skipped = bool(event.data.get("skipped", False))
            source = event.data.get("source") or "unknown"
            authoritative = bool(event.data.get("authoritative")) or source == "server_tellraw"
            existing_idx = self._meeting_summary_index(self._meeting_count)
            if existing_idx is not None:
                existing = self.meeting_summaries[existing_idx]
                same_result = (
                    bool(existing.get("skipped", False)) == skipped
                    and existing.get("ejected") == ejected
                    and existing.get("ejected_role") == ejected_role
                )
                existing_source = existing.get("source") or "unknown"
                existing_authoritative = (
                    bool(existing.get("authoritative"))
                    or existing_source == "server_tellraw"
                )
                if same_result:
                    if authoritative and not existing_authoritative:
                        existing["source"] = source
                        existing["authoritative"] = True
                        existing["ejection_msg"] = event.message
                    self._finalize_vote_result(ejected)
                    return
                if authoritative and not existing_authoritative:
                    old_ejected = existing.get("ejected")
                    old_tag = self._format_vote_result_tag(
                        old_ejected,
                        existing.get("ejected_role"),
                        bool(existing.get("skipped", False)),
                    )
                    reporter = self.current_meeting_reporter or existing.get("reporter")
                    victim = self.current_meeting_victim or existing.get("victim")
                    existing.update({
                        "tick": self.tick,
                        "ejected": ejected,
                        "ejected_role": ejected_role,
                        "skipped": skipped,
                        "reporter": reporter,
                        "victim": victim,
                        "source": source,
                        "authoritative": True,
                        "ejection_msg": event.message,
                    })
                    if old_ejected and old_ejected != ejected:
                        self._dead_players.discard(old_ejected)
                    self._finalize_vote_result(ejected)
                    tag = self._format_vote_result_tag(ejected, ejected_role, skipped)
                    print(f"[StateBuilder:{self.bot_name}] vote_result override "
                          f"(meeting #{self._meeting_count}, tick={self.tick}, "
                          f"source={source}): {old_tag} -> {tag}")
                    return
                return
            self._save_meeting_summary(event.message, ejected=ejected,
                                       ejected_role=ejected_role,
                                       skipped=skipped,
                                       reporter=self.current_meeting_reporter,
                                       victim=self.current_meeting_victim,
                                       source=source,
                                       authoritative=authoritative)
            self._last_vote_result_meeting = self._meeting_count
            self._finalize_vote_result(ejected)
            tag = self._format_vote_result_tag(ejected, ejected_role, skipped)
            print(f"[StateBuilder:{self.bot_name}] vote_result received "
                  f"(meeting #{self._meeting_count}, tick={self.tick}, "
                  f"source={source}): {tag}")

        elif event.type == "chat":
            self.chat_log.append(event.message)

        # Trim chat — preserve meeting_start_index relative to current meeting
        if len(self.chat_log) > 200:
            overflow = len(self.chat_log) - 200
            new_start = self.meeting_start_index - overflow
            self.chat_log = self.chat_log[-200:]
            # If the meeting start was truncated away, pin to start of buffer
            self.meeting_start_index = max(0, new_start)

    @abstractmethod
    def _update_players(self, game_state: AmongUsGameState) -> None:
        """Subclass-specific player state update."""

    # ── Context generation ──────────────────────────────────────────

    @abstractmethod
    def summarize_for_context(self, module_name: str, recent_chat_k: int = 10) -> str:
        """Generate text context for LLM prompts."""

    def _common_world_lines(self) -> List[str]:
        """Shared world-state lines for both ego and privileged."""
        phase_name = {0: "Explore", 1: "Meeting"}.get(self.phase, f"Unknown({self.phase})")
        lines = [
            f"=== World State [{self.bot_name} / {self.role}] ===",
            f"Tick={self.tick} | Phase={phase_name} | AttackReady={self._attack_ready}",
            f"Own position: {self._fmt_pos_with_room(self.own_position)}",
        ]
        # Imposter teammate reminder — repeated every step so the LLM cannot
        # forget who the fellow imposter is. gpt-4.1-mini was observed to
        # treat the teammate as a crewmate kill target without this.
        if self.role == "imposter" and self.teammate:
            lines.append(
                f"Teammate reminder: your fellow imposter is {self.teammate}. "
                f"{self.teammate} is not a crewmate. Default behavior is to treat them as an ally."
            )
        if self._dead_players:
            lines.append("Dead players: " + ", ".join(sorted(self._dead_players)))
        # Surface the current meeting's report context so the LLM does not
        # have to infer who triggered the meeting from chat scraps.
        if self.phase == 1 and (self.current_meeting_reporter or self.current_meeting_victim):
            rpt = self.current_meeting_reporter or "?"
            vic = self.current_meeting_victim or "?"
            lines.append(f"Current meeting: {rpt} reported {vic}'s body")
        if self.phase == 1 and self.current_meeting_new_dead:
            lines.append(
                "Deaths since previous meeting: "
                + ", ".join(sorted(self.current_meeting_new_dead))
            )
        if self.suspicion:
            lines.append("Suspicion scores:")
            for name, score in sorted(self.suspicion.items(), key=lambda x: -x[1]):
                lines.append(f"  {name}: {score:.2f}")
        pending = self.pending_missions()
        done = self.done_missions()
        if self.mission_status:
            # Self-only counter — explicit "Your" prefix so crewmates don't
            # confuse this with the team-wide progress line below.
            lines.append(f"Your missions: done={len(done)} pending={len(pending)}")
        if self.mission_recently_completed:
            lines.append("Recently completed: " + ", ".join(self.mission_recently_completed))

        # Team-wide mission progress toward crewmate mission-win condition.
        # Uses globally-aggregated values injected by the orchestration layer —
        # total is frozen at initial_crew × 3 (doesn't shrink when crewmates die,
        # matching the datapack rule where ghosts still auto-complete missions).
        # show_mission_progress=True  → exact "done/total" (game-information cheat)
        # show_mission_progress=False → coarse early/mid/late bucket (realistic, mirrors Among Us bar)
        # Shown to both roles so crewmates can pace themselves like the
        # in-game task bar; the wording is explicit ("Team total / All crew")
        # to avoid clashing with the per-agent "Your missions" line above.
        if (self._global_mission_remaining is not None
                and self._global_initial_crew is not None):
            MISSIONS_PER_CREW = 3
            total = self._global_initial_crew * MISSIONS_PER_CREW
            remaining = max(0, self._global_mission_remaining)
            done_count = max(0, total - remaining)

            if self.show_mission_progress:
                lines.append(
                    f"Team total mission progress: {done_count}/{total} "
                    f"(all {self._global_initial_crew} crewmates × {MISSIONS_PER_CREW} each)"
                )
                if remaining <= 2:
                    lines.append(f"Warning: only {remaining} missions left until crew wins.")
            else:
                # Coarse bucket — what a real Among Us imposter sees (a progress bar).
                ratio = done_count / total if total > 0 else 0.0
                if ratio < 1 / 3:
                    bucket = "early"
                elif ratio < 2 / 3:
                    bucket = "mid"
                else:
                    bucket = "late"
                lines.append(f"Crew mission progress: {bucket} stage")
                if bucket == "late":
                    lines.append("Warning: crew is close to winning.")

        # Imposter: flag nearby alive players when attack is ready. The LLM
        # picks the target freely; if too far / witnessed, KillModule DEFERs
        # and the reason is surfaced back via "Recent kill DEFERs" section
        # so the LLM can adapt. Teammate appears in the list with a
        # "(teammate)" tag so team-kill remains an explicit option for the
        # LLM (rare deception move; default ally behavior comes from
        # personal_message).
        if self.role == "imposter" and self._attack_ready:
            nearby = self._nearby_crewmates()
            if nearby:
                lines.append("Kill opportunity: attack_ready=True, visible alive players with distance:")
                for name, dist_str in nearby:
                    lines.append(f"  {name} {dist_str}")
            else:
                lines.append("attack_ready=True but no players visible. Explore to find a target.")

        # Current plan from planner — set per step by Aria._dispatch.
        if self._current_plan is not None:
            plan = self._current_plan
            long_term = getattr(plan, "long_term_plan", None)
            short_term = getattr(plan, "short_term_plan", "") or ""
            reasoning = getattr(plan, "reasoning", "") or ""
            mode = getattr(plan, "mode", "") or ""
            if long_term:
                lines.append(f"[Long-term strategy] {long_term}")
            if short_term:
                lines.append(f"[Current plan / mode={mode}] {short_term}")
            elif reasoning:
                lines.append(f"[Current plan / mode={mode}] {reasoning}")

        return lines

    def _nearby_crewmates(self) -> List[tuple]:
        """Return [(name, dist_str), ...] for visible non-teammate alive players.

        Subclasses override to provide actual data.
        """
        return []

    def _chat_context_lines(self, module_name: str, k: int) -> List[str]:
        """Recent chat lines for context."""
        if module_name in ("meeting", "vote"):
            source = self.meeting_log if self.meeting_start_index >= 0 else self.chat_log
        else:
            source = self.chat_log
        lines = []
        if source:
            recent = source[-k:]
            lines.append(f"\n[Recent chat — last {len(recent)}]")
            for m in recent:
                lines.append(f"  > {m}")
        if module_name in ("meeting", "vote", "reflection", "planning") and self.meeting_summaries:
            lines.append(f"\n[Past Meeting Summaries]")
            for s in self.meeting_summaries:
                rpt = s.get("reporter")
                vic = s.get("victim")
                trigger_tag = (f" [reporter={rpt}, victim={vic}]"
                               if rpt or vic else "")
                if s.get("skipped"):
                    lines.append(f"  Meeting {s['meeting_no']}: skipped (no ejection){trigger_tag}")
                    continue
                ejected = s.get("ejected") or "?"
                role = s.get("ejected_role")
                role_tag = f" ({role})" if role else ""
                lines.append(f"  Meeting {s['meeting_no']}: ejected={ejected}{role_tag}{trigger_tag}")
        return lines

    def _authoritative_vote_outcome_lines(self) -> List[str]:
        """Structured final vote outcomes for reflection prompts.

        Meeting reflections should not infer the final outcome from discussion
        text like "I will skip" or "I'll vote X"; this block is the canonical
        post-vote result parsed from orchestration or server tellraw.
        """
        if not self.meeting_summaries:
            return []

        lines = [
            "[AUTHORITATIVE_VOTE_OUTCOME]",
            "Use this block as the final vote outcome. Do not infer the final outcome from meeting discussion text.",
        ]
        for s in self.meeting_summaries[-3:]:
            meeting_no = s.get("meeting_no", "?")
            reporter = s.get("reporter") or "none"
            victim = s.get("victim") or "none"
            if s.get("skipped"):
                lines.extend([
                    f"meeting_id: {meeting_no}",
                    "result_type: skipped",
                    "skipped: true",
                    "ejected_player: none",
                    "ejected_role: none",
                    f"reporter: {reporter}",
                    f"victim: {victim}",
                ])
                continue

            lines.extend([
                f"meeting_id: {meeting_no}",
                "result_type: ejection",
                "skipped: false",
                f"ejected_player: {s.get('ejected') or 'unknown'}",
                f"ejected_role: {s.get('ejected_role') or 'unknown'}",
                f"reporter: {reporter}",
                f"victim: {victim}",
            ])
        return lines

    # ── Accessors ───────────────────────────────────────────────────

    @property
    def meeting_log(self) -> List[str]:
        if self.meeting_start_index < 0:
            return []
        return self.chat_log[self.meeting_start_index:]

    def get_tick(self) -> int:
        return self.tick

    def get_phase(self) -> int:
        return self.phase

    def get_attack_ready(self) -> bool:
        return self._attack_ready

    def is_player_dead(self, name: str) -> bool:
        return name in self._dead_players

    def get_dead_player_names(self) -> List[str]:
        return sorted(self._dead_players)

    def alive_player_names(self) -> List[str]:
        """Must be overridden by subclasses with player tracking."""
        return []

    def pending_missions(self) -> List[str]:
        return [f"mission{i}" for i in range(1, 21)
                if self.mission_status.get(f"mission{i}") not in (None, 1)
                and self.mission_status.get(f"mission{i}") is not None]

    def done_missions(self) -> List[str]:
        return [f"mission{i}" for i in range(1, 21)
                if self.mission_status.get(f"mission{i}") == 1]

    def has_pending_missions(self) -> bool:
        return len(self.pending_missions()) > 0

    def set_global_mission_progress(self, remaining: int, initial_crew: int) -> None:
        """Inject global crew mission progress (imposter view).

        `remaining` is the aggregated count of pending missions across all
        crewmates; `initial_crew` is the initial crewmate count (fixed — not
        the currently-alive count) so the denominator matches Among Us rules.
        """
        self._global_mission_remaining = int(remaining)
        self._global_initial_crew = int(initial_crew)

    def set_current_plan(self, plan) -> None:
        """Store the planner's PlanResult so summarize_for_context can include it."""
        self._current_plan = plan

    def record_suspicion(self, name: str, delta: float) -> None:
        cur = self.suspicion.get(name, 0.0)
        self.suspicion[name] = max(0.0, min(1.0, cur + delta))

    def all_players_at_max_turns(self, max_turns: int = 5) -> bool:
        if self._force_all_spoke:
            return True
        if self.meeting_start_index < 0:
            return False
        alive = [self.bot_name] + self.alive_player_names()
        if not alive:
            return False
        return all(self.speech_counts.get(n, 0) >= max_turns for n in alive)

    # ── Persistence ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Full serializable snapshot of current state (incl. subclass extras)."""
        data = {
            "bot_name": self.bot_name, "role": self.role,
            "tick": self.tick, "phase": self.phase,
            "attack_ready": self._attack_ready,
            "own_position": list(self.own_position) if self.own_position else None,
            "mission_status": self.mission_status,
            "chat_log": self.chat_log,
            "meeting_start_index": self.meeting_start_index,
            "meeting_summaries": self.meeting_summaries,
            "last_vote_result_meeting": self._last_vote_result_meeting,
            "suspicion": self.suspicion,
            "dead_players": list(self._dead_players),
        }
        data.update(self._extra_save_data())
        return data

    def save_to_json(self, path: str) -> None:
        data = self.to_dict()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[StateBuilder] save failed: {e}")

    def load_from_json(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.tick = data.get("tick", 0)
            self.phase = data.get("phase", 0)
            self._attack_ready = data.get("attack_ready", False)
            pos = data.get("own_position")
            self.own_position = tuple(pos) if pos else None
            self.mission_status = data.get("mission_status", {})
            self.chat_log = data.get("chat_log", [])
            self.meeting_start_index = data.get("meeting_start_index", -1)
            self.meeting_summaries = data.get("meeting_summaries", [])
            self._last_vote_result_meeting = data.get("last_vote_result_meeting", 0)
            self.suspicion = data.get("suspicion", {})
            self._dead_players = set(data.get("dead_players", []))
            self._load_extra_data(data)
            return True
        except Exception as e:
            print(f"[StateBuilder] load failed: {e}")
            return False

    def _extra_save_data(self) -> dict:
        return {}

    def _load_extra_data(self, data: dict) -> None:
        pass

    # ── Internal helpers ────────────────────────────────────────────

    def _save_meeting_summary(self, ejection_msg: str, ejected: Optional[str] = None,
                              ejected_role: Optional[str] = None,
                              skipped: bool = False,
                              reporter: Optional[str] = None,
                              victim: Optional[str] = None,
                              source: str = "unknown",
                              authoritative: bool = False) -> None:
        log = self.meeting_log
        self.meeting_summaries.append({
            "meeting_no": self._meeting_count,
            "tick": self.tick,
            "ejected": ejected,
            "ejected_role": ejected_role,
            "skipped": skipped,
            "reporter": reporter,
            "victim": victim,
            "chat_count": len(log),
            "source": source,
            "authoritative": authoritative,
            "ejection_msg": ejection_msg,
        })

    def _meeting_summary_index(self, meeting_no: int) -> Optional[int]:
        for idx in range(len(self.meeting_summaries) - 1, -1, -1):
            if self.meeting_summaries[idx].get("meeting_no") == meeting_no:
                return idx
        return None

    def _finalize_vote_result(self, ejected: Optional[str]) -> None:
        self._last_vote_result_meeting = self._meeting_count
        self.phase = 0
        self._attack_ready = False
        self.vote_time = False
        # Preserve meeting chat before resetting index
        if self.meeting_start_index >= 0:
            self.last_meeting_chat = list(self.chat_log[self.meeting_start_index:])
        self.meeting_start_index = -1
        self.current_meeting_reporter = None
        self.current_meeting_victim = None
        self.current_meeting_new_dead = []
        if ejected:
            self._dead_players.add(ejected)

    @staticmethod
    def _format_vote_result_tag(ejected: Optional[str],
                                ejected_role: Optional[str],
                                skipped: bool) -> str:
        if skipped or not ejected:
            return "SKIPPED"
        return f"ejected={ejected} role={ejected_role}"

    @staticmethod
    def _pos_to_room(x: float, z: float) -> str:
        """Map x,z coordinates to Among Us room name using door boundaries."""
        if z <= 42:
            return "Navigation"
        if x <= 88 and 53 <= z <= 64:
            return "Weapons"
        if 89 <= x <= 99 and 53 <= z <= 60:
            return "O2"
        if x >= 113 and z <= 59:
            return "Shields"
        if x >= 120 and 60 <= z <= 70:
            return "Communications"
        if x >= 110 and 70 <= z <= 88:
            return "Storage"
        if 105 <= x <= 110 and 70 <= z <= 78:
            return "Admin"
        if x >= 115 and 88 <= z <= 100:
            return "Electrical"
        if x >= 109 and z >= 106:
            return "Lower Engine"
        if 100 <= x <= 108 and 115 <= z:
            return "Reactor"
        if 100 <= x <= 108 and 106 <= z < 115:
            return "Security"
        if x <= 95 and z >= 106:
            return "Upper Engine"
        if 88 <= x <= 95 and 88 <= z <= 100:
            return "Medbay"
        if 80 <= x <= 102 and 64 <= z <= 90:
            return "Cafeteria"
        return "Hallway"

    @staticmethod
    def _fmt_pos(pos) -> str:
        if pos is None:
            return "unknown"
        return f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"

    @staticmethod
    def _fmt_pos_with_room(pos) -> str:
        if pos is None:
            return "unknown"
        room = BaseStateBuilder._pos_to_room(pos[0], pos[2])
        return f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) [{room}]"
