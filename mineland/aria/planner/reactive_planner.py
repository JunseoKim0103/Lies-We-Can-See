"""ReactivePlanner — single-step decision (rule or LLM).

policy="rule": deterministic priority dispatch (Steve's logic)
policy="llm":  single LLM call decides mode from full context, rule fallback on error
"""

from __future__ import annotations

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from .base import BasePlanner, PlanResult


VALID_MODES = {"kill", "report", "vote", "meeting", "mission", "explore"}


class ReactiveDecision(BaseModel):
    mode: str = Field(description="One of: kill, report, vote, meeting, mission, explore.")
    target: str = Field(default="", description="For kill mode: player name to target. Empty otherwise.")
    reasoning: str = Field(description="Why (1 sentence).")


_MAP_LAYOUT = """\
MAP LAYOUT (rooms connected by hallways):
  Cafeteria (center) — Weapons, O2, Admin, Storage, Medbay
  Upper Engine — Reactor, Medbay, Security
  Lower Engine — Electrical, Security, Reactor
  Navigation — O2, Weapons, Shields
  Shields — Communications, Storage, Navigation
  Electrical — Lower Engine, Storage
  Admin — Cafeteria, Storage
  Medbay — Upper Engine, Cafeteria
Your position and other players' positions show [Room] tags — use room names \
in meetings for alibis and accusations."""

_SYSTEM_PROMPT_IMPOSTER = """\
You are {bot_name}, an IMPOSTER in an Among Us-style Minecraft game.

{map_layout}

Decide what to do RIGHT NOW based on the current situation.
Choose exactly ONE: kill, report, mission, explore.

Mode descriptions:
- kill: Attack a nearby player (imposter only, requires attack_ready) or stalk a target to approach
- report: Approach a nearby corpse and report it. As an imposter you may self-report your own kill to deflect suspicion
- mission: Pretend to do a task (FAKE — imposters have no real tasks)
- explore: Move around the map, observe player movements, position for a future kill

You are the ONLY role that can kill. When the context shows a \
"KILL OPPORTUNITY" with crewmates in range, you should choose kill and \
set target to the crewmate you want to approach. You will scan the area \
on arrival to check if it is safe before killing.

When you choose kill, "target" MUST be the exact player name. NEVER leave target empty when mode is kill.

DECEPTION via FAKE MISSIONS: as an imposter you have NO real tasks, but \
you can pick mode="mission" to FAKE doing one. This is a core blending \
tactic — moving between mission spots and pretending to interact builds \
trust and gives you natural cover to follow crewmates. Mix kill, mission, \
and explore so your movements look like a normal crewmate's. Do NOT spam \
kill mode every step when attack is ready; alternate with fake mission \
between kills to keep suspicion off you.

If the context contains a "Recent kill DEFERs" section, use it to ADAPT — \
do not repeat the same target/situation that just failed. Examples:
- "DEFER: target X dist=N too far" → pick a CLOSER target, or switch to \
  explore until you are nearer to someone.
- "DEFER: witness behind/forward" → either switch target to someone away \
  from witnesses, or explore to relocate before trying again.

{style_guidance}

OUTPUT (JSON):
{{
  "mode": "<mode>",
  "target": "<REQUIRED: exact player name when kill, empty otherwise>",
  "reasoning": "<1 sentence>"
}}
"""

_SYSTEM_PROMPT_CREWMATE = """\
You are {bot_name}, a CREWMATE in an Among Us-style Minecraft game.

{map_layout}

Decide what to do RIGHT NOW based on the current situation.
Choose exactly ONE: report, mission, explore.
You cannot kill — that ability belongs only to imposters.

Mode descriptions:
- report: Approach a nearby corpse and report it to call an emergency meeting
- mission: Work on completing one of your assigned tasks (real, contributes to crew win)
- explore: Move around the map, stay close to others for safety, observe player movements.
  When all your missions are done, use explore to patrol OTHER mission locations
  and watch for imposters faking tasks.

{style_guidance}

OUTPUT (JSON):
{{
  "mode": "<mode>",
  "reasoning": "<1 sentence>"
}}
"""

# prompt_style guidance — same shape as Shortterm/Hierarchical planners
_DETERMINISTIC_GUIDANCE_IMPOSTER = """\
Priority guidelines:
- If attack_ready=True and a crewmate is in KILL OPPORTUNITY → kill
- If you just killed and a corpse is nearby → consider self-report to deflect suspicion
- If suspicion on you is rising or attack is on cooldown → fake mission
- Otherwise → explore (move to find isolated targets)"""

_DETERMINISTIC_GUIDANCE_CREWMATE = """\
Priority guidelines:
- If a corpse is visible nearby → report
- If you have pending missions → mission
- If all your missions are done → explore (patrol other mission locations to watch for fake-task imposters)
- Otherwise → explore (stay close to other players for safety)"""

_MINIMAL_GUIDANCE = """\
Choose based on the situation."""


class ReactivePlanner(BasePlanner):
    def __init__(self, policy: str = "rule", role: str = "crewmate",
                 bot_name: str = "Aria", llm=None):
        self.policy = policy
        self.role = role
        self.bot_name = bot_name
        self._chain = None
        self._history: list = []
        if llm is not None:
            self._chain = llm | JsonOutputParser(pydantic_object=ReactiveDecision)

    def set_llm(self, llm) -> None:
        self._chain = llm | JsonOutputParser(pydantic_object=ReactiveDecision)

    def plan(self, state_context, memory_context, skill_context,
             game_state, state_builder, role, prompt_style, **kwargs) -> PlanResult:
        tick = getattr(game_state, "tick", 0)
        # Meeting/vote are phase-locked: never delegated to LLM.
        if game_state.phase == 1:
            all_spoke = state_builder.all_players_at_max_turns(5)
            if game_state.vote_time or all_spoke:
                result = PlanResult("vote", "vote time or all spoke", "Cast vote")
                self._record(tick, role, "phase_locked", result)
                return result
            result = PlanResult("meeting", "phase==1", "Participate in discussion")
            self._record(tick, role, "phase_locked", result)
            return result

        if self.policy == "llm" and self._chain is not None:
            result = self._llm_plan(state_context, memory_context, skill_context,
                                    state_builder, role, prompt_style)
            if result is not None:
                if result.mode == "kill" and not result.target and hasattr(state_builder, "nearest_alive_player"):
                    result.target = state_builder.nearest_alive_player()
                self._record(tick, role, "llm", result)
                return result
        result = self._rule_based(game_state, state_builder, role)
        self._record(tick, role, "rule", result)
        return result

    def _record(self, tick: int, role: str, source: str, result: PlanResult) -> None:
        self._history.append({
            "tick": tick,
            "role": role,
            "policy": self.policy,
            "source": source,
            "mode": result.mode,
            "target": result.target,
            "reasoning": result.reasoning,
            "short_term_plan": result.short_term_plan,
        })

    def _alive_targets_line(self, state_builder, role) -> str:
        """Build an explicit list of valid target candidates for the LLM.
        Mirrors VoteModule/MeetingModule: surface alive non-self names with
        a teammate tag so the LLM cannot accidentally pick a dead/teammate.
        """
        try:
            alive = state_builder.alive_player_names()
        except Exception:
            alive = []
        teammate = getattr(state_builder, "teammate", None)
        items = []
        for name in alive:
            if name == self.bot_name:
                continue
            tag = " (your imposter teammate, not a crewmate)" if (role == "imposter" and name == teammate) else ""
            items.append(f"{name}{tag}")
        if not items:
            return "Alive non-self players: (none)"
        return "Alive non-self players (pick `target` from here when mode=kill):\n  " + "\n  ".join(items)

    def _llm_plan(self, state_context, memory_context, skill_context,
                  state_builder, role, prompt_style) -> Optional[PlanResult]:
        template = _SYSTEM_PROMPT_IMPOSTER if role == "imposter" else _SYSTEM_PROMPT_CREWMATE
        if prompt_style == "deterministic":
            style_guidance = (_DETERMINISTIC_GUIDANCE_IMPOSTER if role == "imposter"
                              else _DETERMINISTIC_GUIDANCE_CREWMATE)
        else:
            style_guidance = _MINIMAL_GUIDANCE
        system_text = template.format(bot_name=self.bot_name, style_guidance=style_guidance, map_layout=_MAP_LAYOUT)
        alive_line = self._alive_targets_line(state_builder, role)
        human_text = f"{state_context}\n{alive_line}\n{memory_context}"
        if skill_context:
            human_text += f"\n{skill_context}"
        human_text += "\n\nDecide now."

        try:
            result = self._chain.invoke([
                SystemMessage(content=system_text),
                HumanMessage(content=human_text),
            ])
            mode = result.get("mode", "explore").strip().lower()
            reasoning = result.get("reasoning", "")
            if mode not in VALID_MODES:
                return None
            target = result.get("target", "").strip() or None
            return PlanResult(mode=mode, reasoning=reasoning, short_term_plan=reasoning, target=target)
        except Exception:
            return None

    def _rule_based(self, game_state, state_builder, role) -> PlanResult:
        """Steve-equivalent priority dispatch."""
        phase = game_state.phase
        all_spoke = state_builder.all_players_at_max_turns(5)

        if phase == 1 and (game_state.vote_time or all_spoke):
            return PlanResult("vote", "vote time or all spoke", "Cast vote")
        if phase == 1:
            return PlanResult("meeting", "phase==1", "Participate in discussion")
        # Resolve nearest crewmate target + distance for imposter
        _kill_target = None
        _kill_dist = float("inf")
        if role == "imposter" and hasattr(state_builder, "nearest_alive_player"):
            _kill_target = state_builder.nearest_alive_player()
            if _kill_target and state_builder.own_position:
                p = state_builder.players.get(_kill_target)
                if p and p.position:
                    ox, oy, oz = state_builder.own_position
                    dx = p.position[0] - ox
                    dy = p.position[1] - oy
                    dz = p.position[2] - oz
                    _kill_dist = (dx * dx + dy * dy + dz * dz) ** 0.5

        # Kill only when a crewmate is nearby; otherwise blend in
        if role == "imposter" and state_builder.get_attack_ready() and _kill_dist < 120:
            return PlanResult("kill", f"imposter + attack_ready ({_kill_target} dist={_kill_dist:.0f})",
                              "Hunt and kill", target=_kill_target)
        if state_builder.has_pending_missions():
            return PlanResult("mission", "pending missions", "Work on missions")

        # Imposter stalk — crewmate within range, approach
        if role == "imposter" and _kill_target and _kill_dist < 120:
            return PlanResult("kill", f"stalking {_kill_target} (dist={_kill_dist:.0f})", "Stalk player", target=_kill_target)

        return PlanResult("explore", "default", "Explore the map")
