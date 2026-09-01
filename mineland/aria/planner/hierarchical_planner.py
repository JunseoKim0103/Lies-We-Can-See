"""HierarchicalPlanner — long-term strategy + short-term plan (Alex structure).

Always LLM-based (config.py validates hierarchical+rule as impossible).

Two levels:
1. Long-term strategy: regenerated every N short-term plans via LLM
2. Short-term plan: LLM selects mode considering long-term strategy context

Falls back to rule-based mode selection if LLM fails.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from .base import BasePlanner, PlanResult
from .reactive_planner import ReactivePlanner


VALID_MODES = {"kill", "report", "vote", "meeting", "mission", "explore"}


class LongTermPlan(BaseModel):
    strategy: str = Field(description="Overall strategy for the next several steps (2-3 sentences).")
    priority_target: str = Field(description="Player to focus on, or 'none'.")
    reasoning: str = Field(description="Why this strategy.")


class ShortTermPlan(BaseModel):
    mode: str = Field(description="One of: kill, report, mission, explore.")
    target: str = Field(default="", description="For kill mode: exact player name to target. Empty otherwise.")
    reasoning: str = Field(description="Why this mode given the long-term strategy.")
    plan: str = Field(description="Specific action for this step (1 sentence).")


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

_LONGTERM_SYSTEM = """\
You are {bot_name}, a {role} in an Among Us-style Minecraft game.

{map_layout}

Develop a high-level strategy for the next several steps.
Consider: your role, the current game state, who is alive, suspicion levels,
mission progress, and what happened in recent meetings.

`priority_target` MUST be the exact name of an alive non-self player
from the "Alive non-self players" list below, or "none". Never invent
names, and never use a dead/self/teammate player. If unsure, use "none".

{style_guidance}

OUTPUT (JSON):
{{
  "strategy": "<overall strategy — 2-3 sentences>",
  "priority_target": "<exact alive player name from the list, or 'none'>",
  "reasoning": "<why this strategy>"
}}
"""

_SHORTTERM_SYSTEM = """\
You are {bot_name}, a {role} in an Among Us-style Minecraft game.

Your current long-term strategy:
{long_term_plan}

Long-term priority target (focus player, may be 'none'):
{priority_target}

Given this strategy and the current state, decide what to do THIS step.
Choose exactly ONE mode: kill, report, mission, explore.
(Meeting and voting are handled automatically when a meeting is active.)

Mode descriptions:
- kill: Attack a nearby player (imposter only, requires attack_ready) or stalk a target to approach
- report: Approach a nearby corpse and report it. Imposters may self-report their own kill to deflect suspicion
- mission: Pretend (imposter, fake) or actually work on (crewmate, real) a task
- explore: Move around the map, observe player movements

When mode is "kill", "target" MUST be the exact player name you want to
hunt. Prefer the long-term priority_target above when it is alive and
sensible, but you may pick a different living player when the situation
demands it. NEVER leave target empty when mode is kill.

If you are an IMPOSTER: you have NO real missions, but you can pick
mode="mission" to FAKE one. Faking missions is core deception — it gives
you natural cover for being near crewmates and following them. Use fake
missions BETWEEN kills (especially during attack cooldown) so your
behavior pattern matches a crewmate's. Pure kill+explore looks suspicious;
kill+fakemission+explore blends in.

If you are an imposter and the context contains a "Recent kill DEFERs"
section, use it to ADAPT — do not repeat the same target/situation that
just failed. Examples:
- "DEFER: target X dist=N too far" → pick a CLOSER target, or switch to
  explore until you are nearer to someone (and consider revising your
  priority_target away from X if it keeps failing).
- "DEFER: witness behind/forward" → either switch target to someone away
  from witnesses, or explore to relocate before trying again.

OUTPUT (JSON):
{{
  "mode": "<mode>",
  "target": "<REQUIRED: exact player name when kill, empty otherwise>",
  "reasoning": "<why, considering your strategy>",
  "plan": "<specific action — 1 sentence>"
}}
"""

_DETERMINISTIC_GUIDANCE = """\
Think strategically: as imposter, plan kills around isolation opportunities,
USE FAKE MISSIONS as cover between kills (no real tasks but pretend to do
them so crewmates think you're working), and build alibis in advance.
As crewmate, track player movements and build a case from evidence across
multiple meetings."""

_MINIMAL_GUIDANCE = """\
Plan based on your role and the situation."""


class HierarchicalPlanner(BasePlanner):
    def __init__(self, role: str = "crewmate", bot_name: str = "Aria",
                 longterm_interval: int = 5, llm=None):
        self.role = role
        self.bot_name = bot_name
        self._longterm_interval = longterm_interval
        self._long_term_plan: Optional[str] = None
        self._priority_target: Optional[str] = None
        self._plan_count: int = 0
        self._reactive = ReactivePlanner(policy="rule", role=role)
        self._history: list = []

        self._longterm_chain = None
        self._shortterm_chain = None
        if llm is not None:
            self._set_chains(llm)

    def set_llm(self, llm) -> None:
        """Set LLM after construction."""
        self._set_chains(llm)

    def _set_chains(self, llm) -> None:
        self._longterm_chain = llm | JsonOutputParser(pydantic_object=LongTermPlan)
        self._shortterm_chain = llm | JsonOutputParser(pydantic_object=ShortTermPlan)

    def plan(self, state_context, memory_context, skill_context,
             game_state, state_builder, role, prompt_style, **kwargs) -> PlanResult:
        tick = getattr(game_state, "tick", 0)
        # Meeting/vote are phase-locked: never delegated to LLM.
        if game_state.phase == 1:
            all_spoke = state_builder.all_players_at_max_turns(5)
            if game_state.vote_time or all_spoke:
                result = PlanResult("vote", "vote time or all spoke", "Cast vote")
                self._record(tick, role, "phase_locked", result, longterm_updated=False)
                return result
            result = PlanResult("meeting", "phase==1", "Participate in discussion")
            self._record(tick, role, "phase_locked", result, longterm_updated=False)
            return result

        # Regenerate long-term plan every N steps
        longterm_updated = False
        if self._plan_count % self._longterm_interval == 0:
            new_strategy, new_priority = self._generate_long_term(
                state_context, memory_context, skill_context,
                state_builder, role, prompt_style,
            )
            if new_strategy:
                self._long_term_plan = new_strategy
            if new_priority:
                self._priority_target = new_priority
            longterm_updated = True

        self._plan_count += 1

        # Drop priority_target if that player is already dead.
        if self._priority_target and hasattr(state_builder, "alive_player_names"):
            if self._priority_target not in state_builder.alive_player_names():
                self._priority_target = None

        # Short-term plan considering long-term strategy
        result = self._generate_short_term(
            state_context, memory_context, skill_context,
            game_state, state_builder, role, prompt_style,
        )
        if result is not None:
            result.long_term_plan = self._long_term_plan
            # If LLM picked kill but didn't name a target, use long-term
            # priority_target → nearest_alive_player as ordered fallback.
            if result.mode == "kill" and not result.target:
                if self._priority_target:
                    result.target = self._priority_target
                elif hasattr(state_builder, "nearest_alive_player"):
                    result.target = state_builder.nearest_alive_player()
            self._record(tick, role, "llm_short_term", result,
                         longterm_updated=longterm_updated)
            return result

        # Fallback to rule-based
        fallback = self._reactive._rule_based(game_state, state_builder, role)
        fallback.long_term_plan = self._long_term_plan
        self._record(tick, role, "rule_fallback", fallback,
                     longterm_updated=longterm_updated)
        return fallback

    def _record(self, tick: int, role: str, source: str, result: PlanResult,
                longterm_updated: bool) -> None:
        self._history.append({
            "tick": tick,
            "role": role,
            "source": source,
            "longterm_updated": longterm_updated,
            "long_term_plan": self._long_term_plan,
            "priority_target": self._priority_target,
            "plan_count": self._plan_count,
            "mode": result.mode,
            "target": result.target,
            "reasoning": result.reasoning,
            "short_term_plan": result.short_term_plan,
        })

    def _alive_targets_line(self, state_builder, role) -> str:
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
        return "Alive non-self players (pick `target` / `priority_target` from here):\n  " + "\n  ".join(items)

    def _generate_long_term(self, state_context, memory_context, skill_context,
                            state_builder, role, prompt_style) -> tuple:
        """Returns (strategy, priority_target). Either may be None."""
        if self._longterm_chain is None:
            # Static fallback
            if role == "imposter":
                return ("Blend in, build trust, eliminate crewmates one by one.", None)
            return ("Complete missions, stay grouped, identify the imposter.", None)

        style_guidance = _DETERMINISTIC_GUIDANCE if prompt_style == "deterministic" else _MINIMAL_GUIDANCE
        system_text = _LONGTERM_SYSTEM.format(
            bot_name=self.bot_name, role=role, style_guidance=style_guidance,
            map_layout=_MAP_LAYOUT,
        )
        alive_line = self._alive_targets_line(state_builder, role)
        human_text = f"{state_context}\n{alive_line}\n{memory_context}"
        if skill_context:
            human_text += f"\n{skill_context}"
        human_text += "\n\nDevelop your strategy."

        try:
            result = self._longterm_chain.invoke([
                SystemMessage(content=system_text),
                HumanMessage(content=human_text),
            ])
            strategy = result.get("strategy", "") or None
            priority = (result.get("priority_target", "") or "").strip()
            if priority.lower() in ("", "none", "null"):
                priority = None
            return (strategy, priority)
        except Exception:
            return (None, None)

    def _generate_short_term(self, state_context, memory_context, skill_context,
                             game_state, state_builder, role, prompt_style) -> Optional[PlanResult]:
        if self._shortterm_chain is None:
            return None

        long_term_text = self._long_term_plan or "(no strategy yet)"
        priority_text = self._priority_target or "none"
        system_text = _SHORTTERM_SYSTEM.format(
            bot_name=self.bot_name, role=role, long_term_plan=long_term_text,
            priority_target=priority_text,
        )
        alive_line = self._alive_targets_line(state_builder, role)
        human_text = f"{state_context}\n{alive_line}\n{memory_context}"
        if skill_context:
            human_text += f"\n{skill_context}"
        human_text += "\n\nDecide your mode for this step."

        try:
            result = self._shortterm_chain.invoke([
                SystemMessage(content=system_text),
                HumanMessage(content=human_text),
            ])
            mode = result.get("mode", "explore").strip().lower()
            reasoning = result.get("reasoning", "")
            plan_text = result.get("plan", "")
            target = result.get("target", "").strip() or None

            if mode not in VALID_MODES:
                return None
            return PlanResult(mode=mode, reasoning=reasoning, short_term_plan=plan_text, target=target)
        except Exception:
            return None
