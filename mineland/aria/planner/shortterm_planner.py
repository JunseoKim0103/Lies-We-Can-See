"""ShorttermPlanner — mode selection + module dispatch (Steve structure).

policy="rule": Steve-style priority IF/ELSE (via ReactivePlanner._rule_based)
policy="llm":  LLM selects mode from full context
"""

from __future__ import annotations

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from .base import BasePlanner, PlanResult
from .reactive_planner import ReactivePlanner


VALID_MODES = {"kill", "report", "vote", "meeting", "mission", "explore"}


class ModeSelection(BaseModel):
    mode: str = Field(description="One of: kill, report, mission, explore.")
    target: str = Field(default="", description="For kill mode: exact player name to target. Empty otherwise.")
    reasoning: str = Field(description="Why this mode was chosen (1-2 sentences).")
    plan: str = Field(description="What to do in this mode (1 sentence).")


_SYSTEM_PROMPT = """\
You are {bot_name}, a {role} in an Among Us-style Minecraft game.

Given the current game state and context, decide what to do this step.
Choose exactly ONE mode from: kill, report, mission, explore.
(Meeting and voting are handled automatically when a meeting is active.)

Mode descriptions:
- kill: Attack a nearby player (imposter only, requires attack_ready) or stalk
- report: Approach and report a nearby corpse
- mission: Work on completing a mission task
- explore: Move around the map, explore, or follow players

When mode is "kill", "target" MUST be the exact player name you want to
hunt. NEVER leave target empty when mode is kill. For other modes, target
should be empty.

If you are an IMPOSTER: you have NO real missions, but you can pick
mode="mission" to FAKE one. Faking missions is core deception — it gives
you cover for moving near crewmates without raising suspicion. Mix kill,
fake mission, and explore so your behavior looks like a normal crewmate's.
Do not spam kill mode every step when attack_ready; alternate with fake
mission between kills.

{style_guidance}

OUTPUT (JSON):
{{
  "mode": "<one of: kill, report, mission, explore>",
  "target": "<REQUIRED: exact player name when kill, empty otherwise>",
  "reasoning": "<why this mode — 1-2 sentences>",
  "plan": "<what to do — 1 sentence>"
}}
"""

_DETERMINISTIC_GUIDANCE = """\
Priority guidelines:
- If imposter and attack_ready → kill
- If there are pending missions → mission
- Otherwise → explore"""

_MINIMAL_GUIDANCE = """\
Choose based on the situation."""


class ShorttermPlanner(BasePlanner):
    def __init__(self, policy: str = "rule", role: str = "crewmate",
                 bot_name: str = "Aria", llm=None):
        self.policy = policy
        self.role = role
        self.bot_name = bot_name
        self._reactive = ReactivePlanner(policy="rule", role=role)
        self._chain = None
        if llm is not None:
            self._chain = llm | JsonOutputParser(pydantic_object=ModeSelection)

    def set_llm(self, llm) -> None:
        """Set LLM after construction (for lazy init)."""
        self._chain = llm | JsonOutputParser(pydantic_object=ModeSelection)

    def plan(self, state_context, memory_context, skill_context,
             game_state, state_builder, role, prompt_style, **kwargs) -> PlanResult:
        # Meeting/vote are phase-locked: never delegated to LLM.
        if game_state.phase == 1:
            all_spoke = state_builder.all_players_at_max_turns(5)
            if game_state.vote_time or all_spoke:
                return PlanResult("vote", "vote time or all spoke", "Cast vote")
            return PlanResult("meeting", "phase==1", "Participate in discussion")

        if self.policy == "llm" and self._chain is not None:
            result = self._llm_plan(
                state_context, memory_context, skill_context,
                game_state, state_builder, role, prompt_style,
            )
            if result is not None:
                # If LLM picked kill but failed to name a target, fall back
                # to nearest_alive_player so KillModule still gets a real name.
                if result.mode == "kill" and not result.target and hasattr(state_builder, "nearest_alive_player"):
                    result.target = state_builder.nearest_alive_player()
                return result
        # Fallback to rule-based
        return self._reactive._rule_based(game_state, state_builder, role)

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
        return "Alive non-self players (pick `target` from here when mode=kill or report):\n  " + "\n  ".join(items)

    def _llm_plan(self, state_context, memory_context, skill_context,
                  game_state, state_builder, role, prompt_style) -> Optional[PlanResult]:
        style_guidance = _DETERMINISTIC_GUIDANCE if prompt_style == "deterministic" else _MINIMAL_GUIDANCE
        system_text = _SYSTEM_PROMPT.format(
            bot_name=self.bot_name, role=role, style_guidance=style_guidance,
        )
        alive_line = self._alive_targets_line(state_builder, role)
        human_text = f"{state_context}\n{alive_line}\n{memory_context}"
        if skill_context:
            human_text += f"\n{skill_context}"
        human_text += "\n\nDecide your mode."

        try:
            result = self._chain.invoke([
                SystemMessage(content=system_text),
                HumanMessage(content=human_text),
            ])
            mode = result.get("mode", "explore").strip().lower()
            reasoning = result.get("reasoning", "")
            plan_text = result.get("plan", "")
            target = result.get("target", "").strip() or None

            if mode not in VALID_MODES:
                return None  # invalid → fall back to rule

            return PlanResult(mode=mode, reasoning=reasoning, short_term_plan=plan_text, target=target)

        except Exception:
            return None  # LLM error → fall back to rule
