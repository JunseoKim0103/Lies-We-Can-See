"""MeetingReflector — LLM-based reflection at meeting phase transitions.

Triggers on phase 0→1 (meeting start) and 1→0 (meeting end).
At meeting end: analyzes discussion, updates suspicion beliefs, extracts skills.
At meeting start: lighter reflection on what happened during explore phase.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from .base import BaseReflector, ReflectionResult


class MeetingReflectionOutput(BaseModel):
    insights: str = Field(description="What did you learn? 2-3 sentences.")
    most_suspicious: str = Field(description="Player name most suspicious, or 'none'.")
    suspicion_delta: float = Field(description="How much to increase suspicion for that player (-0.3 to +0.3).")
    suspicion_updates: list = Field(default=[], description="List of {player, delta} for each discussed player.")
    skill_learned: str = Field(description="A strategic pattern worth remembering, or 'none'.")


_SYSTEM_PROMPT_MEETING_END = """\
You are {bot_name}, a {role} in an Among Us-style Minecraft game.
A meeting just ended. Reflect on what happened.

Analyze the meeting discussion and your observations:
- Who made contradictory statements?
- Who was accused the most? Was it justified?
- Did anyone deflect without providing their own alibi?
- What was the vote outcome?
- If the context contains [AUTHORITATIVE_VOTE_OUTCOME], it is the final vote outcome.
- Never call the vote a skip if [AUTHORITATIVE_VOTE_OUTCOME] says skipped: false.
- Never infer the final vote outcome from discussion statements such as "I will skip" or "I'll vote X".
- If Past Meeting Summaries conflict with discussion text, trust Past Meeting Summaries.

{style_guidance}

OUTPUT (JSON):
{{
  "insights": "<2-3 sentence summary of what you learned>",
  "most_suspicious": "<player name or 'none'>",
  "suspicion_delta": <float -0.3 to +0.3>,
  "suspicion_updates": [
    {{"player": "<name>", "delta": <float -0.3 to +0.3>}}
  ],
  "skill_learned": "<strategic pattern worth remembering, or 'none'>"
}}
For suspicion_updates, evaluate EVERY alive player's behavior during the meeting:
- Redirecting blame without evidence → +suspicion
- Staying silent or evasive → +suspicion
- Providing a verifiable alibi → -suspicion
- Accusing others aggressively without proof → +suspicion
- Corroborating others' testimony consistently → -suspicion
- Being near the body location → +suspicion
Even if the vote was skipped, behavioral signals still matter. Do NOT return empty suspicion_updates.
"""

_SYSTEM_PROMPT_MEETING_START = """\
You are {bot_name}, a {role} in an Among Us-style Minecraft game.
A meeting was just called. Reflect briefly on the explore phase that just ended.

- Where were you? What did you see?
- Did you notice anything suspicious before the meeting?

{style_guidance}

OUTPUT (JSON):
{{
  "insights": "<1-2 sentence summary>",
  "most_suspicious": "<player name or 'none'>",
  "suspicion_delta": <float -0.3 to +0.3>,
  "suspicion_updates": [
    {{"player": "<name>", "delta": <float -0.3 to +0.3>}}
  ],
  "skill_learned": "none"
}}
For suspicion_updates, evaluate players you observed during the explore phase:
- Following you or others closely → +suspicion
- Being alone near a body location → +suspicion
- Completing tasks visibly → -suspicion
- Moving erratically or running away → +suspicion
"""

_DETERMINISTIC_GUIDANCE = """\
Think systematically: cross-reference alibis, track who accused whom,
note behavioral patterns (deflection, silence, over-eagerness)."""

_MINIMAL_GUIDANCE = """\
Reflect based on what you observed."""


class MeetingReflector(BaseReflector):
    def __init__(self, llm=None):
        self._prev_phase: int = 0
        self._chain = None
        self._history: list = []
        if llm is not None:
            self._chain = llm | JsonOutputParser(pydantic_object=MeetingReflectionOutput)

    def set_llm(self, llm) -> None:
        """Set LLM after construction (for lazy init in aria_agent)."""
        self._chain = llm | JsonOutputParser(pydantic_object=MeetingReflectionOutput)

    def should_reflect(self, game_state, prev_game_state, action_completed):
        current_phase = getattr(game_state, "phase", 0)
        prev_phase = self._prev_phase
        self._prev_phase = current_phase
        return current_phase != prev_phase

    def reflect(self, state_context, memory_context, game_state, role, prompt_style):
        if self._chain is None:
            return ReflectionResult()

        phase = getattr(game_state, "phase", 0)
        # phase==1 means we just entered meeting (0→1), phase==0 means meeting just ended (1→0)
        if phase == 1:
            template = _SYSTEM_PROMPT_MEETING_START
        else:
            template = _SYSTEM_PROMPT_MEETING_END

        style_guidance = _DETERMINISTIC_GUIDANCE if prompt_style == "deterministic" else _MINIMAL_GUIDANCE
        bot_name = getattr(game_state, "bot_name", "Aria")  # fallback

        system_text = template.format(
            bot_name=bot_name, role=role, style_guidance=style_guidance,
        )
        human_text = f"{state_context}\n{memory_context}\n\nReflect now."

        try:
            result = self._chain.invoke([
                SystemMessage(content=system_text),
                HumanMessage(content=human_text),
            ])
            insights = result.get("insights", "")
            if not isinstance(insights, str):
                insights = "" if insights is None else str(insights)
            most_sus = result.get("most_suspicious", "none")
            delta = float(result.get("suspicion_delta", 0.0))
            delta = max(-0.3, min(0.3, delta))
            skill = result.get("skill_learned", "none")

            belief_updates = {}
            if most_sus and most_sus.lower() != "none" and delta != 0:
                belief_updates[most_sus] = delta
            for upd in result.get("suspicion_updates", []):
                p = upd.get("player", "")
                d = float(upd.get("delta", 0))
                d = max(-0.3, min(0.3, d))
                if p and d != 0 and p.lower() != "none":
                    belief_updates.setdefault(p, 0)
                    belief_updates[p] = max(-1.0, min(1.0, belief_updates[p] + d))

            skill_candidates = []
            if skill and skill.lower() != "none":
                # Meeting-end skills come from observing other players' behavior
                # (alibis, deflection, contradictions) → source="observed".
                # Meeting-start reflections summarize the agent's own explore
                # phase → source="self".
                source = "observed" if phase == 0 else "self"
                skill_candidates.append({
                    "source": source,
                    "agent_name": bot_name,
                    "phase": "meeting" if phase == 0 else "explore",
                    "situation": f"meeting reflection at tick {getattr(game_state, 'tick', 0)}",
                    "behavior": skill,
                    "outcome": insights[:80] if insights else "",
                    "tick": getattr(game_state, "tick", 0),
                })

            self._history.append({
                "tick": getattr(game_state, "tick", 0),
                "phase": phase,
                "trigger": "meeting_start" if phase == 1 else "meeting_end",
                "role": role,
                "prompt_style": prompt_style,
                "insights": insights,
                "most_suspicious": most_sus,
                "suspicion_delta": delta,
                "skill_learned": skill,
            })

            return ReflectionResult(
                insights=insights,
                belief_updates=belief_updates,
                skill_candidates=skill_candidates,
            )

        except Exception as e:
            self._history.append({
                "tick": getattr(game_state, "tick", 0),
                "phase": phase,
                "trigger": "meeting_start" if phase == 1 else "meeting_end",
                "role": role,
                "error": str(e),
            })
            return ReflectionResult()
