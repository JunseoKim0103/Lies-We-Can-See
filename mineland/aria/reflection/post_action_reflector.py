"""PostActionReflector — LLM-based reflection after every action completes.

Evaluates: was this action effective? Did it raise suspicion?
Outputs: insights, belief updates, optional skill candidate.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from .base import BaseReflector, ReflectionResult


class PostActionOutput(BaseModel):
    insights: str = Field(description="Was the action effective? 1-2 sentences.")
    suspicion_change_player: str = Field(description="Player whose suspicion should change, or 'none'.")
    suspicion_change_delta: float = Field(description="Suspicion delta (-0.2 to +0.2).")
    skill_learned: str = Field(default="none", description="A reusable tactic or lesson learned from this action, or 'none'.")


_SYSTEM_PROMPT = """\
You are {bot_name}, a {role} in an Among Us-style Minecraft game.

Your last action just completed. Briefly evaluate:
- Was it effective? Did it advance your goals?
- Did it potentially raise suspicion on you or someone else?
- What should you adjust going forward?

OUTPUT (JSON):
{{
  "insights": "<1-2 sentence evaluation>",
  "suspicion_change_player": "<player name or 'none'>",
  "suspicion_change_delta": <float -0.2 to +0.2>,
  "skill_learned": "<reusable tactic or lesson, or 'none'>"
}}
"""


class PostActionReflector(BaseReflector):
    def __init__(self, llm=None):
        self._chain = None
        self._history: list = []
        if llm is not None:
            self._chain = llm | JsonOutputParser(pydantic_object=PostActionOutput)

    def set_llm(self, llm) -> None:
        self._chain = llm | JsonOutputParser(pydantic_object=PostActionOutput)

    def should_reflect(self, game_state, prev_game_state, action_completed):
        return action_completed

    def reflect(self, state_context, memory_context, game_state, role, prompt_style):
        # prompt_style accepted for BaseReflector signature parity but unused —
        # post-action reflection uses the fixed _SYSTEM_PROMPT template by design.
        # MeetingReflector is the reflector that branches on prompt_style.
        if self._chain is None:
            return ReflectionResult()

        bot_name = getattr(game_state, "bot_name", "Aria")
        system_text = _SYSTEM_PROMPT.format(bot_name=bot_name, role=role)
        human_text = f"{state_context}\n{memory_context}\n\nEvaluate your last action."

        try:
            result = self._chain.invoke([
                SystemMessage(content=system_text),
                HumanMessage(content=human_text),
            ])
            insights = result.get("insights", "")
            if not isinstance(insights, str):
                insights = "" if insights is None else str(insights)
            player = result.get("suspicion_change_player", "none")
            delta = float(result.get("suspicion_change_delta", 0.0))
            delta = max(-0.2, min(0.2, delta))

            belief_updates = {}
            if player and player.lower() != "none" and delta != 0:
                belief_updates[player] = delta

            skill = result.get("skill_learned", "none")
            skill_candidates = []
            if skill and skill.lower() != "none":
                skill_candidates.append({
                    "source": "self",
                    "agent_name": bot_name,
                    "phase": "action",
                    "situation": f"post-action reflection at tick {getattr(game_state, 'tick', 0)}",
                    "behavior": skill,
                    "outcome": insights[:80] if insights else "",
                    "tick": getattr(game_state, "tick", 0),
                })

            self._history.append({
                "tick": getattr(game_state, "tick", 0),
                "trigger": "post_action",
                "role": role,
                "insights": insights,
                "suspicion_change_player": player,
                "suspicion_change_delta": delta,
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
                "trigger": "post_action",
                "role": role,
                "error": str(e),
            })
            return ReflectionResult()
