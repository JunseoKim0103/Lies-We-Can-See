"""PeriodicReflector — LLM-based reflection every N python steps.

Evaluates overall strategy and situation periodically.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from .base import BaseReflector, ReflectionResult


class PeriodicOutput(BaseModel):
    insights: str = Field(description="Overall situation assessment. 2-3 sentences.")
    most_suspicious: str = Field(description="Most suspicious player, or 'none'.")
    suspicion_delta: float = Field(description="Suspicion delta (-0.3 to +0.3).")
    skill_learned: str = Field(description="Strategic pattern worth remembering, or 'none'.")


_SYSTEM_PROMPT = """\
You are {bot_name}, a {role} in an Among Us-style Minecraft game.

Take a moment to assess the overall situation:
- How is the game going for you?
- Who seems most suspicious based on everything you've observed?
- Is your current strategy working? Should you adjust?

OUTPUT (JSON):
{{
  "insights": "<2-3 sentence assessment>",
  "most_suspicious": "<player name or 'none'>",
  "suspicion_delta": <float -0.3 to +0.3>,
  "skill_learned": "<strategic pattern worth remembering, or 'none'>"
}}
"""


class PeriodicReflector(BaseReflector):
    """Reflects every N python steps (not ticks).

    Uses step counter instead of tick delta because in sync mode
    (enable_auto_pause=False) tick jumps are variable per step.
    """

    def __init__(self, interval: int = 10, llm=None):
        self._interval = interval     # python steps between reflections
        self._step_count: int = 0
        self._chain = None
        self._history: list = []
        if llm is not None:
            self._chain = llm | JsonOutputParser(pydantic_object=PeriodicOutput)

    def set_llm(self, llm) -> None:
        self._chain = llm | JsonOutputParser(pydantic_object=PeriodicOutput)

    def should_reflect(self, game_state, prev_game_state, action_completed):
        self._step_count += 1
        if self._step_count >= self._interval:
            return True
        return False

    def reflect(self, state_context, memory_context, game_state, role, prompt_style):
        # prompt_style accepted for BaseReflector signature parity but unused —
        # periodic reflection uses the fixed _SYSTEM_PROMPT template by design.
        self._step_count = 0  # reset after reflecting

        if self._chain is None:
            return ReflectionResult()

        bot_name = getattr(game_state, "bot_name", "Aria")
        system_text = _SYSTEM_PROMPT.format(bot_name=bot_name, role=role)
        human_text = f"{state_context}\n{memory_context}\n\nAssess the situation."

        try:
            result = self._chain.invoke([
                SystemMessage(content=system_text),
                HumanMessage(content=human_text),
            ])
            insights = result.get("insights", "")
            if not isinstance(insights, str):
                insights = "" if insights is None else str(insights)
            player = result.get("most_suspicious", "none")
            delta = float(result.get("suspicion_delta", 0.0))
            delta = max(-0.3, min(0.3, delta))
            skill = result.get("skill_learned", "none")

            belief_updates = {}
            if player and player.lower() != "none" and delta != 0:
                belief_updates[player] = delta

            skill_candidates = []
            if skill and skill.lower() != "none":
                skill_candidates.append({
                    "source": "self",
                    "agent_name": bot_name,
                    "phase": "explore",
                    "situation": f"periodic reflection at tick {getattr(game_state, 'tick', 0)}",
                    "behavior": skill,
                    "outcome": insights[:80] if insights else "",
                    "tick": getattr(game_state, "tick", 0),
                })

            self._history.append({
                "tick": getattr(game_state, "tick", 0),
                "trigger": "periodic",
                "interval": self._interval,
                "role": role,
                "insights": insights,
                "most_suspicious": player,
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
                "trigger": "periodic",
                "role": role,
                "error": str(e),
            })
            return ReflectionResult()
