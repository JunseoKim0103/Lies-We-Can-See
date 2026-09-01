"""Emergency Meeting Module — decide whether to call an emergency meeting.

policy="rule": never triggers (no rule for spontaneous meeting calls)
policy="vlm":  VLM evaluates context + RGB → should I call an emergency meeting?

Triggers: high suspicion, witnessed suspicious behavior, strategic timing.
Both crewmates and imposters can call emergency meetings.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from ..env_adapter import AmongUsEnvAdapter, AmongUsGameState

if TYPE_CHECKING:
    from ..state.base import BaseStateBuilder
    from ..memory.base import BaseMemory


class EmergencyDecision(BaseModel):
    should_call: bool = Field(description="True if emergency meeting should be called now.")
    reasoning: str = Field(description="Why call/not call (1-2 sentences).")


_VLM_SYSTEM_CREWMATE = """\
You are a CREWMATE in an Among Us-style Minecraft game.

You can call an EMERGENCY MEETING by pressing the button at the center of the map.
This is a powerful action — use it only when you have strong reasons:
- You witnessed suspicious behavior (player near a body, player following you)
- You have important information to share that can't wait
- You suspect someone strongly based on what you've seen

Do NOT call a meeting if:
- You have no evidence or observations to share
- A meeting was called recently (waste of time)
- You're in the middle of completing an important mission

Look at the RGB screenshot and your observations. Should you call an emergency meeting?

OUTPUT (JSON):
{{
  "should_call": true/false,
  "reasoning": "<1-2 sentences>"
}}
"""

_VLM_SYSTEM_IMPOSTER = """\
You are the IMPOSTER in an Among Us-style Minecraft game.

You can call an EMERGENCY MEETING by pressing the button at the center of the map.
Strategic reasons to call a meeting:
- Deflect suspicion by appearing proactive ("I'm calling this because I saw something")
- Interrupt crewmates who are close to completing missions
- Frame someone by claiming you witnessed suspicious behavior

Do NOT call a meeting if:
- You just killed someone (suspicion will fall on you)
- You have no believable story to tell
- Crewmates are nowhere near finishing missions

Look at the RGB screenshot and context. Should you call an emergency meeting?

OUTPUT (JSON):
{{
  "should_call": true/false,
  "reasoning": "<1-2 sentences>"
}}
"""


class EmergencyModule:
    def __init__(self, bot_name: str, role: str = "crewmate",
                 policy: str = "rule", vlm=None, use_vision: bool = True):
        self.bot_name = bot_name
        self.role = role
        self.policy = policy
        self.use_vision = use_vision
        self._chain = None
        if vlm is not None:
            self._chain = vlm | JsonOutputParser(pydantic_object=EmergencyDecision)

    def set_vlm(self, vlm) -> None:
        self._chain = vlm | JsonOutputParser(pydantic_object=EmergencyDecision)

    def run(
        self,
        game_state: AmongUsGameState,
        state_builder: "BaseStateBuilder",
        memory: "BaseMemory",
        verbose: bool = False,
    ) -> Optional[dict]:
        """Decide whether to call emergency meeting.

        Returns dict with JS code to navigate + press button, or None.
        """
        if self.policy != "vlm" or self._chain is None:
            return None  # rule mode: never call emergency meeting

        if game_state.phase != 0:
            return None  # can only call during explore phase

        return self._vlm_decide(game_state, state_builder, memory, verbose)

    def _vlm_decide(self, game_state, state_builder, memory, verbose):
        state_ctx = state_builder.summarize_for_context("emergency", recent_chat_k=5)
        memory_ctx = memory.get_context("emergency")

        system_text = (_VLM_SYSTEM_IMPOSTER if self.role == "imposter"
                       else _VLM_SYSTEM_CREWMATE)
        if not self.use_vision:
            system_text = (system_text
                .replace("Look at the RGB screenshot and your observations.",
                         "You have NO camera — reason from your observations and player positions below.")
                .replace("Look at the RGB screenshot and context.",
                         "You have NO camera — reason from the game state and player positions below."))

        human_parts = [
            {"type": "text", "text": (
                f"Context:\n{state_ctx}\n{memory_ctx}\n\n"
                f"Should you call an emergency meeting right now?"
            )},
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
            should_call = result.get("should_call", False)
            reasoning = result.get("reasoning", "")

            if verbose:
                print(f"[EmergencyModule:{self.bot_name}] VLM decision: "
                      f"call={should_call} | {reasoning}")

            memory.add_module_entry("emergency", {
                "tick": state_builder.get_tick(),
                "action": "call" if should_call else "skip",
                "reasoning": reasoning,
                "summary": f"EMERGENCY {'CALL' if should_call else 'SKIP'}: {reasoning}",
            })

            if should_call:
                js_code = AmongUsEnvAdapter.build_emergency_meeting_js()
                return {"code": js_code, "metadata": {"action": "emergency_meeting"}}
            return None

        except Exception as e:
            if verbose:
                print(f"[EmergencyModule:{self.bot_name}] VLM error: {e}")
            return None
