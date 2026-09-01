"""Report Module — detect and report corpses.

policy="rule": nearest corpse → approach → report (no LLM)
policy="vlm":  RGB + context → VLM judges whether to report, delay, or ignore
               (imposter might strategically NOT report their own kill)
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


class ReportDecision(BaseModel):
    should_report: bool = Field(description="True to report now, False to ignore/delay.")
    reasoning: str = Field(description="Why report/ignore (1-2 sentences).")


_VLM_SYSTEM_CREWMATE = """\
You are a CREWMATE in an Among Us-style Minecraft game.
A corpse has been detected nearby.

Look at the RGB screenshot and context. Decide:
- Should you REPORT immediately? (usually yes for crewmates)
- Is it safe to approach? (anyone suspicious nearby?)

OUTPUT (JSON):
{{
  "should_report": true/false,
  "reasoning": "<1-2 sentences>"
}}
"""

_VLM_SYSTEM_IMPOSTER = """\
You are the IMPOSTER in an Among Us-style Minecraft game.
A corpse has been detected nearby. It might be YOUR kill.

Look at the RGB screenshot and context. Decide strategically:
- REPORT: self-reporting can deflect suspicion ("I found the body!")
- IGNORE: walk away and let someone else find it
- Consider: are other players nearby who might see you near the body?
- Consider: have you been accused before? Self-reporting might look suspicious.

OUTPUT (JSON):
{{
  "should_report": true/false,
  "reasoning": "<1-2 sentences>"
}}
"""


class ReportModule:
    def __init__(self, bot_name: str, role: str = "crewmate",
                 policy: str = "rule", vlm=None, use_vision: bool = True):
        self.bot_name = bot_name
        self.role = role
        self.policy = policy
        self.use_vision = use_vision
        self._chain = None
        if vlm is not None:
            self._chain = vlm | JsonOutputParser(pydantic_object=ReportDecision)

    def set_vlm(self, vlm) -> None:
        self._chain = vlm | JsonOutputParser(pydantic_object=ReportDecision)

    # Sentinel: VLM strategically decided NOT to report (distinct from failure)
    _DEFER = "DEFER"

    def run(
        self,
        game_state: AmongUsGameState,
        state_builder: "BaseStateBuilder",
        memory: "BaseMemory",
        verbose: bool = False,
    ) -> Optional[dict]:
        """Generate report action or decide to skip."""
        if self.policy == "vlm" and self._chain is not None:
            result = self._vlm_report(game_state, state_builder, memory, verbose)
            if result == self._DEFER:
                # VLM strategically said "don't report" → respect it
                if verbose:
                    print(f"[ReportModule:{self.bot_name}] VLM deferred report — skipping")
                return {"code": "", "metadata": {"action": "vlm_deferred"}}
            if result is not None:
                return result
            # result is None → VLM call failed → rule fallback (safety net)

        return self._rule_report(game_state, state_builder, memory, verbose)

    def _vlm_report(
        self, game_state, state_builder, memory, verbose
    ) -> Optional[dict]:
        """VLM-based report: judge whether to report from RGB + context."""
        state_ctx = state_builder.summarize_for_context("report", recent_chat_k=5)
        memory_ctx = memory.get_context("report")

        system_text = (_VLM_SYSTEM_IMPOSTER if self.role == "imposter"
                       else _VLM_SYSTEM_CREWMATE)
        if not self.use_vision:
            system_text = system_text.replace(
                "Look at the RGB screenshot and context.",
                "You have NO camera — reason from the game state and player positions below.")

        human_parts = [
            {"type": "text", "text": (
                f"Context:\n{state_ctx}\n{memory_ctx}\n\n"
                f"A corpse is nearby. Decide whether to report."
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
            should_report = result.get("should_report", True)
            reasoning = result.get("reasoning", "")

            if verbose:
                print(f"[ReportModule:{self.bot_name}] VLM decision: "
                      f"report={should_report} | {reasoning}")

            memory.add_module_entry("report", {
                "tick": state_builder.get_tick(),
                "action": "report" if should_report else "ignore",
                "reasoning": reasoning,
                "policy": "vlm",
                "summary": f"VLM {'REPORT' if should_report else 'IGNORE'}: {reasoning}",
            })

            if should_report:
                js_code = AmongUsEnvAdapter.build_report_js()
                return {"code": js_code, "metadata": {"action": "report"}}
            else:
                return self._DEFER  # VLM strategically chose not to report

        except Exception as e:
            if verbose:
                print(f"[ReportModule:{self.bot_name}] VLM error: {e}")
            return None  # LLM failure → rule fallback

    def _rule_report(
        self, game_state, state_builder, memory, verbose
    ) -> Optional[dict]:
        """Rule-based report: always report nearest corpse."""
        if verbose:
            print(f"[ReportModule:{self.bot_name}] rule: report")

        memory.add_module_entry("report", {
            "tick": state_builder.get_tick(),
            "action": "report_attempt",
            "policy": "rule",
            "summary": "REPORT: approach+report nearest corpse",
        })

        js_code = AmongUsEnvAdapter.build_report_js()
        return {"code": js_code, "metadata": {"action": "report"}}
