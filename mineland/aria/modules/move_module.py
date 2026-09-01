"""Move Module — LLM-based navigation/exploration during explore phase.

Two-step process:
1. LLM decides target (player name or 'wander') based on context
2. AriaCodegen generates Mineflayer JS from the plan (~2KB prompt)
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from ..prompt_template import load_prompt
from ..env_adapter import AmongUsEnvAdapter, AmongUsGameState
from ..action.codegen import AriaCodegen
from ...sim.data.action import Action

if TYPE_CHECKING:
    from ..state.base import BaseStateBuilder
    from ..memory.base import BaseMemory


class MoveDecision(BaseModel):
    target_player: str = Field(description="Player name to move toward, or 'wander'.")
    reasoning: str = Field(description="Brief reasoning (1 sentence).")


class MoveModule:
    def __init__(
        self,
        llm,
        bot_name: str,
        role: str,
        prompt_style: str,
        save_path: str = "./storage",
        personal_message: Optional[str] = None,
        codegen_model: str = "gpt-4.1-mini",
        codegen_max_tokens: int = 1024,
        codegen_temperature: float = 0.2,
        api_base: Optional[str] = None,
    ):
        self.bot_name = bot_name
        self.role = role
        self.prompt_style = prompt_style
        self.personal_message = personal_message
        self._chain = llm | JsonOutputParser(pydantic_object=MoveDecision)

        self._codegen = AriaCodegen(
            model_name=codegen_model,
            max_tokens=codegen_max_tokens,
            temperature=codegen_temperature,
            api_base=api_base,
        )

    def run(
        self,
        raw_obs,
        game_state: AmongUsGameState,
        state_builder: "BaseStateBuilder",
        memory: "BaseMemory",
        verbose: bool = False,
        logger=None,
    ) -> Optional[dict]:
        """Decide where to move and generate navigation JS."""
        state_ctx = state_builder.summarize_for_context("move")
        memory_ctx = memory.get_context("move")

        template_name = f"explore_{self.role}"
        system_text = load_prompt(self.prompt_style, template_name)
        if not system_text:
            system_text = f"You are {self.bot_name}. Decide where to move."
        system_text = system_text.format(bot_name=self.bot_name)
        if self.personal_message:
            system_text += f"\n\n[PERSONAL_MESSAGE]: {self.personal_message}"

        human_content = f"Context:\n{state_ctx}\n{memory_ctx}\n\nDecide where to move."
        if game_state.rgb_base64:
            human_content = [
                {"type": "text", "text": human_content},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{game_state.rgb_base64}",
                    "detail": "low",
                }},
            ]

        try:
            result = self._chain.invoke([
                SystemMessage(content=system_text),
                HumanMessage(content=human_content),
            ])
            target = str(result.get("target_player", "wander") or "wander").strip() or "wander"
            reason = str(result.get("reasoning", "") or "").strip() or "Explore."

            memory.add_module_entry("move", {
                "tick": state_builder.get_tick(),
                "target": target,
                "reasoning": reason,
                "summary": f"move→{target}: {reason}",
            })

            if verbose:
                print(f"[MoveModule:{self.bot_name}] target={target} | {reason}")

            # Step 2: Generate JS via lightweight codegen
            plan_text = f"Navigate toward '{target}' safely. {reason}"
            obs_summary = state_ctx
            action = self._codegen.generate(
                plan_text, obs_summary, verbose=verbose,
                logger=logger, tick=state_builder.get_tick(),
                role=self.role,
            )
            if action.type == Action.NEW and action.code.strip():
                if logger:
                    logger.log_success("move_decision", state_builder.get_tick(), verbose,
                                       detail=f"target={target}")
                return {"code": action.code, "metadata": {"target": target}}

        except Exception as e:
            # Capture raw LLM output (LangChain OutputParserException stashes it
            # on .llm_output). Without this, we can't tell whether the model
            # returned empty/truncated/markdown-wrapped/refusal text.
            raw = getattr(e, "llm_output", None)
            if raw is None:
                raw = getattr(e, "send_to_llm", None)
            raw_repr = repr(raw)[:600] if raw is not None else "<no raw>"
            msg_repr = repr(str(e))[:400]
            print(f"[MoveModule:{self.bot_name}] PARSER_FAIL error_type={type(e).__name__} "
                  f"err_msg={msg_repr} raw_llm_output={raw_repr}")
            if logger:
                logger.log_failure("move_decision", state_builder.get_tick(),
                                   error=type(e).__name__, fallback="wander",
                                   detail=f"raw={raw_repr[:200]}")

        # Fallback: simple wander
        if game_state.own_position:
            js_code = AmongUsEnvAdapter.build_wander_js(
                game_state.own_position, role=self.role,
            )
            return {"code": js_code, "metadata": {"target": "wander"}}
        return None
