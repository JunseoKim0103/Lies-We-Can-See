"""Mission Module — LLM-based mission selection during explore phase.

Two-step: LLM selects mission → AriaCodegen generates JS.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from ..env_adapter import AmongUsEnvAdapter, AmongUsGameState
from ..action.codegen import AriaCodegen
from ...sim.data.action import Action

if TYPE_CHECKING:
    from ..state.base import BaseStateBuilder
    from ..memory.base import BaseMemory


def _lookup_mission_coords(personal_message: str, mission_id: int):
    """Parse '<id>: (x, y, z)' from personal_message Mission Location section."""
    if not personal_message or mission_id is None:
        return None
    pattern = rf'^\s+{mission_id}:\s+\((-?\d+),\s*(-?\d+),\s*(-?\d+)\)'
    for line in personal_message.split('\n'):
        m = re.match(pattern, line)
        if m:
            return tuple(int(x) for x in m.groups())
    return None


class MissionDecision(BaseModel):
    mission_key: str = Field(description="'mission1'-'mission20' or 'none'.")
    objective: str = Field(description="One-line objective description.")


class MissionModule:
    def __init__(
        self,
        llm,
        bot_name: str,
        role: str,
        save_path: str = "./storage",
        personal_message: Optional[str] = None,
        codegen_model: str = "gpt-4.1-mini",
        codegen_max_tokens: int = 1024,
        codegen_temperature: float = 0.2,
        api_base: Optional[str] = None,
        coords_lookup: bool = True,
    ):
        self.bot_name = bot_name
        self.role = role
        self.personal_message = personal_message
        # Toggle for the deterministic coord override; see config.py
        # `mission_coords_lookup` for rationale.
        self.coords_lookup = coords_lookup
        self._chain = llm | JsonOutputParser(pydantic_object=MissionDecision)

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
        """Select a mission and generate action JS."""
        pending = state_builder.pending_missions()
        done = state_builder.done_missions()
        state_ctx = state_builder.summarize_for_context("mission")
        memory_ctx = memory.get_context("mission")

        system_text = (
            f"You are {self.bot_name}, a {self.role} in Among Us.\n"
            f"Select a pending mission to work on.\n"
            f"Pending: {', '.join(pending) if pending else 'none'}\n"
            f"Done: {', '.join(done) if done else 'none'}\n\n"
            f"CRITICAL — coordinate accuracy:\n"
            f"  * The objective field MUST contain the EXACT (x, y, z) coords for the\n"
            f"    chosen mission_key, taken from the 'Mission Location' section in your\n"
            f"    personal message.\n"
            f"  * Look up the number AFTER 'mission' in mission_key (e.g. mission2 → look\n"
            f"    up the line starting with '2:'), and copy that line's (x, y, z) verbatim.\n"
            f"  * NEVER substitute coords from a different mission. Do NOT invent or\n"
            f"    average coords. Mismatched coords cause the bot to walk to the wrong\n"
            f"    spot and the mission silently fails.\n\n"
            f"Example: if mission_key='mission2' and your Mission Location section says\n"
            f"  '2: (101, 65, 89)', then objective MUST be exactly:\n"
            f"  'Proceed to mission2 at coordinates (101, 65, 89) to complete the task.'\n\n"
            f"OUTPUT (JSON):\n"
            f'{{"mission_key": "<mission key or none>", "objective": "<1 line including (x, y, z)>"}}'
        )
        if self.personal_message:
            system_text += f"\n\n[PERSONAL_MESSAGE]: {self.personal_message}"

        human_text = f"{state_ctx}\n{memory_ctx}\n\nSelect a mission."

        try:
            result = self._chain.invoke([
                SystemMessage(content=system_text),
                HumanMessage(content=human_text),
            ])
            mission_key = result.get("mission_key", "none")
            # Coerce to str: some models (e.g. gemma) occasionally return a
            # non-string objective (int/None), which broke `objective[:60]`
            # below with "'int' object is not subscriptable".
            objective = str(result.get("objective") or "")

            if mission_key not in pending and mission_key != "none":
                mission_key = pending[0] if pending else "none"
        except Exception as e:
            if logger:
                logger.log_failure("mission_decision", state_builder.get_tick(),
                                   error=type(e).__name__, fallback="first_pending")
            mission_key = pending[0] if pending else "none"
            objective = f"Work on {mission_key}"

        memory.add_module_entry("mission", {
            "tick": state_builder.get_tick(),
            "mission": mission_key,
            "objective": objective,
            "summary": f"mission={mission_key}: {objective[:60]}",
        })

        if verbose:
            print(f"[MissionModule:{self.bot_name}] mission={mission_key} | {objective}")

        # Force-inject correct coordinates from personal_message to bypass
        # LLM coordinate hallucination (LLM may pick correct mission_key but
        # generate wrong (x,y,z) in `objective` text, sending the bot to the
        # wrong location and stalling the game). Lookup is deterministic so
        # the bot navigates to the actual mission position regardless of LLM
        # text quality. mission_key strategic choice still belongs to LLM.
        # Disable via config.mission_coords_lookup=False for ablations that
        # measure the LLM's lookup / instruction-following capability.
        if self.coords_lookup:
            mission_id = None
            if mission_key.startswith("mission"):
                try:
                    mission_id = int(mission_key.replace("mission", ""))
                except ValueError:
                    pass
            coords = _lookup_mission_coords(self.personal_message or "", mission_id)
            if coords:
                objective = (f"Proceed to {mission_key} at coordinates "
                             f"({coords[0]}, {coords[1]}, {coords[2]}) to complete the task.")
                if verbose:
                    print(f"[MissionModule:{self.bot_name}] coords lookup: {mission_key} → {coords}")

        # Generate JS via lightweight codegen.
        # personal_message is intentionally NOT injected here: codegen is a
        # plain plan→JS transformer. Decision-stage LLM (line 78-79) already
        # consumes personal_message and encodes deception intent into
        # `objective` text, which carries through plan_text below. Re-injecting
        # personal_message would duplicate the deception signal and risk
        # conflicting interpretations. Same convention as MoveModule.
        plan_text = (
            f"Complete {mission_key}. {objective}. Find the mission location and press the button. "
            f"IMPORTANT: After pressing the button, you MUST remain completely stationary for "
            f"90 seconds (1800 ticks = 1 minute 30 seconds). During this stay period, do NOT move, "
            f"take other actions, or interact with anything — any movement cancels the mission. "
            f"You will see a chat message 'Mission {mission_key.upper()} Completed' when finished. "
            f"Only move if you are in immediate danger (e.g., visible imposter approaching)."
        )
        obs_summary = state_ctx
        tick = state_builder.get_tick()
        action = self._codegen.generate(
            plan_text, obs_summary, verbose=verbose, logger=logger, tick=tick,
            role=self.role,
        )
        if action.type == Action.NEW and action.code.strip():
            if logger:
                logger.log_success("mission_codegen", tick, verbose,
                                   detail=f"mission={mission_key}")
            return {"code": action.code, "metadata": {"mission": mission_key}}

        # Fallback
        if game_state.own_position:
            js_code = AmongUsEnvAdapter.build_wander_js(
                game_state.own_position, role=self.role,
            )
            return {"code": js_code, "metadata": {"mission": mission_key}}
        return None
