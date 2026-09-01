"""Meeting Module — generates chat responses during Among Us meetings."""

from __future__ import annotations

import random
from typing import List, Optional, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from ..prompt_template import load_prompt
from ..env_adapter import AmongUsEnvAdapter

if TYPE_CHECKING:
    from ..state.base import BaseStateBuilder
    from ..memory.base import BaseMemory


class MeetingResponse(BaseModel):
    response_text: str = Field(description="Chat message. 1-2 sentences max.")
    suspicion_cast_on: str = Field(description="Player being blamed, or 'none'.")
    reasoning: str = Field(description="Internal reasoning (NOT spoken aloud).")


MAX_TURNS_PER_MEETING = 5


class MeetingModule:
    def __init__(
        self,
        llm,
        bot_name: str,
        role: str,
        prompt_style: str,
        personal_message: Optional[str] = None,
        all_players: Optional[list] = None,
    ):
        self.bot_name = bot_name
        self.role = role
        self.prompt_style = prompt_style
        self.personal_message = personal_message
        self.player_roster = self._build_player_roster(all_players)
        self._chain = llm | JsonOutputParser(pydantic_object=MeetingResponse)
        self._speech_count: int = 0
        self._last_meeting_count: int = 0

    def run(
        self,
        state_builder: "BaseStateBuilder",
        memory: "BaseMemory",
        verbose: bool = False,
    ) -> Optional[dict]:
        """Generate a meeting response.

        Returns dict with "code" (JS) and "metadata" (for memory), or None.
        """
        if not state_builder.can_talk:
            return None

        # Reset speech count on new meeting. Use _meeting_count (monotonic)
        # rather than meeting_start_index (which resets to -1 on vote_result
        # and would spuriously re-reset during the post-vote meeting tail).
        cur_meeting = getattr(state_builder, "_meeting_count", 0)
        if cur_meeting != self._last_meeting_count:
            self._speech_count = 0
            self._last_meeting_count = cur_meeting

        if self._speech_count >= MAX_TURNS_PER_MEETING:
            if verbose:
                print(f"[MeetingModule:{self.bot_name}] max turns reached")
            return None

        # Build context — include surveillance observations for meeting discussion
        state_ctx = state_builder.summarize_for_context("meeting", recent_chat_k=30)
        memory_ctx = memory.get_context("meeting")
        surveillance_history = memory.get_module_history("surveillance", k=10)
        surveillance_ctx = ""
        if surveillance_history:
            lines = ["[Your observations from explore phase]"]
            for entry in surveillance_history:
                lines.append(f"  - {entry.get('summary', entry.get('observation', ''))}")
            surveillance_ctx = "\n".join(lines)
        context = state_ctx
        if memory_ctx:
            context += f"\n{memory_ctx}"
        if surveillance_ctx:
            context += f"\n{surveillance_ctx}"

        # Load prompt template
        template_name = f"meeting_{self.role}"
        system_text = load_prompt(self.prompt_style, template_name)
        if not system_text:
            system_text = f"You are {self.bot_name}, a {self.role} in Among Us. Respond in the meeting."

        # Build dead player section (style-dependent)
        dead_section = ""
        if hasattr(state_builder, 'get_dead_player_names'):
            dead_names = state_builder.get_dead_player_names()
            if dead_names:
                dead_list = ', '.join(dead_names)
                victim_verb = 'is a VICTIM' if len(dead_names) == 1 else 'are VICTIMS'
                # Facts (both styles)
                killed_by = "an imposter" if len(dead_names) == 1 else "imposters"
                dead_section = (
                    f"DEAD: {dead_list}\n"
                    f"This meeting was called because a body was reported.\n"
                    f"{dead_list} {victim_verb} — killed by {killed_by}.\n"
                    f"Do NOT accuse dead players — they cannot be imposters.\n"
                    f"Do NOT address dead players — they cannot hear or respond."
                )
                # Strategy guidance (deterministic only)
                if self.prompt_style == "deterministic":
                    dead_section += (
                        f"\nFocus on: who was near the body? Who has a weak alibi? "
                        f"Who is deflecting questions?"
                    )

        system_text = system_text.format(
            bot_name=self.bot_name,
            player_roster=self.player_roster,
            dead_section=dead_section,
        )
        if self.personal_message:
            system_text += f"\n\n[PERSONAL_MESSAGE]: {self.personal_message}"

        human_text = f"{context}\n\nGenerate {self.bot_name}'s meeting response."

        try:
            result = self._chain.invoke([
                SystemMessage(content=system_text),
                HumanMessage(content=human_text),
            ])
            response_text = result.get("response_text", "").strip()
            suspicion_target = result.get("suspicion_cast_on", "none")
            reasoning = result.get("reasoning", "")

            if response_text:
                state_builder.chat_log.append(f"{self.bot_name}: {response_text}")
                self._speech_count += 1
                state_builder.speech_counts[self.bot_name] = self._speech_count

            if suspicion_target and suspicion_target.lower() != "none":
                state_builder.record_suspicion(suspicion_target, +0.15)

            memory.add_module_entry("meeting", {
                "tick": state_builder.get_tick(),
                "said": response_text,
                "blamed": suspicion_target,
                "summary": f'said:"{response_text[:60]}" blame={suspicion_target}',
            })

            js_code = AmongUsEnvAdapter.build_chat_js(response_text)
            return {"code": js_code, "metadata": {"said": response_text}}

        except Exception as e:
            if verbose:
                print(f"[MeetingModule:{self.bot_name}] LLM error: {e}")
            fallback = self._fallback_message(state_builder)
            self._speech_count += 1
            state_builder.speech_counts[self.bot_name] = self._speech_count
            js_code = AmongUsEnvAdapter.build_chat_js(fallback)
            return {"code": js_code, "metadata": {"said": fallback}}

    def _fallback_message(self, state_builder: "BaseStateBuilder") -> str:
        alive = state_builder.alive_player_names()
        if self.role == "imposter":
            options = [
                "I was doing my tasks the whole time.",
                "I didn't see anything suspicious near me.",
                "everyone needs to share where they were",
            ]
        else:
            options = [
                "can everyone share where they were?",
                "who found the body and where exactly?",
                "I was doing my tasks.",
            ]
        return random.choice(options)

    @staticmethod
    def _build_player_roster(all_players: Optional[list]) -> str:
        if not all_players:
            return ""
        lines = ["[All Players in This Game]"]
        for p in all_players:
            lines.append(f"  - {p['name']} (color: {p.get('color', 'unknown')})")
        return "\n".join(lines)
