"""Vote Module — decides who to vote for during Among Us meetings."""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from ..prompt_template import load_prompt
from ..env_adapter import AmongUsEnvAdapter, AmongUsGameState

if TYPE_CHECKING:
    from ..state.base import BaseStateBuilder
    from ..memory.base import BaseMemory


class VoteDecision(BaseModel):
    vote_target: str = Field(description="Exact player name or 'skip'.")
    reasoning: str = Field(description="Internal reasoning for this vote.")


class VoteModule:
    def __init__(
        self,
        llm,
        bot_name: str,
        role: str,
        prompt_style: str,
        player_id_map: dict,
        personal_message: Optional[str] = None,
        all_players: Optional[list] = None,
        teammate: Optional[str] = None,
    ):
        self.bot_name = bot_name
        self.role = role
        self.prompt_style = prompt_style
        self.player_id_map = player_id_map
        self.personal_message = personal_message
        self.teammate = teammate
        self.player_roster = self._build_player_roster(all_players)
        self._chain = llm | JsonOutputParser(pydantic_object=VoteDecision)
        self._has_voted: bool = False
        self._last_vote_target: Optional[str] = None
        self._last_meeting_count: int = 0

    def run(
        self,
        game_state: AmongUsGameState,
        state_builder: "BaseStateBuilder",
        memory: "BaseMemory",
        verbose: bool = False,
    ) -> Optional[dict]:
        """Decide vote target and return JS action."""
        # Reset on new meeting. Use _meeting_count (monotonic, increments
        # exactly once per meeting_start) rather than meeting_start_index
        # (which resets to -1 on vote_result, causing spurious double-vote
        # because the change re-triggers the reset between vote_result inject
        # and server cleanup).
        cur_meeting = getattr(state_builder, "_meeting_count", 0)
        if cur_meeting != self._last_meeting_count:
            self._has_voted = False
            self._last_vote_target = None
            self._last_meeting_count = cur_meeting

        if self._has_voted:
            if verbose:
                print(f"[VoteModule:{self.bot_name}] already voted this meeting → skip")
            return None

        alive = game_state.alive_votable_players
        if verbose:
            print(f"[VoteModule:{self.bot_name}] alive_votable={alive}")
        if not alive:
            alive = state_builder.alive_player_names()
            if verbose:
                print(f"[VoteModule:{self.bot_name}] fallback alive_player_names={alive}")
        if not alive:
            alive = [n for n in self.player_id_map if n != self.bot_name]
            if verbose:
                print(f"[VoteModule:{self.bot_name}] fallback player_id_map={alive}")
        if not alive:
            if verbose:
                print(f"[VoteModule:{self.bot_name}] no alive players → skip vote")
            return None

        state_ctx = state_builder.summarize_for_context("vote", recent_chat_k=30)
        memory_ctx = memory.get_context("vote")
        surveillance_history = memory.get_module_history("surveillance", k=10)
        surveillance_ctx = ""
        if surveillance_history:
            lines = ["[Your observations from explore phase]"]
            for entry in surveillance_history:
                lines.append(f"  - {entry.get('summary', entry.get('observation', ''))}")
            surveillance_ctx = "\n".join(lines)
        if surveillance_ctx:
            memory_ctx = f"{memory_ctx}\n{surveillance_ctx}" if memory_ctx else surveillance_ctx

        template_name = f"vote_{self.role}"
        system_text = load_prompt(self.prompt_style, template_name)
        if not system_text:
            system_text = f"You are {self.bot_name}. Vote for the imposter or skip."
        system_text = system_text.format(
            bot_name=self.bot_name,
            alive_players=", ".join(alive),
            player_roster=self.player_roster,
        )
        if self.personal_message:
            system_text += f"\n\n[PERSONAL_MESSAGE]: {self.personal_message}"
        if self.role == "imposter" and self.teammate:
            system_text += (
                f"\n\n[TEAMMATE]: Your teammate imposter is {self.teammate}. "
                f"Default: protect {self.teammate} and vote a crewmate or skip. "
                f"Advanced: a deliberate vote against {self.teammate} can be a "
                f"deception play (sacrifice to earn trust) — only use this when "
                f"{self.teammate} is already very likely to be ejected and you "
                f"need to clear your own suspicion. State your reasoning explicitly."
            )

        human_text = (
            f"{state_ctx}\n{memory_ctx}\n\n"
            f"Alive players: {', '.join(alive)}\n"
            f"Choose {self.bot_name}'s vote target."
        )

        vote_target = None
        reasoning = ""
        try:
            result = self._chain.invoke([
                SystemMessage(content=system_text),
                HumanMessage(content=human_text),
            ])
            candidate = result.get("vote_target", "").strip()
            reasoning = result.get("reasoning", "")

            if candidate.lower() == "skip" and self.role == "crewmate":
                vote_target = "skip"
            elif candidate in alive and candidate != self.bot_name:
                vote_target = candidate
        except Exception as e:
            if verbose:
                print(f"[VoteModule:{self.bot_name}] LLM error: {e}")

        # Fallback — defensive: exclude teammate so LLM errors don't cause
        # accidental team-kill votes. Intentional sacrifice must come from LLM.
        fallback_pool = (
            [n for n in alive if n != self.teammate]
            if (self.role == "imposter" and self.teammate)
            else alive
        )
        if vote_target is None:
            vote_target = self._rule_based_pick(state_builder, fallback_pool)
        if vote_target is None and self.role == "imposter":
            vote_target = fallback_pool[0] if fallback_pool else "skip"
        if vote_target is None:
            vote_target = "skip"

        memory.add_module_entry("vote", {
            "tick": state_builder.get_tick(),
            "voted_for": vote_target,
            "reasoning": reasoning or "rule-based fallback",
            "summary": f"voted→{vote_target}",
        })

        self._has_voted = True
        self._last_vote_target = vote_target
        js_code = AmongUsEnvAdapter.build_vote_js(vote_target, self.player_id_map)
        print(f"[VoteModule:{self.bot_name}] VOTE → {vote_target} (reason: {reasoning[:80]})")
        return {"code": js_code, "metadata": {"voted_for": vote_target}}

    def _rule_based_pick(
        self, state_builder: "BaseStateBuilder", alive: List[str]
    ) -> Optional[str]:
        exclude = {self.bot_name}
        if self.role == "imposter" and self.teammate:
            exclude.add(self.teammate)
        candidates = {
            n: state_builder.suspicion.get(n, 0.0)
            for n in alive if n not in exclude
        }
        if candidates:
            best = max(candidates, key=lambda n: candidates[n])
            if candidates[best] > 0:
                return best
        if self.role == "crewmate":
            return None
        others = [n for n in alive if n not in exclude]
        return others[0] if others else None

    @staticmethod
    def _build_player_roster(all_players: Optional[list]) -> str:
        if not all_players:
            return ""
        lines = ["[All Players in This Game]"]
        for p in all_players:
            lines.append(f"  - {p['name']} (color: {p.get('color', 'unknown')})")
        return "\n".join(lines)
