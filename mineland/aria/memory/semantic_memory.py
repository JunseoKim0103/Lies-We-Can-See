"""SemanticMemory — structured belief state with player tracking.

belief_update modes:
- "rule": mechanical updates (suspicion managed by state_builder)
- "llm":  LLM evaluates events and updates player beliefs each step
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from .base import BaseMemory

MAX_MODULE_HISTORY = 30


class BeliefUpdateOutput(BaseModel):
    updates: List[dict] = Field(
        description="List of {player, suspicion_delta, note} dicts."
    )


_BELIEF_PROMPT = """\
You are a belief-tracking system for an Among Us game agent ({role}).

Given recent events, update your beliefs about each player.
For each player where your suspicion should change, provide an update.

CURRENT BELIEFS:
{beliefs_text}

RECENT EVENTS:
{events_text}

OUTPUT (JSON):
{{
  "updates": [
    {{"player": "<name>", "suspicion_delta": <float -0.3 to +0.3>, "note": "<reason>"}}
  ]
}}
If no updates needed, return {{"updates": []}}.
"""


class SemanticMemory(BaseMemory):
    """Structured beliefs: player suspicion, alibis, meeting summaries."""

    def __init__(self, belief_update: str = "rule", role: str = "crewmate",
                 bot_name: str = "", llm=None):
        self.belief_update = belief_update
        self.role = role
        self.bot_name = bot_name
        self.player_beliefs: Dict[str, dict] = {}
        self.meeting_summaries: List[dict] = []
        self._module_histories: Dict[str, list] = {}
        self._reflections: List[str] = []
        self._pending_events: List[dict] = []
        self._meeting_belief_pending: bool = False
        self._chain = None
        if llm is not None:
            self._chain = llm | JsonOutputParser(pydantic_object=BeliefUpdateOutput)

    def set_llm(self, llm) -> None:
        self._chain = llm | JsonOutputParser(pydantic_object=BeliefUpdateOutput)

    def update(self, tick, action_summary, state_summary, events):
        # Only buffer events when LLM belief-update is active. In rule mode the
        # buffer is never drained (no _llm_update_beliefs call), so accumulating
        # would leak memory across the whole game.
        if events and self.belief_update == "llm" and self._chain is not None:
            self._pending_events.extend(events)
        # Belief update is deferred to aria_agent's post-reflection hook during
        # meetings (avoids double LLM call). Only auto-fire here for non-meeting
        # events (e.g., surveillance findings).
        if self.belief_update == "llm" and self._chain is not None and self._pending_events:
            if not self._meeting_belief_pending:
                self._llm_update_beliefs()

    def _llm_update_beliefs(self) -> None:
        """Use LLM to update player beliefs from accumulated events."""
        beliefs_text = "none" if not self.player_beliefs else "\n".join(
            f"  {name}: {info}" for name, info in self.player_beliefs.items()
        )
        events_text = "\n".join(
            f"  tick={e.get('tick', '?')}: {e.get('message', str(e))}"
            for e in self._pending_events[-20:]
        )

        try:
            result = self._chain.invoke([
                SystemMessage(content="You track beliefs for an Among Us agent."),
                HumanMessage(content=_BELIEF_PROMPT.format(
                    role=self.role, beliefs_text=beliefs_text, events_text=events_text,
                )),
            ])
            for upd in result.get("updates", []):
                player = upd.get("player", "")
                delta = float(upd.get("suspicion_delta", 0))
                note = upd.get("note", "")
                if player and player != self.bot_name:
                    if player not in self.player_beliefs:
                        self.player_beliefs[player] = {"suspicion": 0.0, "notes": []}
                    cur = self.player_beliefs[player].get("suspicion", 0.0)
                    self.player_beliefs[player]["suspicion"] = max(0.0, min(1.0, cur + delta))
                    if note:
                        notes = self.player_beliefs[player].get("notes", [])
                        notes.append(note)
                        if len(notes) > 10:
                            notes.pop(0)
                        self.player_beliefs[player]["notes"] = notes
        except Exception as exc:
            raw = getattr(exc, "llm_output", None)
            if raw is None:
                raw = getattr(exc, "send_to_llm", None)
            raw_repr = repr(raw)[:600] if raw is not None else "<no raw>"
            print(f"[SemanticMem:{self.bot_name}] belief_update FAILED: "
                  f"err_type={type(exc).__name__} err_msg={repr(str(exc))[:300]} "
                  f"raw_llm_output={raw_repr}")
        finally:
            self._pending_events.clear()

    def get_context(self, module_name, max_entries=5):
        lines = []

        if self.player_beliefs:
            lines.append("[Player Beliefs]")
            for name, beliefs in self.player_beliefs.items():
                sus = beliefs.get("suspicion", 0.0)
                notes = beliefs.get("notes", [])
                last_note = notes[-1] if notes else ""
                line = f"  {name}: suspicion={sus:.2f}"
                if last_note:
                    line += f" | {last_note[:60]}"
                lines.append(line)

        if self.meeting_summaries:
            lines.append(f"\n[Meeting Summaries ({len(self.meeting_summaries)})]")
            for s in self.meeting_summaries[-3:]:
                if s.get("skipped"):
                    lines.append(f"  Meeting {s.get('meeting_no', '?')}: skipped (no ejection)")
                    continue
                ejected = s.get("ejected") or "?"
                role = s.get("ejected_role")
                role_tag = f" ({role})" if role else ""
                lines.append(f"  Meeting {s.get('meeting_no', '?')}: ejected={ejected}{role_tag}")

        mod_hist = self._module_histories.get(module_name, [])
        if mod_hist:
            recent = mod_hist[-max_entries:]
            lines.append(f"\n[{module_name.upper()} — last {len(recent)} decisions]")
            for e in recent:
                lines.append(f"  tick={e.get('tick', '?')}: {e.get('summary', str(e))}")

        # When the planner is the consumer, surface recent kill DEFERs so it
        # can adapt: "deferred because too far / witness behind" → don't
        # repeat the same target/decision next step.
        if module_name == "planning":
            kill_hist = self._module_histories.get("kill", [])
            defers = [e for e in kill_hist
                      if str(e.get("policy", "")).startswith("defer")]
            if defers:
                recent = defers[-max_entries:]
                lines.append(f"\n[Recent kill DEFERs — last {len(recent)}]")
                for e in recent:
                    lines.append(f"  tick={e.get('tick', '?')}: {e.get('summary', str(e))}")

        surv_hist = self._module_histories.get("surveillance", [])
        if surv_hist and module_name != "surveillance":
            recent_surv = surv_hist[-3:]
            lines.append(f"\n[SURVEILLANCE — last {len(recent_surv)} observations]")
            for e in recent_surv:
                lines.append(f"  tick={e.get('tick', '?')}: {e.get('summary', str(e))}")

        if self._reflections:
            lines.append(f"\n[Recent reflections]")
            for r in self._reflections[-3:]:
                lines.append(f"  {r[:100]}")

        return "\n".join(lines)

    def add_reflection(self, reflection_text):
        self._reflections.append(reflection_text)
        if len(self._reflections) > 30:
            self._reflections.pop(0)

    def get_module_history(self, module_name, k=5):
        return self._module_histories.get(module_name, [])[-k:]

    def add_module_entry(self, module_name, entry):
        if module_name not in self._module_histories:
            self._module_histories[module_name] = []
        self._module_histories[module_name].append(entry)
        if len(self._module_histories[module_name]) > MAX_MODULE_HISTORY:
            self._module_histories[module_name].pop(0)

    def save_to_json(self, path):
        data = {
            "player_beliefs": self.player_beliefs,
            "meeting_summaries": self.meeting_summaries,
            "module_histories": self._module_histories,
            "reflections": self._reflections,
        }
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[SemanticMemory] save failed: {e}")

    def load_from_json(self, path):
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.player_beliefs = data.get("player_beliefs", {})
            self.meeting_summaries = data.get("meeting_summaries", [])
            self._module_histories = data.get("module_histories", {})
            self._reflections = data.get("reflections", [])
            return True
        except Exception:
            return False
