"""EpisodicMemory — full history with LLM-based importance compression.

Stores all step entries in a large buffer. Periodically compresses
low-importance entries into summaries using LLM, keeping high-importance
entries intact.
"""

from __future__ import annotations

import json
import os
from collections import deque
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from .base import BaseMemory

MAX_BUFFER = 200
COMPRESS_THRESHOLD = 150     # trigger compression when buffer exceeds this
COMPRESS_TARGET = 80         # compress down to this many entries
MAX_MODULE_HISTORY = 30


class CompressionOutput(BaseModel):
    summary: str = Field(description="Compressed summary of the oldest entries (3-5 sentences).")
    important_indices: List[int] = Field(description="Indices (0-based) of entries too important to discard.")


_COMPRESS_PROMPT = """\
You are a memory compression system for an Among Us game agent.

Below are the oldest {count} entries from the agent's episodic memory.
Compress them into a short summary (3-5 sentences) that preserves the most
important information (key events, suspicion clues, strategic outcomes).

Also identify which entries (by 0-based index) are too important to lose
and should be kept individually.

ENTRIES:
{entries_text}

OUTPUT (JSON):
{{
  "summary": "<3-5 sentence compressed summary>",
  "important_indices": [<list of int indices to preserve>]
}}
"""


class EpisodicMemory(BaseMemory):
    """Full episodic history with periodic LLM-based compression."""

    def __init__(self, llm=None):
        self._buffer: List[dict] = []
        self._compressed_summaries: List[str] = []
        self._module_histories: Dict[str, deque] = {}
        self._reflections: deque = deque(maxlen=30)
        self._chain = None
        if llm is not None:
            self._chain = llm | JsonOutputParser(pydantic_object=CompressionOutput)

    def set_llm(self, llm) -> None:
        self._chain = llm | JsonOutputParser(pydantic_object=CompressionOutput)

    def update(self, tick, action_summary, state_summary, events):
        entry = {
            "tick": tick,
            "action": action_summary,
            "state": state_summary,
        }
        if events:
            entry["events"] = [e.get("message", str(e)) if isinstance(e, dict) else str(e) for e in events]
        self._buffer.append(entry)
        if len(self._buffer) > COMPRESS_THRESHOLD:
            self._compress()

    def _compress(self) -> None:
        """Compress oldest entries, keeping important ones."""
        n_to_compress = len(self._buffer) - COMPRESS_TARGET
        if n_to_compress <= 0:
            return

        oldest = self._buffer[:n_to_compress]

        if self._chain is not None:
            entries_text = "\n".join(
                f"[{i}] tick={e['tick']}: {e['action']}"
                for i, e in enumerate(oldest)
            )
            try:
                result = self._chain.invoke([
                    SystemMessage(content="You compress game agent memory."),
                    HumanMessage(content=_COMPRESS_PROMPT.format(
                        count=len(oldest), entries_text=entries_text,
                    )),
                ])
                summary = result.get("summary", "")
                important = set(result.get("important_indices", []))

                if summary:
                    self._compressed_summaries.append(summary)

                # Keep important entries, discard the rest
                preserved = [oldest[i] for i in sorted(important) if i < len(oldest)]
                self._buffer = preserved + self._buffer[n_to_compress:]
                return
            except Exception as exc:
                print(f"[EpisodicMemory] compression failed: {exc}")

        # Fallback: no LLM, just summarize mechanically
        ticks = [e["tick"] for e in oldest]
        self._compressed_summaries.append(
            f"[Compressed: ticks {ticks[0]}-{ticks[-1]}, {len(oldest)} entries]"
        )
        self._buffer = self._buffer[n_to_compress:]

    def get_context(self, module_name, max_entries=5):
        lines = []

        if self._compressed_summaries:
            lines.append("[Compressed history]")
            for s in self._compressed_summaries[-3:]:
                lines.append(f"  {s[:120]}")

        recent = self._buffer[-max_entries:]
        if recent:
            lines.append(f"\n[Recent history — last {len(recent)} steps]")
            for e in recent:
                line = f"  tick={e['tick']}: {e['action']}"
                if e.get('events'):
                    line += f" | events: {'; '.join(str(ev)[:60] for ev in e['events'][:3])}"
                lines.append(line)

        mod_hist = self._module_histories.get(module_name)
        if mod_hist:
            recent_mod = list(mod_hist)[-max_entries:]
            lines.append(f"\n[{module_name.upper()} — last {len(recent_mod)} decisions]")
            for e in recent_mod:
                lines.append(f"  tick={e.get('tick', '?')}: {e.get('summary', str(e))}")

        surv_hist = self._module_histories.get("surveillance")
        if surv_hist and module_name != "surveillance":
            recent_surv = list(surv_hist)[-3:]
            lines.append(f"\n[SURVEILLANCE — last {len(recent_surv)} observations]")
            for e in recent_surv:
                lines.append(f"  tick={e.get('tick', '?')}: {e.get('summary', str(e))}")

        if module_name == "planning":
            kill_hist = self._module_histories.get("kill")
            if kill_hist:
                defers = [e for e in kill_hist if str(e.get("policy", "")).startswith("defer")]
                if defers:
                    recent_def = defers[-max_entries:]
                    lines.append(f"\n[Recent kill DEFERs — last {len(recent_def)}]")
                    for e in recent_def:
                        lines.append(f"  tick={e.get('tick', '?')}: {e.get('summary', str(e))}")

        if self._reflections:
            lines.append(f"\n[Recent reflections]")
            for r in list(self._reflections)[-3:]:
                lines.append(f"  {r[:100]}")

        return "\n".join(lines)

    def add_reflection(self, reflection_text):
        self._reflections.append(reflection_text)

    def get_module_history(self, module_name, k=5):
        hist = self._module_histories.get(module_name)
        if not hist:
            return []
        return list(hist)[-k:]

    def add_module_entry(self, module_name, entry):
        if module_name not in self._module_histories:
            self._module_histories[module_name] = deque(maxlen=MAX_MODULE_HISTORY)
        self._module_histories[module_name].append(entry)

    def save_to_json(self, path):
        data = {
            "buffer": self._buffer,
            "compressed_summaries": self._compressed_summaries,
            "module_histories": {k: list(v) for k, v in self._module_histories.items()},
            "reflections": list(self._reflections),
        }
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[EpisodicMemory] save failed: {e}")

    def load_from_json(self, path):
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._buffer = data.get("buffer", [])
            self._compressed_summaries = data.get("compressed_summaries", [])
            for name, entries in data.get("module_histories", {}).items():
                self._module_histories[name] = deque(entries, maxlen=MAX_MODULE_HISTORY)
            for r in data.get("reflections", []):
                self._reflections.append(r)
            return True
        except Exception:
            return False
