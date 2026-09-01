"""WindowMemory — rolling buffer of recent steps."""

from __future__ import annotations

import json
import os
from collections import deque
from typing import Dict, List

from .base import BaseMemory

MAX_MODULE_HISTORY = 20


class WindowMemory(BaseMemory):
    """Rolling window of last K (tick, action, state) entries."""

    def __init__(self, window_size: int = 10):
        self._buffer: deque = deque(maxlen=window_size)
        self._module_histories: Dict[str, deque] = {}
        self._reflections: deque = deque(maxlen=20)

    def update(self, tick, action_summary, state_summary, events):
        entry = {
            "tick": tick,
            "action": action_summary,
            "state": state_summary,
        }
        if events:
            entry["events"] = [e.get("message", str(e)) if isinstance(e, dict) else str(e) for e in events]
        self._buffer.append(entry)

    def get_context(self, module_name, max_entries=5):
        lines = []
        if self._buffer:
            recent = list(self._buffer)[-max_entries:]
            lines.append(f"[Recent history — last {len(recent)} steps]")
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
            "buffer": list(self._buffer),
            "module_histories": {k: list(v) for k, v in self._module_histories.items()},
            "reflections": list(self._reflections),
        }
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WindowMemory] save failed: {e}")

    def load_from_json(self, path):
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("buffer", []):
                self._buffer.append(item)
            for name, entries in data.get("module_histories", {}).items():
                self._module_histories[name] = deque(entries, maxlen=MAX_MODULE_HISTORY)
            for r in data.get("reflections", []):
                self._reflections.append(r)
            return True
        except Exception:
            return False
