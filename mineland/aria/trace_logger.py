"""Per-call LLM input/output tracing for Aria agents.

A LangChain ``BaseCallbackHandler`` that dumps every model call (the full
system + human messages it receives, and the raw completion it returns) to a
separate JSON file. Images embedded as base64 data URLs are written to a
sibling ``images/`` folder and replaced by a relative-path reference, so the
JSON stays small and readable.

Activation is purely env-driven so no call sites change:

    ARIA_TRACE_DIR=/path/to/trace   # enables tracing; one file per LLM call

One JSON file per agent, with steps (ticks) as keys, each holding the ordered
list of every LLM call that agent made on that step:

    trace/
      James.json
      Olivia.json
      images/
        James_s0042_0.jpeg   # <agent>_s<step>_<idx>

    James.json:
    {
      "agent": "James",
      "steps": {
        "7": [
          {
            "seq": 42, "module": "kill_module", "model": "gpt-5-mini",
            "input": [
              {"role": "system", "content": "..."},
              {"role": "human", "content": [
                  {"type": "text", "text": "..."},
                  {"type": "image_ref", "path": "images/James_s0007_0.jpg", "bytes": 51234}
              ]}
            ],
            "output": "{...raw completion...}",
            "tokens": {"input": 1234, "output": 210},
            "error": null
          },
          { ... next call this step (planner, codegen, ...) ... }
        ],
        "8": [ ... ]
      }
    }
"""
from __future__ import annotations

import base64
import inspect
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# Source subpackages whose file stem is a meaningful "module" label.
_MODULE_DIRS = ("modules", "planner", "reflection", "memory", "action")


def _infer_module() -> str:
    """Best-effort module label from the call stack (e.g. 'kill_module')."""
    for frame in inspect.stack():
        path = frame.filename.replace("\\", "/")
        if "/mineland/aria/" in path and any(
            f"/aria/{d}/" in path for d in _MODULE_DIRS
        ):
            return Path(path).stem
    return "unknown"


# Per-agent output directories. The Aria agent registers its storage folder
# at init so each agent's trace.json + images/ land alongside its memory/state
# JSON (e.g. storage/<session>/<agent>/) instead of one shared trace dir.
_AGENT_DIRS: Dict[str, str] = {}
_AGENT_DIRS_LOCK = threading.Lock()


def register_agent_trace_dir(agent: str, path: str) -> None:
    """Map an agent name to the directory its trace output should go to."""
    with _AGENT_DIRS_LOCK:
        _AGENT_DIRS[str(agent)] = str(path)


def _serialize_image(image_url: Dict[str, Any], images_dir: Path,
                     name_stem: str) -> Dict[str, Any]:
    """Write a base64 data-URL image to images_dir as ``<name_stem>.<ext>``.

    `name_stem` is e.g. ``James_s0042_0`` — agent, step, and per-step index —
    so images are browsable by step instead of opaque content hashes.
    """
    url = image_url.get("url", "") if isinstance(image_url, dict) else ""
    detail = image_url.get("detail") if isinstance(image_url, dict) else None
    if not url.startswith("data:"):
        # Already a remote/relative URL — keep as-is.
        return {"type": "image_ref", "url": url, "detail": detail}
    try:
        header, b64 = url.split(",", 1)
        ext = "jpg"
        if "image/" in header:
            ext = header.split("image/", 1)[1].split(";", 1)[0] or "jpg"
        raw = base64.b64decode(b64)
        images_dir.mkdir(parents=True, exist_ok=True)
        img_path = images_dir / f"{name_stem}.{ext}"
        img_path.write_bytes(raw)
        return {
            "type": "image_ref",
            "path": f"images/{img_path.name}",
            "bytes": len(raw),
            "detail": detail,
        }
    except Exception as e:  # never let logging break a real run
        return {"type": "image_ref", "error": f"decode failed: {e}"}


def _serialize_content(content: Any, images_dir: Path, next_stem) -> Any:
    """Serialize message content, swapping base64 images for path refs.

    `next_stem` is a no-arg callable returning the next image filename stem.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                out.append(_serialize_image(
                    part.get("image_url", {}), images_dir, next_stem()))
            else:
                out.append(part)
        return out
    return content


class PromptTraceHandler(BaseCallbackHandler):
    """Groups each LLM call by agent → step into per-step JSON files."""

    def __init__(self, out_dir: Path):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.images_dir = self.out_dir / "images"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = 0
        self._pending: Dict[str, Dict[str, Any]] = {}
        # agent -> {step_label -> [call, ...]}. One JSON file per agent is
        # rewritten in full on each call, with steps as keys.
        self._agents: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        # (agent, step_label) -> count of images saved so far, for naming.
        self._img_counter: Dict[tuple, int] = {}

    @staticmethod
    def _step_label(step: Optional[int]) -> str:
        return str(step) if isinstance(step, int) else "unknown"

    def _make_namer(self, agent: str, step: Optional[int]):
        """Return a no-arg callable yielding `<agent>_s<step>_<idx>` stems."""
        step_str = f"{step:04d}" if isinstance(step, int) else "xxxx"
        key = (agent, self._step_label(step))

        def next_stem() -> str:
            with self._lock:
                idx = self._img_counter.get(key, 0)
                self._img_counter[key] = idx + 1
            return f"s{step_str}_{idx}"

        return next_stem

    def _agent_dir(self, agent: str) -> Path:
        """Output dir for this agent — its registered storage folder, else a
        per-agent subdir of out_dir as fallback."""
        with _AGENT_DIRS_LOCK:
            reg = _AGENT_DIRS.get(agent)
        return Path(reg) if reg else (self.out_dir / agent)

    # ── input capture ────────────────────────────────────────────
    def on_chat_model_start(self, serialized, messages, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        # Imported lazily to avoid an import cycle with llm.py.
        from .llm import get_current_agent, get_current_step
        agent = get_current_agent()
        step = get_current_step()
        namer = self._make_namer(agent, step)
        images_dir = self._agent_dir(agent) / "images"
        batch = messages[0] if messages else []
        serial: List[Dict[str, Any]] = []
        for msg in batch:
            role = getattr(msg, "type", msg.__class__.__name__)
            serial.append({
                "role": role,
                "content": _serialize_content(
                    getattr(msg, "content", ""), images_dir, namer
                ),
            })
        self._stash(run_id, agent, step, serial)

    def on_llm_start(self, serialized, prompts, **kwargs: Any) -> None:
        # Fallback for completion-style models (prompts are plain strings).
        run_id = str(kwargs.get("run_id", ""))
        from .llm import get_current_agent, get_current_step
        serial = [{"role": "prompt", "content": p} for p in (prompts or [])]
        self._stash(run_id, get_current_agent(), get_current_step(), serial)

    def _stash(self, run_id: str, agent: str, step: Optional[int],
               serial: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._pending[run_id] = {
                "agent": agent,
                "step": step,
                "module": _infer_module(),
                "input": serial,
                "ts": time.time(),
            }

    # ── output capture ───────────────────────────────────────────
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        outputs: List[str] = []
        for gen_list in (response.generations or []):
            for gen in gen_list:
                outputs.append(getattr(gen, "text", "") or "")
        usage = (response.llm_output or {}).get("token_usage", {}) or {}
        model = (response.llm_output or {}).get("model_name", "unknown")
        self._flush(run_id, model, "\n".join(outputs), usage, error=None)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        self._flush(run_id, "unknown", "", {}, error=repr(error))

    def _flush(self, run_id: str, model: str, output: str,
               usage: Dict[str, Any], error: Optional[str]) -> None:
        with self._lock:
            rec = self._pending.pop(run_id, None)
            if rec is None:
                rec = {"agent": "unknown", "step": None, "module": "unknown",
                       "input": []}
            seq = self._seq
            self._seq += 1

            entry = {
                "seq": seq,
                "module": rec["module"],
                "model": model,
                "input": rec["input"],
                "output": output,
                "tokens": {
                    "input": int(usage.get("prompt_tokens", 0) or 0),
                    "output": int(usage.get("completion_tokens", 0) or 0),
                },
                "error": error,
            }

            agent = str(rec["agent"]).replace("/", "_") or "unknown"
            step = rec["step"]
            step_label = str(step) if isinstance(step, int) else "unknown"
            steps = self._agents.setdefault(agent, {})
            steps.setdefault(step_label, []).append(entry)
            snapshot = {"agent": agent, "steps": dict(steps)}

        try:
            agent_dir = self._agent_dir(agent)
            agent_dir.mkdir(parents=True, exist_ok=True)
            (agent_dir / "trace.json").write_text(
                json.dumps(snapshot, indent=2, ensure_ascii=False, default=str)
            )
        except Exception:
            pass  # tracing must never crash a real run


_handler: Optional[PromptTraceHandler] = None
_handler_lock = threading.Lock()


def get_trace_handler() -> Optional[PromptTraceHandler]:
    """Return the singleton trace handler if ARIA_TRACE_DIR is set, else None."""
    global _handler
    out_dir = os.getenv("ARIA_TRACE_DIR")
    if not out_dir:
        return None
    with _handler_lock:
        if _handler is None or str(_handler.out_dir) != out_dir:
            _handler = PromptTraceHandler(Path(out_dir))
        return _handler
