"""
Aria structured logger for LLM/VLM call tracking.

ALWAYS logs failures (not gated by verbose).
Logs successes only when verbose=True.

Output format (grep-friendly):
  [ARIA_FAIL] bot=James module=kill step=7 error=JSONDecodeError fallback=rule
  [ARIA_OK]   bot=James module=codegen step=7 tokens=245
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LLMCallRecord:
    module: str
    bot_name: str
    tick: int
    success: bool
    fallback: Optional[str] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None


class AriaLogger:
    """Centralized LLM/VLM call logger. One instance per Aria agent."""

    def __init__(self, bot_name: str):
        self.bot_name = bot_name
        self._records: List[LLMCallRecord] = []
        # Running counters
        self.total_calls: int = 0
        self.total_failures: int = 0
        self.failures_by_module: Dict[str, int] = {}
        self.calls_by_module: Dict[str, int] = {}

    def log_success(self, module: str, tick: int, verbose: bool = False,
                    detail: str = "", latency_ms: float = None) -> None:
        """Log a successful LLM/VLM call. Prints only if verbose."""
        self.total_calls += 1
        self.calls_by_module[module] = self.calls_by_module.get(module, 0) + 1
        self._records.append(LLMCallRecord(
            module=module, bot_name=self.bot_name, tick=tick,
            success=True, latency_ms=latency_ms,
        ))
        if verbose:
            lat = f" ({latency_ms:.0f}ms)" if latency_ms else ""
            print(f"[ARIA_OK]   bot={self.bot_name} module={module} tick={tick}{lat}"
                  + (f" {detail}" if detail else ""))

    def log_failure(self, module: str, tick: int, error: str,
                    fallback: str = "none", detail: str = "") -> None:
        """Log a failed LLM/VLM call. ALWAYS prints (not gated by verbose)."""
        self.total_calls += 1
        self.total_failures += 1
        self.calls_by_module[module] = self.calls_by_module.get(module, 0) + 1
        self.failures_by_module[module] = self.failures_by_module.get(module, 0) + 1
        self._records.append(LLMCallRecord(
            module=module, bot_name=self.bot_name, tick=tick,
            success=False, error=error, fallback=fallback,
        ))
        print(f"[ARIA_FAIL] bot={self.bot_name} module={module} tick={tick} "
              f"error={error} fallback={fallback}"
              + (f" {detail}" if detail else ""))

    def log_placeholder(self, module: str, tick: int, fallback: str = "wander") -> None:
        """Log when LLM returned a placeholder/template instead of real code."""
        self.log_failure(module, tick, error="placeholder_response", fallback=fallback)

    def summary(self) -> str:
        """Return a summary string for end-of-game logging."""
        if self.total_calls == 0:
            return f"[ARIA_SUMMARY] bot={self.bot_name} no LLM calls"
        fail_rate = self.total_failures / self.total_calls * 100
        lines = [
            f"[ARIA_SUMMARY] bot={self.bot_name} "
            f"calls={self.total_calls} failures={self.total_failures} "
            f"fail_rate={fail_rate:.1f}%",
        ]
        for module in sorted(self.calls_by_module):
            calls = self.calls_by_module[module]
            fails = self.failures_by_module.get(module, 0)
            rate = fails / calls * 100 if calls else 0
            lines.append(
                f"  {module:20s}: {calls:3d} calls, {fails:3d} fails ({rate:.0f}%)"
            )
        return "\n".join(lines)
