"""Aria LLM helpers — self-contained, no Steve dependency.

Supports OpenAI, OpenRouter-hosted models, Qwen 3.6 (vLLM), and Gemma 4 (vLLM).
Construct instances via ``create_chat_openai()``.
"""

from __future__ import annotations

import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI


# ── Model config ────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    name: str
    provider: str  # 'openai' | 'openrouter' | 'qwen' | 'gemma'
    supports_vision: bool = False
    default_temperature: float = 0.1
    input_price_per_1m: Optional[float] = None
    output_price_per_1m: Optional[float] = None
    openrouter_model: Optional[str] = None
    temperature_fixed: Optional[float] = None
    mask_score: Optional[float] = None
    ars_score: Optional[float] = None
    casi_score: Optional[float] = None
    api_base_env: Optional[str] = None
    # True if the model burns hidden reasoning tokens before visible content
    # (gpt-5 family, kimi-k2.5, qwen *-thinking, etc.). When True, max_tokens
    # is bumped to REASONING_COMPLETION_FLOOR so reasoning doesn't starve the
    # visible-output budget and return empty strings.
    is_reasoning: bool = False
    # OpenRouter provider blacklist. Sent as `extra_body.provider.ignore`.
    # Example: ['DeepInfra'] to avoid DeepInfra's vision-less `-turbo` variant
    # of gemma-4-31b-it. Only used when provider='openrouter'.
    openrouter_provider_ignore: Optional[List[str]] = None
    # Default reasoning_effort for the direct-OpenAI branch. The branch falls
    # back to "minimal", but not every reasoning model accepts that value —
    # gpt-6-astra only takes low/medium/high/xhigh — so set it per model.
    reasoning_effort: Optional[str] = None

MODEL_CONFIGS: Dict[str, ModelConfig] = {
    # OpenAI
    "gpt-4.1-nano": ModelConfig("gpt-4.1-nano", "openai", True, 0.1, 0.10, 0.40, mask_score=61.40, casi_score=54.05),
    "gpt-4.1-mini": ModelConfig("gpt-4.1-mini", "openai", True, 0.1, 0.40, 1.60, mask_score=50.00),
    "gpt-4.1": ModelConfig("gpt-4.1", "openai", True, 0.1, 2.00, 8.00, mask_score=51.13),
    "gpt-5-nano": ModelConfig("gpt-5-nano", "openai", True, 1.0, 0.05, 0.40, temperature_fixed=1.0, is_reasoning=True),
    "gpt-5-mini": ModelConfig("gpt-5-mini", "openai", True, 1.0, 0.25, 2.00, temperature_fixed=1.0, mask_score=82.60, ars_score=90.08, casi_score=85.80, is_reasoning=True),
    "gpt-5": ModelConfig("gpt-5", "openai", True, 1.0, 1.25, 10.00, temperature_fixed=1.0, mask_score=79.33, ars_score=81.88, casi_score=81.98, is_reasoning=True),
    "gpt-5.4-nano": ModelConfig("gpt-5.4-nano", "openai", True, 1.0, 0.20, 1.25, temperature_fixed=1.0, is_reasoning=True),
    "gpt-5.4-mini": ModelConfig("gpt-5.4-mini", "openai", True, 1.0, 0.75, 4.50, temperature_fixed=1.0, is_reasoning=True),
    "gpt-5.4": ModelConfig("gpt-5.4", "openai", True, 1.0, 2.50, 15.00, temperature_fixed=1.0, is_reasoning=True),
    "gpt-5.1": ModelConfig("gpt-5.1", "openai", True, 1.0, 2.00, 8.00, temperature_fixed=1.0, is_reasoning=True),

    # Gemini via OpenRouter
    "gemini-2.5-flash-lite": ModelConfig("google/gemini-2.5-flash-lite", "openrouter", True, 0.1, 0.10, 0.40, "google/gemini-2.5-flash-lite"),
    "gemini-2.5-flash": ModelConfig("google/gemini-2.5-flash", "openrouter", True, 0.1, 0.30, 2.50, "google/gemini-2.5-flash", mask_score=49.13),
    "gemini-2.5-pro": ModelConfig("google/gemini-2.5-pro", "openrouter", True, 0.1, 1.25, 10.00, "google/gemini-2.5-pro", mask_score=53.07),
    "gemini-3.1-flash-lite-preview": ModelConfig("google/gemini-3.1-flash-lite-preview", "openrouter", True, 0.1, 0.25, 1.50, "google/gemini-3.1-flash-lite-preview", mask_score=48.40),
    "gemini-3-flash-preview": ModelConfig("google/gemini-3-flash-preview", "openrouter", True, 0.1, None, None, "google/gemini-3-flash-preview"),
    "gemini-3.1-pro-preview": ModelConfig("google/gemini-3.1-pro-preview", "openrouter", True, 0.1, 2.00, 12.00, "google/gemini-3.1-pro-preview", mask_score=42.40),

    # Qwen aliases used by the requested sweep. OpenRouter is the default for
    # these aliases; legacy hyphenated local names are kept below.
    "qwen3.5-4b": ModelConfig("Qwen/Qwen3.5-4B", "openrouter", True, 0.7, api_base_env="QWEN35_4B_API_BASE"),
    "detection-rq2-0610": ModelConfig("detection-rq2-0610", "openrouter", True, 0.7, api_base_env="DETECTION_RQ2_API_BASE"),
    "detection-vote-0610": ModelConfig("detection-vote-0610", "openrouter", True, 0.7, api_base_env="DETECTION_VOTE_API_BASE"),
    "detection-votex1": ModelConfig("detection-votex1", "openrouter", True, 0.7, api_base_env="DETECTION_VOTEX1_API_BASE"),
    "detection-voteonly": ModelConfig("detection-voteonly", "openrouter", True, 0.7, api_base_env="DETECTION_VOTEONLY_API_BASE"),
    "detection-voteout": ModelConfig("detection-voteout", "openrouter", True, 0.7, api_base_env="DETECTION_VOTEOUT_API_BASE"),
    "detection-winvoteout": ModelConfig("detection-winvoteout", "openrouter", True, 0.7, api_base_env="DETECTION_WINVOTEOUT_API_BASE"),
    "detection-nowrongvote": ModelConfig("detection-nowrongvote", "openrouter", True, 0.7, api_base_env="DETECTION_NOWRONGVOTE_API_BASE"),
    "detection-voteshufx1": ModelConfig("detection-voteshufx1", "openrouter", True, 0.7, api_base_env="DETECTION_VOTESHUFX1_API_BASE"),
    "detection-voteshuf": ModelConfig("detection-voteshuf", "openrouter", True, 0.7, api_base_env="DETECTION_VOTESHUF_API_BASE"),
    "qwen3.5-9b": ModelConfig("qwen/qwen3.5-9b", "openrouter", True, 0.7, 0.04, 0.15, "qwen/qwen3.5-9b"),
    "qwen3.5-27b": ModelConfig("qwen/qwen3.5-27b", "openrouter", True, 0.7, 0.195, 1.56, "qwen/qwen3.5-27b"),
    "qwen3.6-27b": ModelConfig("qwen/qwen3.6-27b", "openrouter", True, 0.7, 0.32, 3.20, "qwen/qwen3.6-27b"),

    # Judge backbones. Separate keys from the agent aliases above because the
    # judge runs these with reasoning ON, while the agents run kimi and the
    # non-thinking qwen with reasoning OFF.
    "qwen3.6-27b-thinking": ModelConfig("qwen/qwen3.6-27b", "openrouter", True, 0.7, 0.32, 3.20, "qwen/qwen3.6-27b", is_reasoning=True),
    "kimi-k2.5-thinking": ModelConfig("moonshotai/kimi-k2.5", "openrouter", True, 0.1, 0.40, 1.90, "moonshotai/kimi-k2.5", is_reasoning=True),
    "glm-5.2": ModelConfig("z-ai/glm-5.2", "openrouter", False, 0.7, 0.35, 1.10, "z-ai/glm-5.2", is_reasoning=True),

    # Legacy local vLLM names
    "qwen-3.6-27b": ModelConfig("Qwen/Qwen3.6-27B", "qwen", True, 0.7, 0.32, 3.20),
    "qwen-3.6-27b-thinking": ModelConfig("Qwen/Qwen3.6-27B", "qwen", True, 0.7, 0.32, 3.20, is_reasoning=True),

    # Gemma and Kimi via OpenRouter, plus legacy local Gemma alias.
    # DeepInfra routes paid gemma-4-31b-it to a `-turbo` variant that rejects
    # image input (Aria's VLM modules send screenshots). Blacklist DeepInfra so
    # OpenRouter falls back to a multimodal-capable provider.
    "gemma4-31b": ModelConfig("google/gemma-4-31b-it", "openrouter", True, 0.7, 0.12, 0.37, "google/gemma-4-31b-it", openrouter_provider_ignore=["DeepInfra"]),
    "gemma-4-31b": ModelConfig("google/gemma-4-31B-it", "gemma", True, 0.7, 0.12, 0.37),
    "kimi-k2.5": ModelConfig("moonshotai/kimi-k2.5", "openrouter", True, 0.1, 0.40, 1.90, "moonshotai/kimi-k2.5", mask_score=70.47),

    # Gemma 3 via OpenRouter (added 2026-05-18). pricing needs confirmation
    # for 12b/27b — verify slug + price before billing-sensitive runs.
    "gemma3-4b": ModelConfig("google/gemma-3-4b-it", "openrouter", True, 0.7, 0.0, 0.0, "google/gemma-3-4b-it"),
    "gemma3-12b": ModelConfig("google/gemma-3-12b-it", "openrouter", True, 0.7, 0.04, 0.13, "google/gemma-3-12b-it"),
    "gemma3-27b": ModelConfig("google/gemma-3-27b-it", "openrouter", True, 0.7, 0.08, 0.16, "google/gemma-3-27b-it"),

    # NVIDIA Nemotron Nano Omni (vision+audio+text, reasoning variant, free tier).
    # is_reasoning=False so aria/llm.py sends reasoning.enabled=False at API level
    # — the reasoning variant should respect the toggle. Audio modality is OFF by
    # default (only sent if msg has audio content or modalities=['audio']).
    "nemotron-3-nano-omni-30b-a3b": ModelConfig("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "openrouter", True, 0.1, 0.0, 0.0, "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"),

    # Gemma 4 26B A4B (4B active params MoE, OpenRouter). Slug guess — verify
    # on OR page if 404.
    "gemma4-26b-a4b": ModelConfig("google/gemma-4-26b-a4b-it", "openrouter", True, 0.7, None, None, "google/gemma-4-26b-a4b-it"),

    # ── RQ2 model-pool extension (12 -> 14 models) ─────────────────────
    # GPT-6 Astra on the DIRECT OpenAI API (not OpenRouter). provider="openai"
    # plus the widened _is_openai_reasoning() predicate routes it down the same
    # branch gpt-5.x uses: reasoning_effort + max_completion_tokens floor +
    # temperature pinned to 1.0.
    #
    # Reasoning cannot be switched off for this model; every disable variant
    # returns 400 "Reasoning is mandatory for this endpoint and cannot be
    # disabled." The direct API also rejects "minimal":
    #   400 "reasoning_effort does not support 'minimal' with this model.
    #        Supported values are: 'low', 'medium', 'high', and 'xhigh'."
    # So this entry pins reasoning_effort="low", the lowest the direct API
    # takes. Measured: low burns 0 reasoning tokens on a belief-shaped prompt,
    # i.e. the same effective setting as gpt-5 / gpt-5-mini at minimal.
    #
    # The branch sends `model=<alias>`, so this key must equal OpenAI's model id.
    # Price: $10/M in, $50/M out; $20/$75 above 272k prompt tokens.
    "gpt-6-astra": ModelConfig("gpt-6-astra", "openai", True, 1.0, 10.00, 50.00, temperature_fixed=1.0, is_reasoning=True, reasoning_effort="low"),

    # Muse Spark 1.3 (Meta). Multimodal text+image+video+audio; Aria only ever
    # sends text+image.
    # is_reasoning MUST stay True. Like gpt-6-astra, this endpoint refuses to
    # run without reasoning (400 "Reasoning is mandatory ..."). is_reasoning=True
    # sends effort=minimal + exclude=True and restores the 32768 completion floor.
    # Unlike gpt-6-astra (0 reasoning tokens at minimal), this model always
    # thinks. Measured on one belief-shaped prompt:
    #   effort=minimal -> 239 reasoning tok of 288 completion (~83% hidden)
    #   effort=low     -> 454 of 504
    #   effort=medium  -> 636 of 685
    # minimal IS the floor: effort="none" / reasoning.enabled=False -> 400, and
    # reasoning.max_tokens=N is not honoured (it used more reasoning tokens than
    # effort=minimal). Output stayed valid JSON of the same length in every
    # variant.
    # Price: $1.25/M in, $4.25/M out.
    "muse-spark-1.3": ModelConfig("meta/muse-spark-1.3", "openrouter", True, 0.1, 1.25, 4.25, "meta/muse-spark-1.3", is_reasoning=True),
}

def get_model_config(model_name: str) -> Optional[ModelConfig]:
    return MODEL_CONFIGS.get(model_name)


def _find_model_config(model_name: str) -> Optional[ModelConfig]:
    cfg = MODEL_CONFIGS.get(model_name)
    if cfg is not None:
        return cfg
    for key, candidate in MODEL_CONFIGS.items():
        names = {key, candidate.name}
        if candidate.openrouter_model:
            names.add(candidate.openrouter_model)
        if model_name in names or model_name.startswith(key) or key.startswith(model_name):
            return candidate
    return None


# ── Default server URLs (round-robin) ──────────────────────────────────
# LIVE as of 2026-05-11:
#   Qwen3.6-27B : localhost:8000, localhost:8000, localhost:9000, localhost:8001,
#                 localhost:8003, localhost:7000, localhost:9000, localhost:8002
#   Gemma-4-31B : localhost:8000

DEFAULT_QWEN36_URLS = [
    "http://localhost:8000/v1",
    "http://localhost:8000/v1",
    "http://localhost:9000/v1",
    "http://localhost:8001/v1",
    "http://localhost:8003/v1",
    "http://localhost:7000/v1",
    "http://localhost:9000/v1",
    "http://localhost:8002/v1",
]

DEFAULT_GEMMA4_URLS = [
    "http://localhost:8000/v1",
]

# Floor on max_tokens for models marked is_reasoning=True. Hidden reasoning
# tokens (effort=low can still burn 500-1500) must not starve the visible-
# content budget — otherwise the model returns an empty completion and the
# JsonOutputParser raises "Invalid json output: ".
REASONING_COMPLETION_FLOOR = 32768

# Back-compat alias; old name kept for any external imports.
MIN_REASONING_COMPLETION_TOKENS = REASONING_COMPLETION_FLOOR


def _reasoning_disabled() -> bool:
    """Global override: when MINELAND_DISABLE_REASONING is set, every model is
    treated as non-reasoning — no max_tokens floor, no thinking directive sent
    upstream. Useful for fast/cheap baselining or when reasoning behavior is
    suspected of causing failures."""
    return os.getenv("MINELAND_DISABLE_REASONING", "").lower() in ("1", "true", "yes", "on")


def _effective_is_reasoning(cfg: Optional["ModelConfig"]) -> bool:
    if _reasoning_disabled():
        return False
    return bool(cfg and cfg.is_reasoning)


# ── Round-robin URL selection ──────────────────────────────────────────

_url_counter = 0
_url_counter_lock = threading.Lock()


def _select_url(urls: List[str]) -> str:
    global _url_counter
    if not urls:
        raise ValueError("No URLs available")
    with _url_counter_lock:
        idx = _url_counter % len(urls)
        _url_counter += 1
    return urls[idx]


def _parse_multi_urls(env_value: str, defaults: List[str]) -> List[str]:
    if not env_value:
        return defaults
    urls = [u.strip() for u in env_value.split(",") if u.strip()]
    return urls or defaults


# ── Token tracker ──────────────────────────────────────────────────────

_agent_local = threading.local()


def set_current_agent(agent_name: str) -> None:
    _agent_local.agent_name = agent_name


def get_current_agent() -> str:
    return getattr(_agent_local, "agent_name", "unknown")


def set_current_step(step: Optional[int]) -> None:
    _agent_local.step = step


def get_current_step() -> Optional[int]:
    return getattr(_agent_local, "step", None)


class _TokenCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._usage: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"input": 0, "output": 0, "calls": 0}
        )
        self._agent_usage: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"input": 0, "output": 0, "calls": 0})
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage = (response.llm_output or {}).get("token_usage", {})
        if not usage:
            return
        model = (response.llm_output or {}).get("model_name", "unknown")
        inp = int(usage.get("prompt_tokens", 0))
        out = int(usage.get("completion_tokens", 0))
        agent = get_current_agent()
        with self._lock:
            self._usage[model]["input"] += inp
            self._usage[model]["output"] += out
            self._usage[model]["calls"] += 1
            self._agent_usage[agent][model]["input"] += inp
            self._agent_usage[agent][model]["output"] += out
            self._agent_usage[agent][model]["calls"] += 1

    def get_usage(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            return {k: dict(v) for k, v in self._usage.items()}

    def get_agent_usage(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        with self._lock:
            return {
                a: {m: dict(c) for m, c in ms.items()}
                for a, ms in self._agent_usage.items()
            }

    def reset(self) -> None:
        with self._lock:
            self._usage.clear()
            self._agent_usage.clear()


_token_tracker = _TokenCallbackHandler()


def get_token_tracker() -> _TokenCallbackHandler:
    return _token_tracker


def get_token_summary(show_cost: bool = True) -> str:
    usage = _token_tracker.get_usage()
    if not usage:
        return "No token usage recorded."

    lines = ["=" * 60, "  TOKEN USAGE SUMMARY", "=" * 60]
    total_input = total_output = total_cost = 0.0
    has_cost = False

    for model_name, counts in sorted(usage.items()):
        inp, out, calls = counts["input"], counts["output"], counts["calls"]
        total_input += inp
        total_output += out
        cfg = _find_model_config(model_name)
        if cfg and cfg.input_price_per_1m is not None:
            cost = (inp * cfg.input_price_per_1m + out * cfg.output_price_per_1m) / 1_000_000
            total_cost += cost
            has_cost = True
            lines.append(
                f"  {model_name}\n"
                f"    calls={calls}  input={inp:,}  output={out:,}  total={inp+out:,}\n"
                f"    cost=${cost:.4f}"
            )
        else:
            lines.append(
                f"  {model_name}\n"
                f"    calls={calls}  input={inp:,}  output={out:,}  total={inp+out:,}\n"
                f"    cost=N/A"
            )

    lines.append("-" * 60)
    lines.append(f"  TOTAL  input={int(total_input):,}  output={int(total_output):,}")
    if has_cost:
        lines.append(f"  TOTAL COST: ${total_cost:.4f}")

    agent_usage = _token_tracker.get_agent_usage()
    if agent_usage:
        lines.append("=" * 60)
        lines.append("  PER-AGENT COST BREAKDOWN")
        lines.append("=" * 60)
        for agent_name in sorted(agent_usage.keys()):
            models = agent_usage[agent_name]
            agent_inp = agent_out = agent_cost = 0
            agent_has_cost = False
            for mn, c in models.items():
                agent_inp += c["input"]
                agent_out += c["output"]
                cfg = _find_model_config(mn)
                if cfg and cfg.input_price_per_1m is not None:
                    agent_cost += (c["input"] * cfg.input_price_per_1m + c["output"] * cfg.output_price_per_1m) / 1_000_000
                    agent_has_cost = True
            cost_str = f"${agent_cost:.4f}" if agent_has_cost else "N/A"
            lines.append(f"  [{agent_name}]  input={agent_inp:,}  output={agent_out:,}  cost={cost_str}")

    lines.append("=" * 60)
    return "\n".join(lines)


# ── Provider detection ─────────────────────────────────────────────────

def _is_qwen(name: str) -> bool:
    return str(name).startswith("qwen-")

def _is_gemma(name: str) -> bool:
    return str(name).startswith("gemma-")

def _is_openai_reasoning(name: str) -> bool:
    """OpenAI reasoning families served on the direct API (gpt-5*, gpt-6*).

    They share one contract: reasoning_effort, max_completion_tokens instead of
    max_tokens, and temperature fixed at 1.0.
    """
    n = str(name)
    return n.startswith("gpt-5") or n.startswith("gpt-6")


# Back-compat alias; the predicate covered only gpt-5 before the pool extension.
_is_gpt5 = _is_openai_reasoning

def _is_openrouter(name: str, cfg: Optional[ModelConfig]) -> bool:
    return bool(cfg and cfg.provider == "openrouter")

def _supports_thinking(name: str) -> bool:
    return "thinking" in name.lower()


def _inject_token_tracker(init: Dict[str, Any]) -> None:
    existing = list(init.get("callbacks") or [])
    if _token_tracker not in existing:
        existing.append(_token_tracker)
    # Per-call prompt/response tracing (only when ARIA_TRACE_DIR is set).
    from .trace_logger import get_trace_handler
    tracer = get_trace_handler()
    if tracer is not None and tracer not in existing:
        existing.append(tracer)
    init["callbacks"] = existing


# ── Main factory ───────────────────────────────────────────────────────

def create_chat_openai(
    *,
    model_name: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    response_format: Optional[Any] = None,
    reasoning_effort: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    api_base: Optional[str] = None,
    reasoning_floor: bool = True,
    **kwargs: Any,
) -> ChatOpenAI:
    """Create a ChatOpenAI instance for any supported provider.

    `reasoning_floor` raises the completion budget for reasoning models so
    hidden thinking cannot starve the visible answer. Agents want that. Callers
    that need the budget honoured exactly as given pass False.
    """

    merged_model_kwargs: Dict[str, Any] = dict(model_kwargs or {})

    if api_base:
        kwargs["base_url"] = api_base

    if response_format is not None:
        merged_model_kwargs.setdefault("response_format", response_format)

    cfg = get_model_config(model_name)

    # ── vLLM common: retry on busy servers ───────────────────────
    _VLLM_MAX_RETRIES = 5
    _VLLM_TIMEOUT = 120  # seconds

    # ── OpenRouter ───────────────────────────────────────────────
    if _is_openrouter(model_name, cfg):
        if "base_url" not in kwargs and cfg.api_base_env:
            api_base_from_env = os.getenv(cfg.api_base_env)
            if api_base_from_env:
                kwargs["base_url"] = _select_url(_parse_multi_urls(api_base_from_env, []))

        if not cfg.openrouter_model and "base_url" not in kwargs:
            raise ValueError(
                f"{model_name} is registered, but no OpenRouter model id is known. "
                f"Pass api_base, set {cfg.api_base_env or 'a custom API_BASE env var'}, "
                "or update ModelConfig.openrouter_model."
            )

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key and "base_url" in kwargs:
            api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"
        if not api_key:
            raise ValueError(
                f"{model_name} uses OpenRouter. Set OPENROUTER_API_KEY, "
                "or pass api_base for a custom OpenAI-compatible endpoint."
            )
        kwargs.setdefault("api_key", api_key)
        if "base_url" not in kwargs:
            kwargs["base_url"] = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

        actual_name = cfg.openrouter_model or cfg.name
        resolved_temp = cfg.temperature_fixed
        if resolved_temp is None:
            resolved_temp = temperature if temperature is not None else cfg.default_temperature

        extra = merged_model_kwargs.get("extra_body", {})
        reasoning = extra.get("reasoning", {})
        if cfg and cfg.is_reasoning:
            # Model wants reasoning ON — minimize budget but keep enabled.
            reasoning.setdefault("effort", "minimal")
            reasoning.setdefault("exclude", True)
        else:
            # Reasoning explicitly OFF for non-reasoning OpenRouter models
            # (kimi-k2.5 default, qwen3.5/3.6 non-thinking, etc.). Avoids
            # phantom thinking-mode that returns empty completions.
            reasoning["enabled"] = False
        extra["reasoning"] = reasoning
        if cfg and cfg.openrouter_provider_ignore:
            provider_cfg = extra.get("provider", {})
            existing = provider_cfg.get("ignore", [])
            provider_cfg["ignore"] = list({*existing, *cfg.openrouter_provider_ignore})
            extra["provider"] = provider_cfg
        merged_model_kwargs["extra_body"] = extra

        init: Dict[str, Any] = {
            "model": actual_name,
            "max_retries": _VLLM_MAX_RETRIES,
            "request_timeout": _VLLM_TIMEOUT,
            **kwargs,
        }
        # Reasoning models burn hidden tokens before visible content; floor
        # ensures visible budget remains even with small caller-supplied max.
        floor = REASONING_COMPLETION_FLOOR if (reasoning_floor and cfg and cfg.is_reasoning) else 0
        if max_tokens is not None:
            init["max_tokens"] = max(max_tokens, floor)
        elif floor:
            init["max_tokens"] = floor
        if resolved_temp is not None:
            init["temperature"] = resolved_temp
        if merged_model_kwargs:
            init["model_kwargs"] = merged_model_kwargs
        _inject_token_tracker(init)
        return ChatOpenAI(**init)

    # ── Qwen (vLLM or OpenRouter) ─────────────────────────────────
    if _is_qwen(model_name) or (cfg and cfg.provider == "qwen"):
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        use_openrouter = bool(openrouter_key)

        if use_openrouter:
            kwargs.setdefault("api_key", openrouter_key)
            if "base_url" not in kwargs:
                kwargs["base_url"] = os.getenv(
                    "OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"
                )
            # Map e.g. "qwen-3.6-27b" -> "qwen/qwen3.6-27b"
            suffix = model_name[len("qwen-"):] if model_name.startswith("qwen-") else model_name
            actual_name = f"qwen/qwen{suffix}"
        else:
            urls = _parse_multi_urls(
                os.getenv("QWEN36_27B_API_BASE", ""), DEFAULT_QWEN36_URLS
            )
            api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "EMPTY"
            kwargs.setdefault("api_key", api_key)

            if "base_url" not in kwargs:
                kwargs["base_url"] = _select_url(urls)

            actual_name = cfg.name if cfg else model_name

        # Disable thinking/reasoning unless the model is marked reasoning. The
        # legacy name-based heuristic (`_supports_thinking`) is kept only as a
        # fallback for unknown model names that lack a ModelConfig.
        wants_reasoning = (cfg.is_reasoning if cfg is not None
                           else _supports_thinking(model_name))
        if not wants_reasoning:
            extra = merged_model_kwargs.get("extra_body", {})
            if use_openrouter:
                extra.setdefault("reasoning", {"enabled": False})
            else:
                extra.setdefault("chat_template_kwargs", {"enable_thinking": False})
            merged_model_kwargs["extra_body"] = extra
        else:
            # Reasoning ON for qwen-style models: ensure effort directive is
            # present so OpenRouter knows to allocate thinking budget.
            extra = merged_model_kwargs.get("extra_body", {})
            if use_openrouter:
                reasoning = extra.get("reasoning", {})
                reasoning.setdefault("effort", "low")
                reasoning.setdefault("exclude", True)
                reasoning.setdefault("enabled", True)
                extra["reasoning"] = reasoning
            else:
                extra.setdefault("chat_template_kwargs", {"enable_thinking": True})
            merged_model_kwargs["extra_body"] = extra

        resolved_temp = temperature if temperature is not None else (cfg.default_temperature if cfg else None)

        init: Dict[str, Any] = {
            "model": actual_name,
            "max_retries": _VLLM_MAX_RETRIES,
            "request_timeout": _VLLM_TIMEOUT,
            **kwargs,
        }
        floor = REASONING_COMPLETION_FLOOR if (reasoning_floor and cfg and cfg.is_reasoning) else 0
        if max_tokens is not None:
            init["max_tokens"] = max(max_tokens, floor)
        elif floor:
            init["max_tokens"] = floor
        if resolved_temp is not None:
            init["temperature"] = resolved_temp
        if merged_model_kwargs:
            init["model_kwargs"] = merged_model_kwargs
        _inject_token_tracker(init)
        return ChatOpenAI(**init)

    # ── Gemma (vLLM) ───────────────────────────────────────────────
    if _is_gemma(model_name) or (cfg and cfg.provider == "gemma"):
        urls = _parse_multi_urls(
            os.getenv("GEMMA4_27B_API_BASE", ""), DEFAULT_GEMMA4_URLS
        )
        api_key = os.getenv("GEMMA_API_KEY") or "EMPTY"
        kwargs.setdefault("api_key", api_key)

        if "base_url" not in kwargs:
            kwargs["base_url"] = _select_url(urls)

        actual_name = cfg.name if cfg else model_name
        resolved_temp = temperature if temperature is not None else (cfg.default_temperature if cfg else None)

        init: Dict[str, Any] = {
            "model": actual_name,
            "max_retries": _VLLM_MAX_RETRIES,
            "request_timeout": _VLLM_TIMEOUT,
            **kwargs,
        }
        if max_tokens is not None:
            init["max_tokens"] = max_tokens
        if resolved_temp is not None:
            init["temperature"] = resolved_temp
        if merged_model_kwargs:
            init["model_kwargs"] = merged_model_kwargs
        _inject_token_tracker(init)
        return ChatOpenAI(**init)

    # ── OpenAI reasoning models: gpt-5*, gpt-6* ────────────────────
    if _is_openai_reasoning(model_name):
        effort = (
            reasoning_effort
            or merged_model_kwargs.get("reasoning_effort")
            or os.getenv("MINELAND_REASONING_EFFORT")
            or (cfg.reasoning_effort if cfg else None)
            or "minimal"
        )
        # NOTE: MINELAND_REASONING_EFFORT outranks the per-model value, so
        # setting it to "minimal" globally will 400 on models that reject that
        # level (gpt-6-astra). Leave it unset unless every model in play
        # accepts the value.
        merged_model_kwargs.setdefault("reasoning_effort", effort)
        # GPT-5 family is always reasoning; apply floor even when caller didn't
        # pass max_tokens (otherwise reasoning can exhaust the default budget
        # and return empty visible content).
        completion_budget = max(max_tokens or 0, REASONING_COMPLETION_FLOOR)
        merged_model_kwargs.setdefault("max_completion_tokens", completion_budget)

        init: Dict[str, Any] = {"model": model_name, "temperature": 1.0, **kwargs}
        if merged_model_kwargs:
            init["model_kwargs"] = merged_model_kwargs
        _inject_token_tracker(init)
        return ChatOpenAI(**init)

    # ── GPT-4 / others ─────────────────────────────────────────────
    init: Dict[str, Any] = {"model": model_name, **kwargs}
    if max_tokens is not None:
        init["max_tokens"] = max_tokens
    if temperature is not None:
        init["temperature"] = temperature
    if merged_model_kwargs:
        init["model_kwargs"] = merged_model_kwargs
    _inject_token_tracker(init)
    return ChatOpenAI(**init)
