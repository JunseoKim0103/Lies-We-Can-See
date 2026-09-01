"""Memory module factory for Aria agent."""

from .base import BaseMemory
from .none_memory import NoneMemory
from .window_memory import WindowMemory
from .episodic_memory import EpisodicMemory
from .semantic_memory import SemanticMemory


def create_memory(config) -> BaseMemory:
    if config.memory_mode == "none":
        return NoneMemory()
    elif config.memory_mode == "window":
        return WindowMemory(window_size=config.memory_window_size)
    elif config.memory_mode == "episodic":
        return EpisodicMemory()  # LLM injected later via set_llm()
    elif config.memory_mode == "semantic":
        return SemanticMemory(
            belief_update=config.memory_belief_update,
            role=config.role,
            bot_name=config.bot_name,
        )  # LLM injected later via set_llm()
    raise ValueError(f"Unknown memory_mode: {config.memory_mode}")
