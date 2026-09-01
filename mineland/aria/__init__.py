"""
Aria — Config-driven ablation agent for Among Us (MineLand).

Usage::

    from mineland.aria import Aria, AriaConfig
    from mineland.aria.config import steve_equivalent, baseline, full

    # Custom config
    agent = Aria(AriaConfig(
        role="imposter",
        state_mode="privileged",
        memory_mode="semantic",
        planning_mode="shortterm",
        reflection_mode="meeting",
        prompt_style="minimal",
        bot_name="MyBot",
    ))

    # Preset: Steve-equivalent
    agent = Aria(steve_equivalent(role="imposter", bot_name="Steve"))

    # Drop-in usage (same as Steve/Alex)
    action = agent.run(obs, code_info, done, task_info)
"""

from .aria_agent import Aria
from .config import AriaConfig, steve_equivalent, baseline, full

__all__ = ["Aria", "AriaConfig", "steve_equivalent", "baseline", "full"]
