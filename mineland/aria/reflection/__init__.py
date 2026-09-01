"""Reflection module factory for Aria agent."""

from .base import BaseReflector, ReflectionResult
from .none_reflector import NoneReflector
from .post_action_reflector import PostActionReflector
from .periodic_reflector import PeriodicReflector
from .meeting_reflector import MeetingReflector


def create_reflector(config) -> BaseReflector:
    if config.reflection_mode == "none":
        return NoneReflector()
    elif config.reflection_mode == "post_action":
        return PostActionReflector()
    elif config.reflection_mode == "periodic":
        return PeriodicReflector(interval=config.reflection_interval)
    elif config.reflection_mode == "meeting":
        return MeetingReflector()
    raise ValueError(f"Unknown reflection_mode: {config.reflection_mode}")
