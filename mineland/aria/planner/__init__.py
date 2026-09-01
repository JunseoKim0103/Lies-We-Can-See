"""Planner factory for Aria agent."""

from .base import BasePlanner, PlanResult
from .reactive_planner import ReactivePlanner
from .shortterm_planner import ShorttermPlanner
from .hierarchical_planner import HierarchicalPlanner


def create_planner(config) -> BasePlanner:
    if config.planning_mode == "reactive":
        return ReactivePlanner(
            policy=config.planning_policy, role=config.role,
            bot_name=config.bot_name,
        )
    elif config.planning_mode == "shortterm":
        return ShorttermPlanner(
            policy=config.planning_policy, role=config.role,
            bot_name=config.bot_name,
        )
    elif config.planning_mode == "hierarchical":
        return HierarchicalPlanner(
            role=config.role,
            bot_name=config.bot_name,
            longterm_interval=config.longterm_plan_interval,
        )
    raise ValueError(f"Unknown planning_mode: {config.planning_mode}")
