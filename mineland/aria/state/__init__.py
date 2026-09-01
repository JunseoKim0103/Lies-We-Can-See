"""State builder factory for Aria agent."""

from .base import BaseStateBuilder
from .egocentric import EgocentricStateBuilder
from .privileged import PrivilegedStateBuilder


def create_state_builder(config) -> BaseStateBuilder:
    """Instantiate the right state builder based on config.state_mode."""
    common = dict(
        bot_name=config.bot_name,
        role=config.role,
        teammate=config.teammate_imposter,
        show_mission_progress=config.show_mission_progress,
    )
    if config.state_mode == "ego":
        return EgocentricStateBuilder(visible_range=config.ego_visible_range, **common)
    elif config.state_mode == "privileged":
        all_names = [p["name"] for p in config.all_players] if config.all_players else []
        return PrivilegedStateBuilder(
            all_players=all_names,
            visible_range=config.privileged_visible_range,
            **common,
        )
    raise ValueError(f"Unknown state_mode: {config.state_mode}")


__all__ = ["BaseStateBuilder", "EgocentricStateBuilder", "PrivilegedStateBuilder",
           "create_state_builder"]
