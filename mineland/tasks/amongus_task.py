from mineland.sim.data.task_info import TaskInfo

from .base_task import BaseTask


class AmongUsTask(BaseTask):
    def __init__(
        self,
        *,
        goal: str = "Free play (Among Us mode)",
        guidance: str = "Just run; no explicit success condition.",
        mode: str = "cooperative",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.goal = goal
        self.guidance = guidance
        self.mode = mode
        self.winner = None
        self.game_ended = False
        print("Starting AmongUsTask: free-play session running.")

    def reset(self):
        obs = self.env.reset()
        # Reset game state
        self.winner = None
        self.game_ended = False
        if hasattr(self.env, 'sim') and hasattr(self.env.sim, 'server_manager'):
            self.env.sim.server_manager.reset_game_result()
        return obs

    def step(self, action):
        obs, code_info, events, done, _ = self.env.step(action)

        # Check game result from server
        is_success = False
        is_failed = False
        additional_info = {}
        
        winner_from_events, message_from_events = self._detect_winner_from_events(events)

        if hasattr(self.env, 'sim') and hasattr(self.env.sim, 'server_manager'):
            game_result, game_message = self.env.sim.server_manager.get_game_result()
            if not game_result and winner_from_events:
                game_result = winner_from_events
                game_message = message_from_events
                # Persist the detected result for consistency
                self.env.sim.server_manager.game_result = game_result
                self.env.sim.server_manager.game_result_message = game_message

            if game_result and not self.game_ended:
                self._finalize_game(game_result, game_message, additional_info)
                done = True
                is_success = True
        elif winner_from_events and not self.game_ended:
            self._finalize_game(winner_from_events, message_from_events, additional_info)
            done = True
            is_success = True

        task_info = TaskInfo(
            task_id=self.task_id,
            is_success=is_success,
            is_failed=is_failed,
            goal=self.goal,
            guidance=self.guidance,
            mode=self.mode,
        )
        
        # Add additional game result info
        for key, value in additional_info.items():
            setattr(task_info, key, value)

        return obs, code_info, events, done, task_info

    def _detect_winner_from_events(self, events):
        """Inspect event list for win messages."""
        if not events:
            return None, None
        for event in events:
            if not isinstance(event, dict):
                continue
            message = str(event.get('message', '')).strip()
            if not message:
                continue
            lowered = message.lower()
            if 'win' not in lowered:
                continue
            if 'imposter' in lowered:
                return 'imposter', message
            if 'crewmate' in lowered or 'crewmates' in lowered or 'crew mate' in lowered or 'crew win' in lowered:
                return 'crewmate', message
        return None, None

    def _finalize_game(self, winner, message, info_dict):
        """Mark the task as finished and shut down the server."""
        self.game_ended = True
        self.winner = winner
        info_dict['winner'] = winner
        if message:
            info_dict['message'] = message
        print(f"Game ended! Winner: {winner}")
        if message:
            print(f"Message: {message}")

        if hasattr(self.env, 'sim') and hasattr(self.env.sim, 'server_manager'):
            print("Shutting down server due to game completion...")
            try:
                self.env.sim.server_manager.shutdown()
                print("Server shutdown completed.")
            except Exception as e:
                print(f"Error during server shutdown: {e}")
