from typing import Any, SupportsFloat

from gymnasium import Wrapper


class RosarlWrapper(Wrapper):
    """Wrapper which reports unsafe states and task success and also allows termination on reaching an unsafe state or a goal state."""

    def __init__(self, env, unsafe_terminal: bool, goal_terminal: bool):
        super().__init__(env)
        self.unsafe_terminal = unsafe_terminal
        self.goal_terminal = goal_terminal
        self.goal_state_reached = False
        self.unsafe_state_reached = False

        if self.goal_terminal:
            self.env.task.mechanism_conf.continue_goal = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self.goal_state_reached = False
        self.unsafe_state_reached = False
        return super().reset(seed=seed, options=options)

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, cost, terminated, truncated, info = super().step(action)

        is_unsafe = cost > 0.0
        info["unsafe"] = is_unsafe
        self.unsafe_state_reached |= is_unsafe

        goal_met = info.get("goal_met", False)
        self.goal_state_reached |= goal_met

        info["success"] = self.is_success(truncated, is_unsafe, goal_met)

        terminated = (
            terminated
            or (self.unsafe_terminal and is_unsafe)
            or (self.goal_terminal and goal_met)
        )
        reward = -1.0 if (self.unsafe_terminal and is_unsafe) else reward

        return obs, reward, cost, terminated, truncated, info

    def is_success(self, truncated, is_unsafe, goal_met):
        if self.goal_terminal:
            if self.unsafe_terminal:
                # terminal: goal and unsafe
                success = goal_met and not is_unsafe
            else:
                # terminal: goal and not unsafe
                success = goal_met and not self.unsafe_state_reached
        elif self.unsafe_terminal:
            # terminal: not goal and unsafe
            success = truncated and self.goal_state_reached
        else:
            # terminal: not goal and not unsafe
            success = (
                truncated and self.goal_state_reached and not self.unsafe_state_reached
            )

        return success
