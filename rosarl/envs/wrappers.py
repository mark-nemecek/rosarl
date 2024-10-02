from typing import Any, SupportsFloat

from gymnasium import Wrapper


class TerminalUnsafeWrapper(Wrapper):
    def __init__(self, env, unsafe_terminal, goal_terminal):
        super().__init__(env)
        self.unsafe_terminal = unsafe_terminal
        self.goal_terminal = goal_terminal
        self.goal_has_been_met = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self.goal_has_been_met = False
        return super().reset(seed=seed, options=options)

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, cost, terminated, truncated, info = super().step(action)

        is_unsafe = cost > 0.0
        info["unsafe"] = is_unsafe
        goal_met = info.get("goal_met", False)
        self.goal_has_been_met |= goal_met

        if self.goal_terminal:
            success = goal_met
        else:
            success = truncated and self.goal_has_been_met and not is_unsafe
        info["success"] = success

        terminated = (
            terminated
            or (self.unsafe_terminal and is_unsafe)
            or (self.goal_terminal and goal_met)
        )
        reward = -1.0 if (self.unsafe_terminal and is_unsafe) else reward

        return obs, reward, cost, terminated, truncated, info
