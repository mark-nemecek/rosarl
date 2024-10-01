from typing import Any
import torch
from gymnasium import Wrapper


class TerminalUnsafeWrapper(Wrapper):
    def __init__(self, env, goal_terminal):
        super().__init__(env)
        self.goal_terminal = goal_terminal
        self.goal_has_been_met = False
    
    def reset(self, *, seed, options = None):
        self.goal_has_been_met = False
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)

        is_unsafe = cost > 0.0
        info["unsafe"] = is_unsafe
        goal_met = info.get("goal_met", False)
        self.goal_has_been_met |= goal_met

        if self.goal_terminal:
            success = goal_met
        else:
            success = truncated and self.goal_has_been_met and not is_unsafe
        info["success"] = success

        terminated = terminated or is_unsafe or (self.goal_terminal and goal_met)
        reward = -1.0 if is_unsafe else reward

        return obs, reward, cost, terminated, truncated, info
