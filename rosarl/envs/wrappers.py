import torch
from gymnasium import Wrapper
from safety_gymnasium.wrappers import SafeAutoResetWrapper


class TerminalUnsafeWrapper(Wrapper):
    def __init__(self, env, goal_terminal):
        super().__init__(env)
        self.goal_terminal = goal_terminal

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)

        is_unsafe = cost > 0.0
        info["unsafe"] = is_unsafe
        goal_met = info.get("goal_met", False)
        terminated = terminated or is_unsafe or (self.goal_terminal and goal_met)
        reward = -1.0 if is_unsafe else reward

        return obs, reward, cost, terminated, truncated, info
