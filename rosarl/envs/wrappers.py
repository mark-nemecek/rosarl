from safety_gymnasium.wrappers import SafeAutoResetWrapper
from gymnasium import Wrapper


class TerminalUnsafeWrapper(Wrapper):
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)

        is_unsafe = cost > 0.0
        info["unsafe"] = is_unsafe
        terminated = terminated or is_unsafe
        reward = -1 if is_unsafe else reward

        return obs, reward, cost, terminated, truncated, info
