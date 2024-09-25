import random
from typing import Any, ClassVar

import omnisafe
import torch
from gymnasium import spaces
from omnisafe.envs.core import CMDP, Wrapper, env_register, env_unregister
from safety_gymnasium.tasks import GoalLevel0, GoalLevel1, GoalLevel2


class TerminalUnsafeGoalLevel0(GoalLevel0):
    """An agent must navigate to a goal while avoiding more hazards and vases."""

    def __init__(self, config) -> None:
        super().__init__(config=config)
        # pylint: disable=no-member

        self.unsafe_terminate = 2


class TerminalUnsafeGoalLevel1(GoalLevel1):
    """An agent must navigate to a goal while avoiding more hazards and vases."""

    def __init__(self, config) -> None:
        super().__init__(config=config)
        # pylint: disable=no-member

        self.unsafe_terminate = 2


class TerminalUnsafeGoalLevel2(GoalLevel2):
    """An agent must navigate to a goal while avoiding more hazards and vases."""

    def __init__(self, config) -> None:
        super().__init__(config=config)
        # pylint: disable=no-member

        self.unsafe_terminate = 2
