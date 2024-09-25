import random
from typing import Any, ClassVar

import omnisafe
import torch
from gymnasium import spaces
from omnisafe.envs.core import CMDP, Wrapper, env_register, env_unregister
from safety_gymnasium.tasks import PushLevel0, PushLevel1, PushLevel2

push_all = {
    'task': 'push',
    'box_size': 0.2,
    'box_null_dist': 0,
    'hazards_size': 0.3,
    'unsafe_terminate':2,
    }


class TerminalUnsafePushLevel0(PushLevel0):
    """An agent must push a box to a goal."""

    def __init__(self, config) -> None:
        super().__init__(config=config)
        # pylint: disable=no-member

        self.unsafe_terminate = 2


class TerminalUnsafePushLevel1(PushLevel1):
    """An agent must push a box to a goal."""

    def __init__(self, config) -> None:
        super().__init__(config=config)
        # pylint: disable=no-member

        self.unsafe_terminate = 2


class TerminalUnsafePushLevel2(PushLevel2):
    """An agent must push a box to a goal."""

    def __init__(self, config) -> None:
        super().__init__(config=config)
        # pylint: disable=no-member

        self.unsafe_terminate = 2