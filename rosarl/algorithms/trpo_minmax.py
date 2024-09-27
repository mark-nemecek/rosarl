"""Implementation of the Minmax version of the TRPO algorithm."""

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.trpo import TRPO
from omnisafe.utils import distributed

from rosarl.adapter import MinmaxAdapter


@registry.register
class TRPOMinmax(TRPO):
    """The TRPO algorithm with Minmax.

    A combination of the Minmax penalty and the Trust Region Policy Optimization algorithm.
    """

    def _init_log(self) -> None:
        """Log the TRPOMinmax specific information.

        +----------------------------+--------------------------+
        | Things to log              | Description              |
        +============================+==========================+
        | Penalty/Minmax             | The Minmax penalty.      |
        +----------------------------+--------------------------+
        """
        super()._init_log()
        self._logger.register_key('Penalty/Minmax')

    def _init_env(self) -> None:
        """Initialize the environment.

        Uses :class:`rosarl.adapter.MinmaxAdapter` to adapt the environment to the
        algorithm and implement the Minmax Penalty.

        Raises:
            AssertionError: If the number of steps per epoch is not divisible by the number of
                environments.
        """
        self._env: MinmaxAdapter = MinmaxAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )
