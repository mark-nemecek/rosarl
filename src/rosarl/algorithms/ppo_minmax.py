"""Implementation of the Minmax version of the PPO algorithm."""

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO

from rosarl.algorithms.minmax_mixin import MinmaxBatchMixin, MinmaxMixin


@registry.register
class PPOMinmax(MinmaxMixin, PPO):
    """The PPO algorithm with Minmax.

    A combination of the Minmax penalty and the Proximal Policy Optimization algorithm.
    """


@registry.register
class PPOMinmaxBatch(MinmaxBatchMixin, PPO):
    """The PPO algorithm with Minmax.

    A combination of the Minmax penalty and the Proximal Policy Optimization algorithm.
    """
