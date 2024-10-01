"""Implementation of the Minmax version of the TRPO algorithm."""

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.trpo import TRPO

from rosarl.algorithms.minmax_mixin import MinmaxMixin


@registry.register
class TRPOMinmax(MinmaxMixin, TRPO):
    """The TRPO algorithm with Minmax.

    A combination of the Minmax penalty and the Trust Region Policy Optimization algorithm.
    """
