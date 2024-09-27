from types import MappingProxyType

import omnisafe

from rosarl.algorithms.trpo_minmax import TRPOMinmax

ALGORITHMS = {
    "all": tuple(a for a in omnisafe.ALGORITHMS["all"]) + ("TRPOMinmax",),
    "on-policy": tuple(a for a in omnisafe.ALGORITHMS["on-policy"]) + ("TRPOMinmax",),
    "off-policy": omnisafe.ALGORITHMS["off-policy"],
    "model-based": omnisafe.ALGORITHMS["model-based"],
    "offline": omnisafe.ALGORITHMS["offline"],
}

ALGORITHM2TYPE = {
    algo: algo_type
    for algo_type, algorithms in ALGORITHMS.items()
    for algo in algorithms
}

ALGORITHMS = MappingProxyType(ALGORITHMS)  # make this immutable
ALGORITHM2TYPE = MappingProxyType(ALGORITHM2TYPE)  # make this immutable
