from omnisafe.utils import distributed

from rosarl.adapter import MinmaxAdapter


class MinmaxMixin:
    def _init_log(self) -> None:
        """Log additional information.

        +----------------------------+--------------------------+
        | Things to log              | Description              |
        +============================+==========================+
        | Misc/MinmaxPenalty         | The Minmax penalty.      |
        | Metrics/CumulativeCost     | The cumulative cost.     |
        +----------------------------+--------------------------+
        """
        super()._init_log()
        self._logger.register_key("Misc/MinmaxPenalty", window_length=1)

    def _init_env(self) -> None:
        """Initialize the environment.

        Uses :class:`rosarl.adapter.MinmaxAdapter` to adapt the environment to the
        algorithm and implement the Minmax Penalty.

        Raises:
            AssertionError: If the number of steps per epoch is not divisible by the number of
                environments.
        """
        self._minmax_penalty = MinmaxPenalty(device=self._device)

        self._env: MinmaxAdapter = MinmaxAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
            self._minmax_penalty,
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, "The number of steps per epoch is not divisible by the number of environments."
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )

    def _update(self) -> None:
        super()._update()

        self._logger.store(
            {
                "Misc/MinmaxPenalty": self._minmax_penalty.penalty,
            },
        )

