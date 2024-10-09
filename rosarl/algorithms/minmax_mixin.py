from omnisafe.utils import distributed

from rosarl.adapter import MinmaxAdapter, MinmaxBatchAdapter
from rosarl.algorithms.minmax_penalty import MinmaxPenalty
from rosarl.common.buffer.minmax_vector_onpolicy_buffer import (
    MinmaxVectorOnPolicyBuffer,
)


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


class MinmaxBatchMixin(MinmaxMixin):
    def _init(self) -> None:
        self._buf: MinmaxVectorOnPolicyBuffer = MinmaxVectorOnPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._steps_per_epoch,
            gamma=self._cfgs.algo_cfgs.gamma,
            lam=self._cfgs.algo_cfgs.lam,
            lam_c=self._cfgs.algo_cfgs.lam_c,
            advantage_estimator=self._cfgs.algo_cfgs.adv_estimation_method,
            standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
            standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
            penalty_coefficient=self._cfgs.algo_cfgs.penalty_coef,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )

    def _init_env(self) -> None:
        """Initialize the environment.

        Uses :class:`rosarl.adapter.MinmaxAdapter` to adapt the environment to the
        algorithm and implement the Minmax Penalty.

        Raises:
            AssertionError: If the number of steps per epoch is not divisible by the number of
                environments.
        """
        self._minmax_penalty = MinmaxPenalty(device=self._device)

        self._env: MinmaxBatchAdapter = MinmaxBatchAdapter(
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
        self._buf.update_minmax(self._minmax_penalty)
        self._buf.apply_minmax(self._minmax_penalty)

        super()._update()
