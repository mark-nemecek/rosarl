from __future__ import annotations

from typing import Any

import torch
from omnisafe.adapter import OnPolicyAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config
from rich.progress import track

from rosarl.utils.info_data import extract_data_from_info


class MinmaxAdapter(OnPolicyAdapter):
    """Adapter for Minmax algo for OmniSafe.

    :class:`MinmaxAdapter` is used to adapt the environment to the Minmax training.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    _minmax_penalty: MinMaxPenalty

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize an instance of :class:`MinmaxAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)

        self._minmax_penalty = MinMaxPenalty(self._device)

    def rollout(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        buffer: VectorOnPolicyBuffer,
        logger: Logger,
    ) -> None:
        """Rollout the environment and store the data in the buffer.

        .. warning::
            As OmniSafe uses :class:`AutoReset` wrapper, the environment will be reset automatically,
            so the final observation will be stored in ``info['final_observation']``.

        Args:
            steps_per_epoch (int): Number of steps per epoch.
            agent (ConstraintActorCritic): Constraint actor-critic, including actor , reward critic
                and cost critic.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``, ``Minmax Penalty``.
        """
        self._reset_log()

        obs, _ = self.reset()
        for step in track(
            range(steps_per_epoch),
            description=f"Processing rollout for epoch: {logger.current_epoch}...",
        ):
            act, value_r, value_c, logp = agent.step(obs)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            self._log_value(reward=reward, cost=cost, info=info)

            reward = self._penalize_reward(reward, value_r, info)

            if self._cfgs.algo_cfgs.use_cost:
                logger.store({"Value/cost": value_c})
            logger.store({"Value/reward": value_r})

            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=torch.zeros_like(cost),
                value_r=value_r,
                value_c=torch.zeros_like(value_c),
                logp=logp,
            )

            obs = next_obs
            epoch_end = step >= steps_per_epoch - 1
            if epoch_end:
                num_dones = int(terminated.contiguous().sum())
                if self._env.num_envs - num_dones:
                    logger.log(
                        f"\nWarning: trajectory cut off when rollout by epoch\
                            in {self._env.num_envs - num_dones} of {self._env.num_envs} environments.",
                    )

            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1)
                    last_value_c = torch.zeros(1)
                    if not done:
                        if epoch_end:
                            _, last_value_r, last_value_c, _ = agent.step(obs[idx])
                        if time_out:
                            _, last_value_r, last_value_c, _ = agent.step(
                                info["final_observation"][idx],
                            )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)

                    if done or time_out:
                        self._log_metrics(logger, idx)
                        self._reset_log(idx)

                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0

                    buffer.finish_path(last_value_r, last_value_c, idx)

    def _penalize_reward(
        self,
        reward: torch.Tensor,
        value_r: torch.Tensor,
        info: dict[str, Any],
    ):
        self._minmax_penalty.update(reward, value_r)

        unsafe = extract_data_from_info(info, "unsafe")
        if unsafe is None:
            unsafe = torch.zeros_like(reward, dtype=torch.bool)
        else:
            unsafe = torch.as_tensor(unsafe, device=self._device)

        if torch.any(unsafe):
            reward = torch.where(unsafe, self._minmax_penalty.penalty, reward)

        return reward

    def _log_metrics(self, logger: Logger, idx: int) -> None:
        logger.store({"Misc/MinmaxPenalty": self._minmax_penalty.penalty})
        return super()._log_metrics(logger, idx)


class MinMaxPenalty:
    """
    Learn the highest reward/penalty that minimises the probability of reaching bad terminal states
    Arguments:
        - rmin (optional) (torch.Tensor): The lower bound for environment rewards
        - rmax (optional) (torch.Tensor): The upper bound for environment rewards
    Return:
        - The minmax penalty estimate
    Usage:
    Symlink to the desired folder and import, or copy-paste to where needed
    In training loop:
        minmaxpenalty = MinMaxPenalty()
        for each step:
            - take an action and get reward and q_value (or just [value] if using policy gradient)
            penalty = minmaxpenalty.update(reward, Q[state])
            if info["unsafe"]:
                reward = penalty
    """

    def __init__(
        self,
        device: str = "cpu",
        r_min: torch.Tensor = None,
        r_max: torch.Tensor = None,
    ):
        self.r_min = torch.tensor([0.0], device=device) if r_min is None else r_min
        self.r_max = torch.tensor([0.0], device=device) if r_max is None else r_max
        self.v_min = self.r_min
        self.v_max = self.r_max
        self.penalty = min(self.r_min, (self.v_min - self.v_max))
        try:
            self.minmax_update = torch.compile(minmax_update)
            # test compile in case GPU doesn't support it
            self.minmax_update(
                self.r_min, self.r_max, self.v_min, self.v_max, self.r_min, self.r_min
            )
        except RuntimeError:
            self.minmax_update = minmax_update

    def update(self, reward: torch.Tensor, value: torch.Tensor):
        self.r_min, self.r_max, self.v_min, self.v_max, self.penalty = (
            self.minmax_update(
                self.r_min, self.r_max, self.v_min, self.v_max, reward, value
            )
        )


def minmax_update(
    r_min: torch.Tensor,
    r_max: torch.Tensor,
    v_min: torch.Tensor,
    v_max: torch.Tensor,
    reward: torch.Tensor,
    value: torch.Tensor,
):
    new_r_min = min(r_min, reward.min().unsqueeze(0))
    new_r_max = max(r_max, reward.max().unsqueeze(0))
    new_v_min = min(v_min, new_r_min, value.min().unsqueeze(0))
    new_v_max = max(v_max, new_r_max, value.max().unsqueeze(0))

    penalty = min(new_r_min, (new_v_min - new_v_max))

    return new_r_min, new_r_max, new_v_min, new_v_max, penalty
