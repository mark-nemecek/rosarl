from __future__ import annotations

from typing import Any

import torch
from omnisafe.adapter import OnPolicyAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config
from rich.progress import track

from rosarl.algorithms.minmax_penalty import MinmaxPenalty
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

    def __init__(
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
        minmax_penalty: MinmaxPenalty,
    ) -> None:
        """Initialize an instance of :class:`MinmaxAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)

        self._minmax_penalty = minmax_penalty

    def rollout(
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

