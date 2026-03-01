from __future__ import annotations

from typing import Any

import torch
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config
from rich.progress import track

from rosarl.adapter import MinmaxAdapter
from rosarl.common.buffer import MinmaxVectorOnPolicyBuffer
from rosarl.utils.info_data import extract_data_from_info


class MinmaxBatchAdapter(MinmaxAdapter):
    def rollout(
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        buffer: MinmaxVectorOnPolicyBuffer,
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

            unsafe = extract_data_from_info(info, "unsafe")
            unsafe = torch.atleast_1d(torch.as_tensor(unsafe, device=self._device))

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
                unsafe=unsafe,
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
