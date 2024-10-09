from typing import Any

import torch
from omnisafe.adapter import OnPolicyAdapter
from omnisafe.algorithms import PolicyGradient

from rosarl.utils.info_data import extract_data_from_info


def patch_PolicyGradient():
    original__init_log = PolicyGradient._init_log

    def _init_log(self) -> None:
        original__init_log(self)
        self._logger.register_key(
            "Metrics/EpSuccess",
            window_length=self._cfgs.logger_cfgs.window_lens,
        )
        self._logger.register_key(
            "Metrics/CumulativeCost",
            window_length=1,
        )

    PolicyGradient._init_log = _init_log


def patch_OnPolicyAdapter():
    original___init__ = OnPolicyAdapter.__init__
    original__log_value = OnPolicyAdapter._log_value
    original__log_metrics = OnPolicyAdapter._log_metrics
    original__reset_log = OnPolicyAdapter._reset_log

    def __init__(
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs,
    ) -> None:
        original___init__(self, env_id, num_envs, seed, cfgs)
        self._cumulative_cost = torch.tensor(0.0, device=self._device)

    def _log_value(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        info: dict[str, Any],
    ) -> None:
        original__log_value(self, reward, cost, info)

        success = extract_data_from_info(info, "success")
        if success is None:
            success = torch.zeros_like(self._ep_success)
        else:
            success = torch.as_tensor(success, device=self._device)

        self._ep_success += torch.as_tensor(success, device=self._device)
        self._cumulative_cost += cost.sum()

    def _log_metrics(self, logger, idx: int) -> None:
        original__log_metrics(self, logger, idx)
        logger.store(
            {
                "Metrics/EpSuccess": self._ep_success[idx],
                "Metrics/CumulativeCost": self._cumulative_cost,
            }
        )

    def _reset_log(self, idx: int | None = None) -> None:
        original__reset_log(self, idx)
        if idx is None:
            self._ep_success = torch.zeros(self._env.num_envs, device=self._device)
        else:
            self._ep_success[idx] = 0

    OnPolicyAdapter.__init__ = __init__
    OnPolicyAdapter._log_value = _log_value
    OnPolicyAdapter._log_metrics = _log_metrics
    OnPolicyAdapter._reset_log = _reset_log


def monkeypatch():
    patch_PolicyGradient()
    patch_OnPolicyAdapter()
