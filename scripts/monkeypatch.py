from typing import Any

import torch
from omnisafe.adapter import OnPolicyAdapter
from omnisafe.algorithms import PolicyGradient


def patch_PolicyGradient():
    original__init_log = PolicyGradient._init_log

    def _init_log(self) -> None:
        original__init_log(self)
        self._logger.register_key(
            "Metrics/EpSuccess",
            window_length=self._cfgs.logger_cfgs.window_lens,
        )

    PolicyGradient._init_log = _init_log


def patch_OnPolicyAdapter():
    original__log_value = OnPolicyAdapter._log_value
    original__log_metrics = OnPolicyAdapter._log_metrics
    original__reset_log = OnPolicyAdapter._reset_log

    def _log_value(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        info: dict[str, Any],
    ) -> None:
        original__log_value(self, reward, cost, info)
        success = (
            info["final_info"].get("success", False)
            if "final_info" in info
            else info.get("success", False)
        )
        self._ep_success += success

    def _log_metrics(self, logger, idx: int) -> None:
        original__log_metrics(self, logger, idx)
        logger.store({"Metrics/EpSuccess": self._ep_success[idx]})

    def _reset_log(self, idx: int | None = None) -> None:
        original__reset_log(self, idx)
        if idx is None:
            self._ep_success = torch.zeros(self._env.num_envs, device=self._device)
        else:
            self._ep_success[idx] = 0

    OnPolicyAdapter._log_value = _log_value
    OnPolicyAdapter._log_metrics = _log_metrics
    OnPolicyAdapter._reset_log = _reset_log


def monkeypatch():
    patch_PolicyGradient()
    patch_OnPolicyAdapter()
