# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of OnPolicyBuffer."""

from __future__ import annotations

import torch
from omnisafe.common.buffer.onpolicy_buffer import OnPolicyBuffer
from omnisafe.typing import DEVICE_CPU, AdvatageEstimator, OmnisafeSpace
from omnisafe.utils import distributed
from omnisafe.utils.math import discount_cumsum


class MinmaxOnPolicyBuffer(OnPolicyBuffer):
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        gamma: float,
        lam: float,
        lam_c: float,
        advantage_estimator: AdvatageEstimator,
        penalty_coefficient: float = 0,
        standardized_adv_r: bool = False,
        standardized_adv_c: bool = False,
        device: torch.device = DEVICE_CPU,
    ) -> None:

        super().__init__(
            obs_space,
            act_space,
            size,
            gamma,
            lam,
            lam_c,
            advantage_estimator,
            penalty_coefficient,
            standardized_adv_r,
            standardized_adv_c,
            device,
        )

        self.data["last_value_r"] = torch.zeros(
            (size,), dtype=torch.float32, device=device
        )
        self.data["last_value_c"] = torch.zeros(
            (size,), dtype=torch.float32, device=device
        )
        self.data["unsafe"] = torch.zeros((size,), dtype=torch.bool, device=device)
        self._episode_slices = []

    def finish_path(
        self,
        last_value_r: torch.Tensor | None = None,
        last_value_c: torch.Tensor | None = None,
    ) -> None:
        """Finish the current path and calculate the advantages of state-action pairs.

        On-policy algorithms need to calculate the advantages of state-action pairs
        after the path is finished. This function calculates the advantages of
        state-action pairs and stores them in the buffer, following the steps:

        .. hint::
            #. Calculate the discounted return.
            #. Calculate the advantages of the reward.
            #. Calculate the advantages of the cost.

        Args:
            last_value_r (torch.Tensor, optional): The value of the last state of the current path.
                Defaults to torch.zeros(1).
            last_value_c (torch.Tensor, optional): The value of the last state of the current path.
                Defaults to torch.zeros(1).
        """
        if last_value_r is None:
            last_value_r = torch.zeros(1, device=self._device)
        if last_value_c is None:
            last_value_c = torch.zeros(1, device=self._device)

        # path_slice = slice(self.path_start_idx, self.ptr)
        last_value_r = last_value_r.to(self._device)
        last_value_c = last_value_c.to(self._device)

        self._episode_slices.append((self.path_start_idx, self.ptr))
        self.data["last_value_r"][self.ptr - 1] = last_value_r
        self.data["last_value_c"][self.ptr - 1] = last_value_c

        self.path_start_idx = self.ptr

    def _calculate_advantages(self):
        for slice_start_idx, slice_end_idx in self._episode_slices:
            last_value_r = self.data["last_value_r"][slice_end_idx - 1 : slice_end_idx]
            last_value_c = self.data["last_value_c"][slice_end_idx - 1 : slice_end_idx]

            path_slice = slice(slice_start_idx, slice_end_idx)
            rewards = torch.cat([self.data["reward"][path_slice], last_value_r])
            values_r = torch.cat([self.data["value_r"][path_slice], last_value_r])
            costs = torch.cat([self.data["cost"][path_slice], last_value_c])
            values_c = torch.cat([self.data["value_c"][path_slice], last_value_c])

            discountred_ret = discount_cumsum(rewards, self._gamma)[:-1]
            self.data["discounted_ret"][path_slice] = discountred_ret
            rewards -= self._penalty_coefficient * costs

            adv_r, target_value_r = self._calculate_adv_and_value_targets(
                values_r,
                rewards,
                lam=self._lam,
            )
            adv_c, target_value_c = self._calculate_adv_and_value_targets(
                values_c,
                costs,
                lam=self._lam_c,
            )

            self.data["adv_r"][path_slice] = adv_r
            self.data["target_value_r"][path_slice] = target_value_r
            self.data["adv_c"][path_slice] = adv_c
            self.data["target_value_c"][path_slice] = target_value_c

    def get(self) -> dict[str, torch.Tensor]:
        """Get the data in the buffer.

        .. hint::
            We provide a trick to standardize the advantages of state-action pairs. We calculate the
            mean and standard deviation of the advantages of state-action pairs and then standardize
            the advantages of state-action pairs. You can turn on this trick by setting the
            ``standardized_adv_r`` to ``True``. The same trick is applied to the advantages of the
            cost.

        Returns:
            The data stored and calculated in the buffer.
        """
        self._calculate_advantages()
        self._episode_slices.clear()

        return super().get()

    def update_minmax(self, minmax_penalty) -> None:
        minmax_penalty.update(self.data["reward"], self.data["value_r"])

    def apply_minmax(self, minmax_penalty) -> None:
        reward = self.data["reward"]
        unsafe = self.data["unsafe"]
        
        if torch.any(unsafe):
            reward[unsafe] = minmax_penalty.penalty