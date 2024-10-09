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
"""Implementation of VectorOnPolicyBuffer."""

from __future__ import annotations

import torch
from omnisafe.common.buffer.vector_onpolicy_buffer import VectorOnPolicyBuffer
from omnisafe.typing import DEVICE_CPU, AdvatageEstimator, OmnisafeSpace

from rosarl.common.buffer.minmax_onpolicy_buffer import MinmaxOnPolicyBuffer


class MinmaxVectorOnPolicyBuffer(VectorOnPolicyBuffer):
    """Vectorized on-policy buffer for Minmax algos.

    The vector-on-policy buffer is used to store the data from vector environments. The data is
    stored in a list of on-policy buffers, each of which corresponds to one environment.

    .. warning::
        The buffer only supports Box spaces.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        size (int): Size of the buffer.
        gamma (float): Discount factor.
        lam (float): Lambda for GAE.
        lam_c (float): Lambda for GAE for cost.
        advantage_estimator (AdvatageEstimator): Advantage estimator.
        penalty_coefficient (float): Penalty coefficient.
        standardized_adv_r (bool): Whether to standardize the advantage for reward.
        standardized_adv_c (bool): Whether to standardize the advantage for cost.
        num_envs (int, optional): Number of environments. Defaults to 1.
        device (torch.device, optional): Device to store the data. Defaults to
            ``torch.device('cpu')``.

    Attributes:
        buffers (list[OnPolicyBuffer]): List of on-policy buffers.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        gamma: float,
        lam: float,
        lam_c: float,
        advantage_estimator: AdvatageEstimator,
        penalty_coefficient: float,
        standardized_adv_r: bool,
        standardized_adv_c: bool,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize an instance of :class:`VectorOnPolicyBuffer`."""
        self._num_buffers: int = num_envs
        self._standardized_adv_r: bool = standardized_adv_r
        self._standardized_adv_c: bool = standardized_adv_c

        if num_envs < 1:
            raise ValueError("num_envs must be greater than 0.")
        self.buffers: list[MinmaxOnPolicyBuffer] = [
            MinmaxOnPolicyBuffer(
                obs_space=obs_space,
                act_space=act_space,
                size=size,
                gamma=gamma,
                lam=lam,
                lam_c=lam_c,
                advantage_estimator=advantage_estimator,
                penalty_coefficient=penalty_coefficient,
                device=device,
            )
            for _ in range(num_envs)
        ]

    def update_minmax(self, minmax_penalty) -> None:
        for buffer in self.buffers:
            buffer.update_minmax(minmax_penalty)

    def apply_minmax(self, minmax_penalty) -> None:
        for buffer in self.buffers:
            buffer.apply_minmax(minmax_penalty)
