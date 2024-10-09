import torch


class MinmaxPenalty:
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
