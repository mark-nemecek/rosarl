from typing import Any

from omnisafe.algorithms.algo_wrapper import (
    AlgoWrapper,
    Config,
    Evaluator,
    Plotter,
    get_default_kwargs_yaml,
    recursive_check_config,
)

from rosarl.algorithms import ALGORITHM2TYPE, ALGORITHMS
from rosarl.utils.config import (
    get_default_kwargs_yaml as rosarl_get_default_kwargs_yaml,
)


class CustomAlgoWrapper(AlgoWrapper):
    """Algo Wrapper for algorithms.

    Args:
        algo (str): The algorithm name.
        env_id (str): The environment id.
        train_terminal_cfgs (dict[str, Any], optional): The configurations for training termination.
            Defaults to None.
        custom_cfgs (dict[str, Any], optional): The custom configurations. Defaults to None.

    Attributes:
        algo (str): The algorithm name.
        env_id (str): The environment id.
        train_terminal_cfgs (dict[str, Any]): The configurations for training termination.
        custom_cfgs (dict[str, Any]): The custom configurations.
        cfgs (Config): The configurations for the algorithm.
        algo_type (str): The algorithm type.
    """

    algo_type: str

    def __init__(
        self,
        algo: str,
        env_id: str,
        train_terminal_cfgs: dict[str, Any] | None = None,
        custom_cfgs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize an instance of :class:`AlgoWrapper`."""
        self.algo: str = algo
        self.env_id: str = env_id
        # algo_type will set in _init_checks()
        self.train_terminal_cfgs: dict[str, Any] | None = train_terminal_cfgs
        self.custom_cfgs: dict[str, Any] | None = custom_cfgs
        self._evaluator: Evaluator | None = None
        self._plotter: Plotter | None = None
        self.cfgs: Config = self._init_config()
        self._init_checks()
        self._init_algo()

    def _init_config(self) -> Config:
        """Initialize config.

        Initialize the configurations for the algorithm, following the order of default
        configurations, custom configurations, and terminal configurations.

        Returns:
            The configurations for the algorithm.

        Raises:
            AssertionError: If the algorithm name is not in the supported algorithms.
        """
        assert (
            self.algo in ALGORITHMS["all"]
        ), f"{self.algo} doesn't exist. Please choose from {ALGORITHMS['all']}."
        self.algo_type = ALGORITHM2TYPE.get(self.algo, "")
        if self.train_terminal_cfgs is not None:
            if self.algo_type in ["model-based", "offline"]:
                assert (
                    self.train_terminal_cfgs["vector_env_nums"] == 1
                ), "model-based and offline only support vector_env_nums==1!"
            if self.algo_type in ["off-policy", "model-based", "offline"]:
                assert (
                    self.train_terminal_cfgs["parallel"] == 1
                ), "off-policy, model-based and offline only support parallel==1!"

        if self.algo in [
            "TRPOMinmax",
            "PPOMinmax",
            "TRPOMinmaxBatch",
            "PPOMinmaxBatch",
        ]:
            cfgs = rosarl_get_default_kwargs_yaml(
                self.algo, self.env_id, self.algo_type
            )
        else:
            cfgs = get_default_kwargs_yaml(self.algo, self.env_id, self.algo_type)

        # update the cfgs from custom configurations
        if self.custom_cfgs:
            # avoid repeatedly record the env_id and algo
            if "env_id" in self.custom_cfgs:
                self.custom_cfgs.pop("env_id")
            if "algo" in self.custom_cfgs:
                self.custom_cfgs.pop("algo")
            # validate the keys of custom configuration
            recursive_check_config(self.custom_cfgs, cfgs)
            # update the cfgs from custom configurations
            cfgs.recurisve_update(self.custom_cfgs)
            # save configurations specified in current experiment
            cfgs.update({"exp_increment_cfgs": self.custom_cfgs})
        # update the cfgs from custom terminal configurations
        if self.train_terminal_cfgs:
            # avoid repeatedly record the env_id and algo
            if "env_id" in self.train_terminal_cfgs:
                self.train_terminal_cfgs.pop("env_id")
            if "algo" in self.train_terminal_cfgs:
                self.train_terminal_cfgs.pop("algo")
            # validate the keys of train_terminal_cfgs configuration
            recursive_check_config(self.train_terminal_cfgs, cfgs.train_cfgs)
            # update the cfgs.train_cfgs from train_terminal configurations
            cfgs.train_cfgs.recurisve_update(self.train_terminal_cfgs)
            # save configurations specified in current experiment
            cfgs.recurisve_update(
                {"exp_increment_cfgs": {"train_cfgs": self.train_terminal_cfgs}}
            )

        # the exp_name format is PPO-{SafetyPointGoal1-v0}
        exp_name = f"{self.algo}-{{{self.env_id}}}"
        cfgs.recurisve_update(
            {"exp_name": exp_name, "env_id": self.env_id, "algo": self.algo}
        )
        if hasattr(cfgs.train_cfgs, "total_steps") and hasattr(
            cfgs.algo_cfgs, "steps_per_epoch"
        ):
            epochs = cfgs.train_cfgs.total_steps // cfgs.algo_cfgs.steps_per_epoch
            cfgs.train_cfgs.recurisve_update(
                {"epochs": epochs},
            )
        return cfgs
