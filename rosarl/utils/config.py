import os

from omnisafe.utils.tools import load_yaml
from omnisafe.utils.config import Config

def get_default_kwargs_yaml(algo: str, env_id: str, algo_type: str) -> Config:
    """Get the default kwargs from ``yaml`` file.

    .. note::
        This function search the ``yaml`` file by the algorithm name and environment name. Make
        sure your new implemented algorithm or environment has the same name as the yaml file.

    Args:
        algo (str): The algorithm name.
        env_id (str): The environment name.
        algo_type (str): The algorithm type.

    Returns:
        The default kwargs.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(path, '..', 'configs', algo_type, f'{algo}.yaml')
    print(f'Loading {algo}.yaml from {cfg_path}')
    kwargs = load_yaml(cfg_path)
    default_kwargs = kwargs['defaults']
    env_spec_kwargs = kwargs.get(env_id)

    default_kwargs = Config.dict2config(default_kwargs)

    if env_spec_kwargs is not None:
        default_kwargs.recurisve_update(env_spec_kwargs)

    return default_kwargs