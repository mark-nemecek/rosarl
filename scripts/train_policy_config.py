import argparse

import yaml
from monkeypatch import monkeypatch
from omnisafe.utils.tools import custom_cfgs_to_dict, update_dict

import rosarl
import rosarl.algorithms
import rosarl.envs


def load_config(config_path: str):
    args = None
    with open(config_path, encoding="utf-8") as file:
        try:
            print(f'Loading {config_path}')
            args = yaml.load(file, Loader=yaml.FullLoader)  # noqa: S506
            assert args is not None, "load file error"
        except yaml.YAMLError as exc:
            raise AssertionError(f"load file error: {exc}") from exc

    return args


if __name__ == "__main__":
    monkeypatch()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        metavar="CONFIG",
        help="config file to use",
    )
    parser.add_argument(
        "--algo",
        type=str,
        metavar="ALGO",
        help="algorithm to train",
        choices=rosarl.ALGORITHMS["all"],
    )
    parser.add_argument(
        "--env-id",
        type=str,
        metavar="ENV",
        help="the name of test environment",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=10000000,
        metavar="STEPS",
        help="total number of steps to train for algorithm",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        metavar="DEVICES",
        help="device to use for training",
    )
    parser.add_argument(
        "--vector-env-nums",
        type=int,
        default=1,
        metavar="VECTOR-ENV",
        help="number of vector envs to use for training",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=16,
        metavar="THREADS",
        help="number of threads to use for torch",
    )
    args, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))

    args_cfgs = vars(args)
    config_file = args_cfgs.pop("config")
    file_cfgs = load_config(config_file)
    if "defaults" in file_cfgs:
        custom_cfgs = file_cfgs["defaults"]
        # if args.env_id in file_cfgs:
        # update_dict(custom_cfgs, file_cfgs[args.env_id])
    else:
        custom_cfgs = file_cfgs

    for k, v in unparsed_args.items():
        update_dict(custom_cfgs, custom_cfgs_to_dict(k, v))

    agent = rosarl.Agent(
        args.algo,
        args.env_id,
        train_terminal_cfgs=args_cfgs,
        custom_cfgs=custom_cfgs,
    )
    agent.learn()
