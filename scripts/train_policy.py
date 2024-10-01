import argparse

from monkeypatch import monkeypatch
from omnisafe.utils.tools import custom_cfgs_to_dict, update_dict

import rosarl
import rosarl.algorithms
import rosarl.envs

if __name__ == "__main__":
    monkeypatch()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        metavar="ALGO",
        default="PPOLag",
        help="algorithm to train",
        choices=rosarl.ALGORITHMS["all"],
    )
    parser.add_argument(
        "--env-id",
        type=str,
        metavar="ENV",
        default="SafetyPointGoal1-v0",
        help="the name of test environment",
    )
    parser.add_argument(
        "--parallel",
        default=1,
        type=int,
        metavar="N",
        help="number of paralleled progress for calculations.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=10000000,
        metavar="STEPS",
        help="total number of steps to train for algorithm",
    )
    parser.add_argument(
        "--layer-size",
        type=int,
        default=0,
        help="size of the hidden layers",
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

    custom_cfgs = {}
    for k, v in unparsed_args.items():
        update_dict(custom_cfgs, custom_cfgs_to_dict(k, v))

    if args.layer_size > 0:
        hidden_sizes = {
            "model_cfgs": {
                "actor": {"hidden_sizes": [args.layer_size, args.layer_size]},
                "critic": {"hidden_sizes": [args.layer_size, args.layer_size]},
            }
        }
        update_dict(custom_cfgs, hidden_sizes)

    args_cfgs = vars(args)
    args_cfgs.pop("layer_size")

    agent = rosarl.Agent(
        args.algo,
        args.env_id,
        train_terminal_cfgs=args_cfgs,
        custom_cfgs=custom_cfgs,
    )
    agent.learn()
