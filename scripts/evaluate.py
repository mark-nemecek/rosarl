import time, os
import numpy as np
from safe_rl.utils.load_utils import load_policy
from safe_rl.utils.logx import EpochLogger
import imageio
from PIL import Image
import gym
import safety_gym
from collections import defaultdict
import pandas as pd

# save=False --> export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
# save=True --> export LD_PRELOAD=""
# python enjoy.py "../data/data_2023_05_27/unsafe_2_cost25/TRPO_Minmax/good_2023-01-16_11-45-53-trpo_minmax_PointGoal1_s0/" --save blend --seed 42

# Awesome model: python enjoy.py ../data/prelim_safety_results/data/unsafe_1/TRPO_Minmax/2023-05-23_10-32-50-trpo_minmax_PointGoal1_s104  --save gif --episode 4


def evaluate_policy(env, get_action, noise=0, num_episodes=100, seed=None):

    # assert env is not None, \
    #     "Environment not found!\n\n It looks like the environment wasn't saved, " + \
    #     "and we can't run the agent in it. :("

    if noise==0: env = gym.make("Safexp-PointCustom0-TerminalUnsafe-v0")
    if noise==2.5: env = gym.make("Safexp-PointCustom3-TerminalUnsafe-v0")
    if noise==5: env = gym.make("Safexp-PointCustom1-TerminalUnsafe-v0")
    if noise==7.5: env = gym.make("Safexp-PointCustom4-TerminalUnsafe-v0")
    if noise==10: env = gym.make("Safexp-PointCustom2-TerminalUnsafe-v0")

    successes, returns, costs, total_steps = [], [], [], []
    n = 0
    while n < num_episodes:
        n += 1
        env.seed(n)

        o, r, d, ep_ret, ep_cost, ep_goal_met, ep_len = env.reset(), 0, False, 0, 0, 0, 0

        while ep_len<1000:
            if d:
                break

            a = get_action(o)
            a = np.clip(a, env.action_space.low, env.action_space.high)
            o, r, d, info = env.step(a)
            ep_ret += r
            ep_cost += info.get('cost', 0)
            ep_goal_met += info['goal_met']
            ep_len += 1

            if d or (ep_len == 1000):
                successes.append(ep_goal_met); returns.append(ep_ret); costs.append(ep_cost); total_steps.append(ep_len)
    return successes, returns, costs, total_steps

def generate_latex_table(data):
    # Group data by noise level
    grouped_data = defaultdict(list)
    for row in data:
        noise = row[3]
        grouped_data[noise].append(row)

    # Define the sorted order for noise levels
    noise_order = [0, 2.5, 5, 7.5, 10]

    latex_table = "\\begin{table}[h]\n\\centering\n"
    latex_table += "\\begin{tabular}{@{}c l c c c c@{}}\n"
    latex_table += "\\toprule\n"
    latex_table += (
        "\\textbf{Noise} & \\textbf{Algorithm} & \\textbf{Costs $\\downarrow$} & \\textbf{Success Rate $\\uparrow$} & "
        "\\textbf{Returns $\\uparrow$} & \\textbf{Total Steps $\\uparrow$} \\\\\n"
    )
    latex_table += "\\midrule\n"

    # Iterate over the sorted noise levels
    for noise in noise_order:
        if noise not in grouped_data:
            continue

        rows = grouped_data[noise]

        # Compute mean and std for each metric for each algorithm
        computed_rows = []
        for row in rows:
            algo = row[1]
            costs_mean, costs_std = np.mean(row[9]), np.std(row[9])
            success_mean, success_std = np.mean(row[5]), np.std(row[5])
            returns_mean, returns_std = np.mean(row[7]), np.std(row[7])
            total_steps_mean, total_steps_std = np.mean(row[11]), np.std(row[11])
            computed_rows.append([algo, costs_mean, costs_std, success_mean, success_std, returns_mean, returns_std, total_steps_mean, total_steps_std])

        # Find the highest/lowest values for the requested metrics
        min_cost = min(row[1] for row in computed_rows)
        max_success = max(row[3] for row in computed_rows)
        max_returns = max(row[5] for row in computed_rows)
        min_steps = max(row[7] for row in computed_rows)

        first_row = True
        for row in computed_rows:
            algo = row[0].replace("_", " ").upper()
            costs = f"{row[1]:.2f} $\\pm$ {row[2]:.2f}"
            success = f"{row[3]:.2f} $\\pm$ {row[4]:.2f}"
            returns = f"{row[5]:.2f} $\\pm$ {row[6]:.2f}"
            total_steps = f"{row[7]:.2f} $\\pm$ {row[8]:.2f}"

            # Bold the highest/lowest values as required
            if row[1] == min_cost:
                costs = f"\\textbf{{{costs}}}"
            if row[3] == max_success:
                success = f"\\textbf{{{success}}}"
            if row[5] == max_returns:
                returns = f"\\textbf{{{returns}}}"
            if row[7] == min_steps:
                total_steps = f"\\textbf{{{total_steps}}}"

            # Format the noise level only in the first row of each group
            if first_row:
                latex_table += f"{noise} & {algo} & {costs} & {success} & {returns} & {total_steps} \\\\\n"
                first_row = False
            else:
                latex_table += f" & {algo} & {costs} & {success} & {returns} & {total_steps} \\\\\n"

        latex_table += "\\midrule\n"

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n\\caption{Evaluation of Models by Noise Levels.}\n\\label{tab:noise_evaluation}\n\\end{table}"

    return latex_table


def get_last_total_env_interacts(csv_path):
    try:
        df = pd.read_csv(csv_path, delim_whitespace=True)
        last_entry = df["TotalEnvInteracts"].iloc[-1]
        return int(last_entry)
    except Exception as e:
        return -1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=5000)
    parser.add_argument('--episodes', '-n', type=int, default=1)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--save', '-s', type=str, default="")
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    algorithms = ["trpo_minmax", "trpo", "trpo_lagrangian", "cpo"]
    seeds = ["s0","s2","s3","s4","s5","s7","s8","s9","s10","s11"]

    results=[]
    for algo in algorithms:
        for folder_name in os.listdir(args.fpath+"/"+algo):
            folder_path = os.path.join(args.fpath+"/"+algo, folder_name)
            noise = 0
            if "PointCustom0" in folder_path: noise = 0
            if "PointCustom3" in folder_path: noise = 2.5
            if "PointCustom1" in folder_path: noise = 5
            if "PointCustom4" in folder_path: noise = 7.5
            if "PointCustom2" in folder_path: noise = 10
            print(noise, folder_path)
            successes, returns, costs, total_steps = [], [], [], []
            for folder_name_ in os.listdir(folder_path):
                folder_path_ = os.path.join(folder_path, folder_name_)
                seed = int(folder_name_.split("_")[-1][1:])
                if get_last_total_env_interacts(folder_path_+"/progress.txt")<9e6: continue

                env, get_action, sess = load_policy(folder_path_, 'last', args.deterministic)
                successes_, returns_, costs_, total_steps_ = evaluate_policy(env, get_action, num_episodes=100, noise=noise, seed=seed)
                # print(successes_, returns_, costs_, total_steps_)
                successes+=successes_; returns+=returns_; costs+=costs_; total_steps+=total_steps_

            results.append(["algo", algo , "noise", noise, "successes", successes, "returns", returns, "costs", costs, "total_steps", total_steps])
    
    table = generate_latex_table(results)
    print("\n", table)