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
"""One example for evaluate saved policy."""

from collections import defaultdict
import json
import os
import pickle
import time

import numpy as np
from scipy.stats import ttest_ind

# from monkeypatch import monkeypatch

# import rosarl
# import rosarl.algorithms
# import rosarl.envs

# from evaluator import Evaluator

# monkeypatch()

noise_level_lookup = {
    "SafetyterminalgoalPointPillar0-v0": 0.0,
    "SafetyterminalgoalPointPillar1-v0": 2.5,
    "SafetyterminalgoalPointPillar2-v0": 5.0,
    "SafetyterminalgoalPointPillar3-v0": 7.5,
    "SafetyterminalgoalPointPillar4-v0": 10.0,
    "SafetyterminalgoalPointPillar5-v0": 0.5,
    "SafetyterminalgoalPointPillar6-v0": 1.0,
    "SafetyterminalgoalPointPillar7-v0": 1.5,
    "SafetyterminalgoalPointPillar8-v0": 2.0,
}


def find_best(samples, pvalue):
    pass


def generate_latex_table(data, algo_list, noise_order, bold_best):
    # Group data by noise level
    grouped_data = defaultdict(list)
    for row in data:
        noise = row[3]
        grouped_data[noise].append(row)

    latex_table = "\\begin{table}[h]\n\\centering\n"
    latex_table += "\\begin{tabular}{@{}l c c c c@{}}\n"
    latex_table += "\\toprule\n"
    latex_table += (
        "\\textbf{Algorithm} & \\textbf{Costs $\\downarrow$} & \\textbf{Success Rate $\\uparrow$} & "
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
            computed_rows.append(
                [
                    algo,
                    costs_mean,
                    costs_std,
                    success_mean,
                    success_std,
                    returns_mean,
                    returns_std,
                    total_steps_mean,
                    total_steps_std,
                ]
            )

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

            if bold_best:
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
                latex_table += f"{algo} & {costs} & {success} & {returns} & {total_steps} \\\\\n"
                first_row = False
            else:
                latex_table += f"{algo} & {costs} & {success} & {returns} & {total_steps} \\\\\n"

        latex_table += "\\midrule\n"

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n\\caption{Evaluation of Models for [ENVNAME].}\n\\label{tab:noise_evaluation}\n\\end{table}"

    return latex_table


# Just fill your experiment's log directory in here.
# LOG_DIR = (
#     "/home/mark/projects/rosarl/scripts/runs/cluster/pointpillar-goal-unsafe-terminal"
# )
if __name__ == "__main__":
    # log_dir = "/home/mark/projects/rosarl/scripts/runs/cluster/pointgoal1-default"
    # log_dir = "/home/mark/projects/rosarl/scripts/runs/cluster/pointgoal1-unsafe-terminal"
    # log_dir = "/home/mark/projects/rosarl/scripts/runs/cluster/pointgoal1-goal-unsafe-terminal"
    # log_dir = "/home/mark/projects/rosarl/scripts/runs/cluster/pointpush1-unsafe-terminal"
    log_dir = "/home/mark/projects/rosarl/scripts/runs/cluster/pointpush1-goal-unsafe-terminal"
    algo_list = ["TRPOMinmax", "PPOMinmax", "TRPOLag", "TRPOSaute", "TRPO" , "CPO", "P3O"]

    loaded_results = {}
    for name in os.listdir(log_dir):
        file_path = os.path.join(log_dir, name)
        if not os.path.isfile(file_path) or name[-3:] != "pkl":
            continue
        print(f"Loading {file_path}")
        with open(file_path, "rb") as handle:
            loaded_data = pickle.load(handle)
        
        loaded_results[loaded_data[0][1]] = loaded_data

    sorted_results = []
    for algo in algo_list:
        sorted_results.extend(loaded_results[algo])

    print("Generating table")
    latex_table = generate_latex_table(sorted_results, algo_list, ["X"], False)
    print("\n", latex_table)

