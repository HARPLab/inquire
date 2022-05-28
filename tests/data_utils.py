"""Visualize various data collected from given query session."""
import pdb
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go


def save_data(
    data: list, labels: list, num_runs: int, directory: str, filename: str
) -> None:
    """Save data to file in directory."""
    agents = labels
    data_stack = np.stack(data, axis=1)
    tasks = [i for i in range(data_stack.shape[0])]
    runs = [i for i in range(num_runs)]
    test_count = data_stack.shape[2] / num_runs
    assert (
        test_count - int(test_count) == 0
    ), f"The test count ({test_count}) needs to be a whole number."
    test_states = [i for i in range(int(data_stack.shape[2] / num_runs))]
    queries = [i for i in range(data_stack.shape[-1])]
    index = pd.MultiIndex.from_product(
        [tasks, agents, runs], #queries, test_states],
        names=["task", "agent", "run"], #"query", "test state"],
    )
    path = Path(directory)
    if not path.exists():
        path.mkdir(parents=True)
    df = pd.DataFrame(data_stack.reshape(-1, data_stack.shape[-1]), index=index)
    df.to_csv(directory + "/" + filename)

def og_plot_results(results, labels, dir_name, filename):
    colors = ["r", "b", "g", "c", "m", "y", "k"]
    task_mat = np.stack(results, axis=1)
    file_path = os.path.realpath(__file__)
    output_dir = os.path.dirname(file_path) + "/" + dir_name + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # For each task:
    for t in range(task_mat.shape[0]):
        # For each agent:
        for a in range(task_mat.shape[1]):
            # Get the #query-by-#(runs*tests) matrix:
            series = np.transpose(task_mat[t, a])
            label = labels[a]
            x = [i + 1 + (0.05 * a) for i in range(series.shape[0])]
            # Get the median across each query's runs*tests:
            med = np.median(series, axis=1)
            # Define error as
            err = abs(np.percentile(series, (25, 75), axis=1) - med)
            plt.errorbar(
                x,
                med,
                fmt=".-",
                yerr=err,
                color=colors[a % len(colors)],
                label=label,
            )


def plot_performance_distance_matrices(
    directory: str = None, file: str = None
) -> None:
    """See reward and distance-from-ground-truth over subsequent queries."""
    if directory is not None:
        file_path = Path(directory + "/" + file)
        if not file_path.exists():
            print(f"The path {file_path} doesn't exist.")
            return
        else:
            df = pd.read_csv(file_path)
    else:
        print(
            "Need to provide a path to the directory and the pertinent "
            "file's name."
        )
    agents = list(df.index.levels[1])
    tasks = list(df.index.levels[0])
    fig = go.Figure()
    for agent in agents:
        b = df.loc[agent].reset_index()
        for t in tasks:
            fig.add_trace(go.Box(x=b["query"], y=b[t]))
    fig.show()


#if __name__ == "__main__":
#    plot_performance_distance_matrices(
#        labels="inquire_agent",
#        directory="output",
#        file="05:03:22:03:52_performance_data_linear_system.csv",
#    )
