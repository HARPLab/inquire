"""Visualize various data collected from given query session."""
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go


def save_data(data: list, labels: list, directory: str, file: str) -> None:
    """Save data to file in directory."""
    agents = labels
    data_stack = np.stack(data, axis=1)
    tasks = [i for i in range(data_stack.shape[0] - 1)]
    queries = [i for i in range(data_stack.shape[-1])]
    index = pd.MultiIndex.from_product([agents, tasks, queries], names=["agent","task","query"])
    breakpoint()
    df = pd.DataFrame(data_stack.squeeze(), index=index)
    df.to_csv(directory + file)


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
            series = np.transpose(task_mat[t, a])
            label = labels[a]
            x = [i + 1 + (0.05 * a) for i in range(series.shape[0])]
            med = np.median(series, axis=1)
            err = abs(np.percentile(series, (25, 75), axis=1) - med)
            plt.errorbar(
                x,
                med,
                fmt=".-",
                yerr=err,
                color=colors[a % len(colors)],
                label=label,
            )
        plt.legend(labels)
        plt.xticks(np.arange(1, task_mat.shape[-1] + 1, 1.0))
        plt.savefig(output_dir + filename + "-task_" + str(t) + ".png")
        plt.clf()


def plot_performance_distance_matrices(
    labels, directory: str, file: str
) -> None:
    """See reward and distance-from-ground-truth over subsequent queries."""
    colors = ["r", "b", "g", "c", "m", "y", "k"]
    file_path = Path(directory + "/" + file)
    if not file_path.exists():
        print(f"The path {file_path} doesn't exist.")
        return

    labels = list(labels)
    data = pd.read_csv(file_path)
    fig = go.Figure()
    task_mat = data.to_numpy()
    task_mat = np.stack(task_mat, axis=1)
    series = task_mat.T
    x = [i + 1 + (0.05 * 1) for i in range(series.shape[0])]
    med = np.median(series, axis=1)
    err = abs(np.percentile(series, (25, 75), axis=1) - med)
    fig.add_trace(go.Box(y=err))
    plt.errorbar(
        x, med, fmt=".-", yerr=err, color=colors[1 % len(colors)], label=labels
    )
    plt.legend(labels)
    plt.xticks(np.arange(1, task_mat.shape[-1] + 1, 1.0))
    plt.show()
    fig.show()


if __name__ == "__main__":
    plot_performance_distance_matrices(
        labels="inquire_agent",
        directory="output",
        file="05:03:22:03:52_performance_data_linear_system.csv",
    )
