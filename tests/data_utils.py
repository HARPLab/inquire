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


def load_data(directory, filename):
    df = pd.read_csv(directory + '/' + filename)
    pdb.set_trace()

def save_data(
    data: list, labels: list, num_runs: int, directory: str, filename: str
) -> None:
    """Save data to file in directory."""
    agents = labels
    data_stack = np.stack(data, axis=1)
    tasks = [i for i in range(data_stack.shape[0])]
    runs = [i for i in range(num_runs)]
    test_states = [i for i in range(data_stack.shape[3])]
    queries = [i for i in range(data_stack.shape[-1])]
    index = pd.MultiIndex.from_product(
        [tasks, agents, runs, test_states], #queries, test_states],
        names=["task", "agent", "run", "test state"], #"query", "test state"],
    )
    path = Path(directory)
    if not path.exists():
        path.mkdir(parents=True)
    df = pd.DataFrame(data_stack.reshape(-1, data_stack.shape[-1]), index=index)
    df.to_csv(directory + "/" + filename)
    return df

def save_plot(data, labels, y_label, y_range, directory, filename):
    colors = ["r", "b", "g", "c", "m", "y", "k"]
    path = Path(directory)
    if not path.exists():
        path.mkdir(parents=True)

    # For each agent:
    for a in range(len(data)):
        task_mat = data[a]
        x, med, err = [],[],[]
        # For each task:
        for q in range(task_mat.shape[-1]):
            x.append(q + (0.05 * a))
            # Get the median across each query's runs*tests:
            data_pt = task_mat[:,:,:,q].flatten()
            med.append(np.median(data_pt))
            # Define error as
            if data_pt.shape[0] > 2:
                err.append(abs(np.percentile(data_pt, (25, 75)) - med[-1]))
        if len(err) > 0:
            err = np.array(err)
        else:
            err = None
        plt.errorbar(
            np.array(x),
            np.array(med),
            fmt=".-",
            yerr=np.array(err).T,
            color=colors[a % len(colors)],
            label=labels[a],
        )
        plt.xlabel("# of queries")
        plt.ylabel(y_label)
        plt.ylim(y_range[0], y_range[1])
        plt.xticks(range(task_mat.shape[-1]))
        plt.savefig(directory + '/' + filename)


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
