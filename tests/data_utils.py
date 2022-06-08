"""Visualize various data collected from given query session."""
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def save_data(
    data: Union[list, np.ndarray],
    labels: list,
    num_runs: int,
    directory: str,
    filename: str,
) -> None:
    """Save data to file in directory."""
    agents = labels
    data_stack = np.stack(data, axis=1)
    tasks = [i for i in range(data_stack.shape[0])]
    runs = [i for i in range(num_runs)]
    test_states = [i for i in range(data_stack.shape[3])]
    index = pd.MultiIndex.from_product(
        [tasks, agents, runs, test_states],
        names=["task", "agent", "run", "test_state"],
    )
    path = Path(directory)
    if not path.exists():
        path.mkdir(parents=True)
    df = pd.DataFrame(
        data_stack.reshape(-1, data_stack.shape[-1]), index=index
    )
    final_path = directory + "/" + filename
    df.to_csv(final_path)
    print(f"Data saved to {final_path}")
    return df


def get_data(
    file: str, directory: str, combine_into_file: bool = False
) -> pd.DataFrame:
    """Fetch data in directory/file."""
    path = Path(directory)
    df = pd.DataFrame()
    if file != None:
        file = Path(file)
        try:
            df = pd.read_csv(path / file)
        except:
            print(f"Couldn't read from {str(path / file)}")
    else:
        files = np.array(list(Path.iterdir(path)))
        df = pd.DataFrame()
        if combine_into_file:
            for f in files:
                try:
                    df = pd.concat([df, pd.read_csv(f)], ignore_index=True)
                except:
                    print(f"Couldn't read from {str(f)}")
            return df, files
        else:
            dataframes = []
            for f in files:
                try:
                    dataframes.append(pd.read_csv(f))
                except:
                    print(f"Couldn't read from {str(f)}")
            return dataframes, files.astype(str)


def save_plot(data, labels, y_label, y_range, directory, filename):
    colors = ["r", "b", "g", "c", "m", "y", "k"]
    path = Path(directory)
    if not path.exists():
        path.mkdir(parents=True)

    # For each agent:
    for a in range(len(data)):
        task_mat = data[a]
        x, med, err = [], [], []
        # For each task:
        for q in range(task_mat.shape[-1]):
            x.append(q + (0.05 * a))
            # Get the median across each query's runs*tests:
            data_pt = task_mat[:, :, :, q].flatten()
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
        plt.savefig(directory + "/" + filename)


def dempref_viz(directory: str, number_of_demos: list) -> None:
    """View data in manner of DemPref paper."""
    path = Path(directory)
    colors = ["#F19837", "#327ECC", "#9C9FA0"]
    fig = go.Figure()
    df = pd.DataFrame()
    for DEMPREF in number_of_demos:
        file = f"lander_{DEMPREF}_demos.csv"
        db, file_names = get_data(file, path)
        label = "$n_{dem}$ = " + str(DEMPREF)
        db["dempref"] = label
        df = pd.concat([df, db], ignore_index=True)
    number_of_queries = int(df.columns[-2])
    x_axis = np.arange(number_of_queries)
    means = {}
    std_devs = {}
    for DEMPREF in number_of_demos:
        group = df[df.dempref == "$n_{dem}$ = " + str(DEMPREF)]
        group_means = []
        group_std_devs = []
        for i in range(number_of_queries + 1):
            group_means.append(group[str(i)].mean())
            group_std_devs.append(group[str(i)].std())

        means[DEMPREF] = np.array(group_means)
        std_devs[DEMPREF] = np.array(group_std_devs)
    for j, DEMPREF in enumerate(number_of_demos):
        # Add trace of mean here to then properly include standard deviations:
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=means[DEMPREF],
                line_color="red",
                line_width=2,
                name=r"Mean $n_{DEMPREF}$",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=means[DEMPREF] + std_devs[DEMPREF],
                fill="tonexty",
                fillcolor=colors[2 - j],
                line_color=colors[2 - j],
                name=r"Std. dev. $n_{DEMPREF}$ (+)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=means[DEMPREF] - std_devs[DEMPREF],
                fill="tonexty",
                fillcolor=colors[2 - j],
                line_color=colors[2 - j],
                name=r"Std. dev. $n_{DEMPREF}$ (-)",
            )
        )
        # Add another trace of the mean so it can be seen:
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=means[DEMPREF],
                line_color="red",
                line_width=2,
                name=r"Mean $n_{DEMPREF}$",
            )
        )
    fig.show()


def plot_performance_or_distance(
    directory: str = None, file: str = None
) -> None:
    """See reward and distance-from-ground-truth over subsequent queries."""
    dataframes, file_names = get_data(file=file, directory=directory)
    marker_colors = np.random.randint(0, 255, (len(dataframes), 3))
    fig = go.Figure()
    query_count = dataframes[0].columns[-1]
    x_axis = np.arange(int(query_count))
    for i, df in enumerate(dataframes):
        agents = df["agent"].unique()
        tasks = df["task"].unique()
        for agent in agents:
            b = df[df.agent == agent].loc[:, "0":query_count]
            for t in tasks:
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=b.mean(),
                        error_y=dict(type="data", array=b.var().values),
                        visible=True,
                        name=file_names[i],
                    )
                )
                # for c in b.columns:
                #    fig.add_trace(
                #        go.Box(
                #            y=b[c],
                #            fillcolor="rgb("
                #            + ",".join(marker_colors[i].astype(str))
                #            + ")",
                #        )
                #    )
    fig.show()
