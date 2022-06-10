"""Visualize various data collected from given query session."""
import pdb
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from inquire.utils.datatypes import Modality


def save_data(
    data: Union[list, np.ndarray],
    labels: list,
    num_runs: int,
    directory: str,
    filename: str,
    subdirectory: str = None,
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
    if subdirectory != None:
        path = Path(directory) / Path(subdirectory)
    else:
        path = Path(directory)
    if not path.exists():
        path.mkdir(parents=True)
    df = pd.DataFrame(
        data_stack.reshape(-1, data_stack.shape[-1]), index=index
    )
    final_path = path / Path(filename)
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
            return df
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

def convert_x_to_cost_axis(main_data, query_data, costs) -> pd.DataFrame:
    max_cost = max([int(i) for i in costs.values()])
    max_queries = int(main_data.columns[-1])
    new_cols = max_cost * max_queries
    new_columns = pd.Index(list(main_data.columns[:4].values) + list(range(new_cols)))
    converted_data = np.zeros((main_data.shape[0], new_cols))
    for row in range(main_data.shape[0]):
        cost_idx = 0
        for col in range(4,main_data.shape[1]):
            q = query_data.iat[row,col]
            cost_idx += costs[Modality(int(q))]
            for i in range(cost_idx, new_cols):
                converted_data[row, i] = main_data.iat[row, col]
    converted_dict = dict()
    for col in range(4):
        converted_dict[main_data.columns[col]] = main_data.loc[:,main_data.columns[col]]
    for col in range(new_cols):
        converted_dict[col] = converted_data[:,col]
    return pd.DataFrame(data=converted_dict)

def save_plot(data, labels, y_label, y_range, directory, filename, subdirectory=None):
    colors = ["r", "b", "g", "c", "m", "y", "k"]
    if subdirectory != None:
        path = Path(directory) / Path(subdirectory)
    else:
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
        final_path = path / Path(filename)
        plt.savefig(final_path)


def plot_data(inputs: dict) -> None:
    """Chooose data to plot and how to plot it."""
    type_of_plot = inputs["plot_type"].lower()
    try:
        assert Path(inputs["directory"]).exists()
        if type_of_plot == "distance" or type_of_plot == "performance" or type_of_plot == "cost":
            try:
                plot_performance_or_distance(
                    directory=inputs["directory"], file=inputs["file"], title=inputs["plot_title"]
                )
            except KeyError:
                plot_performance_or_distance(
                    directory=inputs["directory"], title=inputs["plot_title"]
                )
            except KeyError:
                plot_performance_or_distance(directory=inputs["directory"], file=inputs["file"])
            except KeyError:
                plot_performance_or_distance(directory=inputs["directory"])
        elif type_of_plot == "dempref":
            try:
                dempref_viz(directory=inputs["directory"], number_of_demos=inputs["number_of_demos"])
            except KeyError:
                print("DemPref visuals need list-argument: number_of_demos.")
        else:
            print(f"Couldn't handle type_of_plot: {type_of_plot}")
            return
    except AssertionError:
        print(f"Couldn't find alleged data location: {directory}.")
        return


def dempref_viz(directory: str, number_of_demos: list) -> None:
    """View data in manner of DemPref paper."""
    path = Path(directory)
    colors = ["#F19837", "#327ECC", "#9C9FA0"]
    fig = go.Figure()
    df = pd.DataFrame()
    for DEMPREF in number_of_demos:
        file = f"lander_{DEMPREF}_demos_dempref_metric.csv"
        db, file_names = get_data(file=None, directory=path)
        db = db[0]
        label = r"$n_{dem}$ = " + DEMPREF
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
    directory: str = None, file: str = None, title: str = ""
) -> None:
    """See reward and distance-from-ground-truth over subsequent queries."""
    dataframes, file_names = get_data(file=file, directory=directory)
    for i, file in enumerate(file_names):
        file_names[i] = (
            file.split("/")[-1]
            .replace("_", ", ")
            .replace(".csv", "")
            .capitalize()
        )
    fig = go.Figure()
    fig.update_layout(title=title)
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
                        line_width=3,
                    )
                )
    fig.show()
