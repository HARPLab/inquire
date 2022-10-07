"""Visualize various data collected from given query session."""
import pdb
import time
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from inquire.utils.datatypes import Modality


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
            return df, file
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

def get_query_counts(query_data) -> np.ndarray:
    counts = np.zeros((4,21))
    for col in range(1,21):
        for row in query_data[str(col)]:
            counts[int(row),col] += 1
    return counts/np.max(np.sum(counts,axis=0))

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

def save_plot(
    data, labels, y_label, y_range, directory, filename, subdirectory=None
):
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
    plot_type = inputs["plot_type"].lower()
    try:
        assert Path(inputs["directory"]).exists()
        if plot_type == "distance" or plot_type == "performance" or plot_type == "cost":
            if plot_type == "cost":
                x_axis = "Accumulated Query Cost"
                y_axis = "Distance from w*"
            else:
                x_axis = "# of Queries"
                if plot_type == "distance":
                    y_axis = "Distance from w*"
                else:
                    y_axis = "Task Performance"
            
            try:
                return generate_plot(
                    plot_type=plot_type,
                    directory=inputs["directory"],
                    file=inputs["file"],
                    title=inputs["title"],
                    save=inputs["save"],
                    show_plot=inputs["show_plot"],
                    x_axis_label = x_axis,
                    y_axis_label = y_axis
                )
            except KeyError:
                return generate_plot(
                    plot_type=plot_type,
                    directory=inputs["directory"],
                    title=inputs["title"],
                    save=inputs["save"],
                    show_plot=inputs["show_plot"],
                    x_axis_label = x_axis,
                    y_axis_label = y_axis
                )
            except KeyError:
                return generate_plot(
                    plot_type=plot_type,
                    directory=inputs["directory"],
                    file=inputs["file"],
                    save=inputs["save"],
                    show_plot=inputs["show_plot"],
                    x_axis_label = x_axis,
                    y_axis_label = y_axis
                )
            except KeyError:
                return generate_plot(
                    plot_type=plot_type,
                    directory=inputs["directory"], 
                    save=inputs["save"],
                    show_plot=inputs["show_plot"],
                    x_axis_label = x_axis,
                    y_axis_label = y_axis
                )
        elif plot_type == "dempref":
            try:
                dempref_viz(
                    directory=inputs["directory"],
                    number_of_demos=inputs["number_of_demos"],
                    title=inputs["title"],
                    save=inputs["save"],
                )
            except KeyError:
                print("DemPref visuals need list-argument: number_of_demos.")
        else:
            print(f"Couldn't handle plot_type: {plot_type}")
            return
    except AssertionError:
        print(f"Couldn't find alleged data location: {inputs['directory']}.")
        return


def dempref_viz(
    directory: str,
    number_of_demos: list,
    title: str = None,
    save: bool = False,
) -> None:
    """View data in manner of DemPref paper."""
    if type(number_of_demos) == str:
        number_of_demos = number_of_demos.split(",")
    path = Path(directory)
    colors = ["#F19837", "#327ECC", "#9C9FA0"]
    fig = go.Figure()
    fig.update_layout(title=title)
    df = pd.DataFrame()
    for DEMPREF in number_of_demos:
        db, file_name = get_data(
            file=f"no_bias_lander_{DEMPREF}_demos_dempref_metric.csv", directory=path
        )
        label = r"$n_{dem}$ = " + str(DEMPREF)
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
            group_std_devs.append(group[str(i)].std() / np.sqrt(8))

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
    if save:
        print_time = time.strftime("%d:%m:%H:%M", time.localtime())
        fig.write_image(directory + f"/dempref_{print_time}.png")


def generate_plot(
        directory: str = None, file: str = None, title: str = "", plot_type="distance", x_axis_label="", y_axis_label="", show_plot=True, save: bool = False
) -> None:
    """See reward and distance-from-ground-truth over subsequent queries."""
    dataframes, file_names = get_data(file=file, directory=directory)
    data_dict = dict(zip(file_names,dataframes))
    if plot_type == "cost":
        filtered_file_names = sorted([f for f in file_names if plot_type in f])
    else:
        filtered_file_names = sorted([f for f in file_names if plot_type in f and "weighted" not in f])
    #For readability, move inquire to last object
    for f in range(len(filtered_file_names)):
        if "inquire" in filtered_file_names[f]:
            filtered_file_names.append(filtered_file_names.pop(f))
    full_name={"dempref": "DemPref", "bnry": "Binary-Only", "corr": "Corrections-Only", "demo": "Demos-Only", "pref": "Preferences-Only", "inquire": "INQUIRE", "inquire-weighted": "INQUIRE"}
    for i, file in enumerate(filtered_file_names):
        original_name = file
        filtered_file_names[i] = (
            file.split("/")[-1]
            .replace("_", ", ")
            .replace(".csv", "")
            .capitalize()
        )
        data_dict[filtered_file_names[i]] = data_dict[original_name]
    fig = go.Figure()
    if plot_type == "cost":
        query_count = '200'
    else:
        query_count = '20'
    x_axis = np.arange(int(query_count) + 1)
    colors = px.colors.sequential.Viridis
    auc_dict = dict()
    for i, filename in enumerate(filtered_file_names):
        df = data_dict[filename]
        agents = df["agent"].unique()
        tasks = df["task"].unique()
        for agent in agents:
            if "combo" in filename and filename > "Demo":
                color_offset = 2
            else:
                color_offset = 0
            color_idx = color_offset+(i*2)
            if color_idx >= 10:
                color_idx = 9
            b = df[df.agent == agent].loc[:, "0":query_count]
            for t in tasks:
                auc_dict[agent + "-task_" + str(t)] = b.mean()[:int(query_count)+1].sum()
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=b.mean(),
                        error_y=dict(type="data", array=b.var().values, thickness=7),
                        visible=True,
                        name=full_name[filtered_file_names[i].split("--")[0].lower()],
                        line_width=10,
                        marker_color=colors[color_idx],
                        marker=dict(
                            symbol="circle",
                            size=30)
                    ),
                )
    if plot_type == "distance":
        fig.update_layout(
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
    else:
        fig.update_layout(showlegend=False)
    fig.update_layout(
            title=title,
            legend=dict(
                itemwidth=50
            ),
            font=dict(size=40,color="black"),
            template='none',
            xaxis = dict(
                automargin= True,
                title=dict(
                  text="<b>" + x_axis_label + "</b>",
                  standoff= 20
                ),
            ),
            yaxis = dict(
                automargin= True,
                title=dict(
                  text="<b>" + y_axis_label + "</b>",
                  standoff= 60
                ),
            ),
            margin=dict(
                r=0
            ),
    )
    if plot_type == "performance":
        fig.update_yaxes(range=[0.4,1.0])
    else:
        fig.update_yaxes(range=[0,0.65])

    if save:
        print_time = time.strftime("%d:%m:%H:%M", time.localtime())
        fig.write_image(directory + f"/{title}_{print_time}.png")
    if show_plot:
        fig.show()
    return fig, auc_dict
