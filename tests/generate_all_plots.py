"""Visualize data from multiple files on same plot.

A few things to note:

- Ensure:
  - the pertinent data is located in a single directory
  - data format is consistent across files

- type_of_plot argument can be:
  - distance
  - performance
  - dempref

  Note: Choose one of distance or performance even if both types of data
        are within your directory; the plotting can handle that.
"""
import pdb
from data_utils import plot_data

all_plot_types = ["performance", "distance", "cost"]
domains=["linear_combo", "linear_system", "lander"]
names=["Parameter Estimation", "Linear Dynamical System", "Lunar Lander"]
statics=[[False],[True,False],[True,False]]
types=[["performance","distance"], all_plot_types, all_plot_types]

""" Optional arguments: """
file_name = ""  # If plotting data from a single .csv
base_directory = "output/static_betas_results/"

def main():
    """Run the program."""
    for i in range(len(domains)):
        domain = domains[i]
        name = names[i]
        static_vals = statics[i]
        plot_types=types[i]
        for static in static_vals:
            for plot_type in types[i]:
                plot_title = name + " - " + plot_type.capitalize()
                if static:
                    static_name="static_"
                    plot_title = "Static " + plot_title
                else:
                    static_name=""
                plot_title = "<b>" + plot_title + "</b>"
                directory = base_directory + static_name + domain + "/"
                args={"directory":directory, "plot_type":plot_type, "save":False, "title":plot_title, "show_plot":False}
                fig = plot_data(args)
                fig.write_image(base_directory + static_name + domain + "_" + plot_type + ".png", width=1250, height=950, scale=2)


if __name__ == "__main__":
    main()
