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
from data_utils import *
from inquire.utils.datatypes import Modality

""" Required arguments: """
static=False
domain="lander"
if static:
    static_name="static_"
else:
    static_name=""
directory = "output/static_betas_results/" + static_name + domain + "/"
type_of_plot = "performance"

""" Optional arguments: """
file_name = ""  # If plotting data from a single .csv
plot_title = ""

costs = {Modality.NONE: 0, Modality.DEMONSTRATION: 20, Modality.PREFERENCE: 10, Modality.CORRECTION: 15, Modality.BINARY: 5}

types = ["pref", "corr", "demo", "bnry", "weighted-inquire"]
prefix = "--" + static_name + domain + "_alpha-0.005_"
def main():
    for t in types:
        p = Path(directory + t + prefix + "distance.csv")
        if p.is_file():
            main_data = get_data(file=t+prefix+"distance.csv", directory=directory)
            query_data = get_data(file=t+prefix+"query_types.csv", directory=directory)
            converted_data = convert_x_to_cost_axis(main_data, query_data, costs)
            converted_data.to_csv(directory + t + prefix + "cost.csv")

if __name__ == "__main__":
    main()
