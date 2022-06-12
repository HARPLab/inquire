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
directory = "output/testdir/"
type_of_plot = "performance"

""" Optional arguments: """
file_name = ""  # If plotting data from a single .csv
plot_title = ""

costs = {Modality.NONE: 0, Modality.DEMONSTRATION: 20, Modality.PREFERENCE: 10, Modality.CORRECTION: 15, Modality.BINARY: 5}

prefix = "bnry--lander_alpha-0.005_"
def main():
    main_data = get_data(file=prefix+"distance.csv", directory=directory)
    query_data = get_data(file=prefix+"query_types.csv", directory=directory)
    converted_data = convert_x_to_cost_axis(main_data, query_data, costs)
    converted_data.to_csv(directory + prefix + "cost.csv")

if __name__ == "__main__":
    main()
