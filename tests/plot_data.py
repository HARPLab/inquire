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
from data_utils import plot_data


""" Required arguments: """
directory = "output/testdir/"
type_of_plot = "distance"

""" Optional arguments: """
file_name = ""  # If plotting data from a single .csv
plot_title = ""


def main():
    """Run the program."""
    plot_data(
        directory=directory, type_of_plot=type_of_plot, plot_title=plot_title
    )


if __name__ == "__main__":
    main()
