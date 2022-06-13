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
import argparse

from data_utils import plot_data


def get_args() -> argparse.ArgumentParser:
    """Get command-line inputs.

    Required arguments:
    - directory = args.path
    - type_of_plot = args.plot_type

    Optional arguments:
    - file_name = args.file
    - plot_title = args.title
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", "--file", type=str, default=None)
    parser.add_argument("-t", "--title", type=str, default="")
    parser.add_argument(
        "-T",
        "--plot_type",
        type=str,
        default="distance",
        choices=["dempref", "distance", "performance"],
    )
    parser.add_argument("-D", "--directory", type=str, default="output/")
    parser.add_argument("-N", "--number_of_demos", type=str, default="0,1,3")
    parser.add_argument("-S", "--save", action="store_true", default=False)
    parser.add_argument("--show_plot", action="store_true", default=True)

    return parser.parse_args()


def main():
    """Run the program."""
    args = get_args()

    plot_data(args.__dict__)


if __name__ == "__main__":
    main()
