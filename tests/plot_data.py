"""Visualize data from multiple files on same plot."""
from data_utils import plot_performance_or_distance

# Ensure:
# - the pertinent data is located in a single directory
# - data format is consistent across files
directory = "output/test_dir/"


def main():
    """Run the program."""
    plot_performance_or_distance(directory)


if __name__ == "__main__":
    main()
