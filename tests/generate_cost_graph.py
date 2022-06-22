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
domains=["linear_combo", "linear_system", "lander", "pizza"]
statics=[[False],[True,False],[True,False],[True,False]]

costs = {Modality.NONE: 0, Modality.DEMONSTRATION: 20, Modality.PREFERENCE: 10, Modality.CORRECTION: 15, Modality.BINARY: 5}

types = ["dempref", "inquire-weighted","pref", "corr", "demo", "bnry", "weighted-inquire"]
def main():
    for d in range(len(domains)):
        domain = domains[d]
        static_vals = statics[d]
        for static in static_vals:
            if static:
                static_name="static_"
            else:
                static_name=""
            directory = "output/static_betas_results/" + static_name + domain + "/"
            for t in types:
                if t == "dempref":
                    prefix = "--" + static_name + domain + "_"
                elif domain == "pizza":
                    prefix = "--" + static_name + domain + "_alpha-0.001_"
                else:
                    prefix = "--" + static_name + domain + "_alpha-0.005_"
                p = Path(directory + t + prefix + "distance.csv")
                if p.is_file():
                    main_data = get_data(file=t+prefix+"distance.csv", directory=directory)[0]
                    query_data = get_data(file=t+prefix+"query_types.csv", directory=directory)[0]
                    converted_data = convert_x_to_cost_axis(main_data, query_data, costs)
                    converted_data.to_csv(directory + t + prefix + "cost.csv")

if __name__ == "__main__":
    main()
