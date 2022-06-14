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
types=[all_plot_types, all_plot_types, all_plot_types]
all_agents=["Binary-only", "Corr-only", "Demo-only", "Pref-only", "INQUIRE"]
""" Optional arguments: """
file_name = ""  # If plotting data from a single .csv
base_directory = "output/static_betas_results/"
def main():
    auc_header = "Agent"
    auc_rows=[dict(),dict(),dict()]
    """Run the program."""
    for i in range(len(domains)):
        domain = domains[i]
        name = names[i]
        static_vals = statics[i]
        for static in static_vals:
            if static:
                static_name="static_"
                plot_title = "Static " + plot_title
            else:
                static_name=""
            auc_header += ", " + static_name + domain
            for j in range(len(types[i])):
                plot_type = types[i][j]
                plot_title = name + " - " + plot_type.capitalize()
                plot_title = "<b>" + plot_title + "</b>"
                directory = base_directory + static_name + domain + "/"
                args={"directory":directory, "plot_type":plot_type, "save":False, "title":plot_title, "show_plot":False}
                fig, auc = plot_data(args)
                agents = list(auc.keys())
                for a in all_agents:
                    matches = [n for n in agents if a in n]
                    if len(matches) == 0:
                        auc_val = "n/a"
                    else:
                        auc_val = str(auc[matches[0]])
                    if a not in auc_rows[j].keys():
                        auc_rows[j][a] = a + ", " + auc_val
                    else:
                        auc_rows[j][a] += ", " + auc_val
                fig.write_image(base_directory + static_name + domain + "_" + plot_type + ".png", width=1250, height=950, scale=2)
    for k in range(len(all_plot_types)):
        plot = all_plot_types[k]
        f = open(plot + "-auc.csv", "a+")
        f.write(auc_header + "\n")
        rows = list(auc_rows[k].values())
        for row in rows:
            f.write(row + "\n")
        f.close()


if __name__ == "__main__":
    main()
