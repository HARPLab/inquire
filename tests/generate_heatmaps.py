import pdb
from data_utils import *
import plotly.graph_objs as go

domains=["linear_combo", "linear_system", "lander", "pizza"]
names=["Param. Estimation", "Linear Dyn. System", "Lunar Lander", "Pizza Arrangement"]
statics=[[False],[True,False],[True,False], [True,False]]
all_agents=[["INQUIRE","inquire"], ["Weighted INQUIRE","inquire-weighted"]]
""" Optional arguments: """
file_name = ""  # If plotting data from a single .csv
base_directory = "output/static_betas_results/"
colorbar=False
def main():
    """Run the program."""
    for i in range(len(domains)):
        domain = domains[i]
        name = names[i]
        static_vals = statics[i]
        for static in static_vals:
            for agent in all_agents:
                if static:
                    static_name="static_"
                    plot_title = name + "<br>(Static State)"
                else:
                    static_name=""
                    plot_title = name + "<br>(Changing State)"
                if "combo" in domain:
                    plot_title = name + "<br>(Static State)"

                plot_title = "<b>" + plot_title + "</b>"
                directory = base_directory + static_name + domain + "/"
                if "pizza" in domain:
                    query_file = agent[1] + "--" + static_name + domain + "_alpha-0.001_query_types.csv"
                else:
                    query_file = agent[1] + "--" + static_name + domain + "_alpha-0.005_query_types.csv"
                query_data = get_data(file=query_file, directory=directory)
                counts = get_query_counts(query_data[0])
                if colorbar:
                    l_margin = 50
                    title_x=0.21
                else:
                    l_margin = 10
                    title_x=0
                data=[go.Heatmap(showscale=colorbar,z=counts,ygap=4,colorscale="reds",colorbar=dict(title="Freq.",x=-0.25))]
                layout = go.Layout(
                    font=dict(size=90,color="black"),
                    title=dict(
                        text=plot_title,
                        xanchor="left",
                        x=title_x,
                        y=0.94,
                        #font=dict(size=90)
                    ),
                    xaxis = dict(
                        automargin= True,
                        #tickfont=dict(size=50,color="black"),
                        title=dict(
                          text="<b>Query #</b>",
                          standoff= 20
                        ),
                        range=(0.5,20.5)
                    ),
                    yaxis=dict(
                        visible=colorbar,
                        #tickfont=dict(size=70,color="black"),
                        tickmode="array",
                        tickvals=[0,1,2,3],
                        ticktext=["<b>Demo.</b>", "<b>Pref.</b>", "<b>Corr.</b>", "<b>Binary</b>"],
                        tickangle=270,
                    ),
                    margin=dict(
                        l=5,
                        r=5,
                        t=300,
                        pad=5,
                    ),
                    paper_bgcolor="white",
                    plot_bgcolor="black"
                )
                fig = go.Figure(data=data, layout=layout)
                if colorbar:
                    fig.write_image(base_directory + agent[1] + "_" + static_name + domain + "_query_types_colorbar.png", width=1950, height=1700, scale=2)
                else:
                    fig.write_image(base_directory + agent[1] + "_" + static_name + domain + "_query_types.png", width=1350, height=1650, scale=2)

if __name__ == "__main__":
    main()
