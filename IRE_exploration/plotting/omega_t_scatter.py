import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from IRE_Bprobe_detection import load_IRE_data


def plot_omega_t_scatter(pkl_file_loc="/home/jupyter-humerben/axuv_paper/IRE_exploration/IRE_Bprobe_detection/data/output.pkl", matplotlib=True, plotly=True):
    df = load_IRE_data()
    shots = np.unique(df.shot)
    
    # initialize arrays
    omega = []
    t = []
    shot_plotting = []
    for shot in shots:
        df_ss = df[df.shot == shot]
        if max(df_ss.omega) <= 0:
            omega.append(np.mean(df_ss.omega))
            t.append(df_ss.t.iloc[0])
            shot_plotting.append(int(shot))
    
    if matplotlib:
        plt.scatter(t, omega, zorder=3)
        plt.title(r"$\omega(t)$ for the Toroidally Rotating Fluctuation Associated with IREs")
        plt.ylabel(r"$\omega\,\mathrm{[rad/s]}$")
        plt.xlabel(r"$t\,\mathrm{[ms]}$")
        plt.tight_layout()
        plt.grid(alpha=0.4, zorder=0)
        plt.savefig("test.png")
        plt.close()
        
    if plotly:
        fig = go.Figure(
            go.Scatter(x=t, y=omega, mode='markers', marker=dict(symbol='circle'),
                       hoverinfo='x,y,text', hovertext=shot_plotting)
        )
        
        fig.update_layout(title="Angular frequency of the Toroidally Rotating Fluctuation Associated with IREs",
                          xaxis=dict(title=r"$t\,\mathrm{[ms]}$"), yaxis=dict(title=r"$\omega(t)$")
        )
        fig.show()
    