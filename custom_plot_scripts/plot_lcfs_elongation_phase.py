# import standard libraries
###################################################################################################
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.stats import norm


# import sandbox client
###################################################################################################
from gf_sandbox_client.sbc import Sbc

# import personal files
###################################################################################################
from data_loaders.load_crash_data import load_crash_data
import processing_tools.turn_time_to_crash_phase as time_to_phase


def plot_elongation_phase_slope_matplotlib(shots, t_lims=[0, 1000], sbc=Sbc()):
    # unpack t limits
    t_min, t_max = t_lims[0], t_lims[1]
    
    # prepare and unpack crash data
    crash_df = load_crash_data(shots)
    
    # load and set up multishot dataframes
        # set up SBC and load data
    sbc.experiment.set("pi3b")
    df_multishot = sbc.get(asset="reconstruction", shot=shots, column=['li1', 'elong_lfs'], criteria="statistic = mean OR statistic = sigma")

    
    x_plotting = []
    elong_lfs_plotting = []
    li1_plotting = []
    shots_plotting = []
    times_plotting = []
    
    for s in range(len(shots)):
        df = df_multishot[df_multishot['shot'] == shots[s]]
        df = df[df.statistic == 'mean']
        crash_df_ss = crash_df[crash_df.shot == shots[s]]
        lifetime = max(df.t+1/1000)
        df = df[(df.t >= t_min) & (df.t <= t_max)]
        # get times, li1, elong_lfs and plot
            # load scrambled arrays
        times = np.array(df.t)
        li1 = np.array(df.li1)
        elong_lfs = np.array(df.elong_lfs)
            # sort arrays
        times_ind = np.argsort(times)
        times = np.array([times[i] for i in times_ind])
        li1 = np.array([li1[i] for i in times_ind])
        elong_lfs = np.array([elong_lfs[i] for i in times_ind])
        phases = np.array([time_to_phase.get_phase_from_time_singlepoint_crash(crash_df_ss, i, lifetime) for i in times])
        
        d_times = np.array([(times[i]+times[i+1])/2 for i in range(len(times)-1)])
        d_li1 = np.array([(li1[i+1]-li1[i])/(phases[i+1]-phases[i]) for i in range(len(times)-1)])
        d_elong_lfs = np.array([(elong_lfs[i+1]-elong_lfs[i])/(phases[i+1]-phases[i]) for i in range(len(times)-1)])
        d_phases = np.array([(phases[i]+phases[i+1])/2 for i in range(len(times)-1)])
        
        if min(len(li1), len(elong_lfs), len(times)) > 0:
            x_plotting.append(d_phases)
            elong_lfs_plotting.append(d_elong_lfs)
            li1_plotting.append(d_li1)
            shots_plotting.append(shots[s])
            times_plotting.append(d_times)
    
    # make plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=400)  # Adjust size based on n

    # Ensure axs is always a 2D array
    axs = np.array([axs])  # Convert 1D to 2D array for consistent indexing
    
    # actually plot values
    for i in range(len(x_plotting)):
        # elongation plot
        axs[0,0].plot(x_plotting[i], elong_lfs_plotting[i], color='grey', alpha=0.3, zorder=1)
        axs[0,0].scatter(x_plotting[i], elong_lfs_plotting[i], alpha=0.7, zorder=2, marker='.')
        
        # inductance plot
        axs[0,1].plot(x_plotting[i], li1_plotting[i], color='grey', alpha=0.3, zorder=1)
        axs[0,1].scatter(x_plotting[i], li1_plotting[i], alpha=0.7, zorder=2, marker='.')
    
    # set labels
    labels_fontsize=14
        # elongation
    axs[0,0].set_ylabel(r"$d\kappa/dp$", fontsize=labels_fontsize)
    axs[0,0].set_xlabel("phase", fontsize=labels_fontsize)
    axs[0,0].set_xlim([0.25, 1.75])
    axs[0,0].grid(True, zorder=0)
        # inductance
    axs[0,1].set_ylabel(r"$d\ell_{i1}/dp$", fontsize=labels_fontsize)
    axs[0,1].set_xlabel("phase", fontsize=labels_fontsize)
    axs[0,1].set_xlim([0.25, 1.75])
    axs[0,1].grid(True, zorder=0)
    
    fig.suptitle(r"Derivatives of $\kappa$ and $\ell_{i1}$ Over Phase"+f"\n{len(shots)} Shots", fontsize=labels_fontsize+4)

    plt.savefig(f"/home/jupyter-humerben/axuv_paper/plot_outputs/elongation_li1_phase_{len(shots)}shots.png")



    
    
    
    
    return



def plot_elongation_phase(shots, t_lims=[0, 1000], sbc=Sbc(), save=False):
    t_min, t_max = t_lims[0], t_lims[1]
    
    # prepare and unpack crash data
    crash_df = load_crash_data(shots)
    
    # load and set up multishot dataframes
        # set up SBC and load data
    sbc.experiment.set("pi3b")
    df_multishot = sbc.get(asset="reconstruction", shot=shots, column=['li1', 'elong_lfs'], criteria="statistic = mean OR statistic = sigma")
    df_multishot = df_multishot[(df_multishot.t >= t_min) & (df_multishot.t <= t_max)]
    
    x_plotting = []
    y_plotting = []
    col_plotting = []
    shots_plotting = []
    times_plotting = []
    
    for s in range(len(shots)):
        df = df_multishot[df_multishot['shot'] == shots[s]]
        df = df[df.statistic == 'mean']
        crash_df_ss = crash_df[crash_df.shot == shots[s]]
        # get times, li1, elong_lfs and plot
            # load scrambled arrays
        times = np.array(df.t)
        li1 = np.array(df.li1)
        elong_lfs = np.array(df.elong_lfs)
            # sort arrays
        times_ind = np.argsort(times)
        times = np.array([times[i] for i in times_ind])
        li1 = np.array([li1[i] for i in times_ind])
        elong_lfs = np.array([elong_lfs[i] for i in times_ind])
        phases = np.array([time_to_phase.get_phase_from_time_singlepoint_crash(crash_df_ss, i) for i in times])
        
        if min(len(li1), len(elong_lfs), len(times)) > 0:
            x_plotting.append(phases)
            y_plotting.append(elong_lfs)
            shots_plotting.append(shots[s])
            times_plotting.append(times)



    fig = go.Figure()

    for i in range(len(shots_plotting)):
        fig.add_trace(go.Scatter(
            x=x_plotting[i],
            y=y_plotting[i],
            name=f"Shot: {shots_plotting[i]}",
            mode='lines',
            hovertext=[f"Shot: {shots_plotting[i]}, Time: {t*1000:.3f}ms" for t in times_plotting[i]],
            hoverinfo="x+y+text",
            # line=dict(color=colormap_lines[i]),
            opacity=0.3
        ))
        
        
        fig.add_trace(go.Scatter(
            x=x_plotting[i],
            y=y_plotting[i],
            name=f"Shot: {shots_plotting[i]}",
            mode='markers',
            hovertext=[f"Shot: {shots_plotting[i]}, Time: {t*1000:.3f}ms" for t in times_plotting[i]],
            hoverinfo="x+y+text",
            # marker=dict(
            #     color=col_plotting[i], colorscale="Viridis", showscale=True if i == 0 else False, 
            #     colorbar=dict(title="Gun Flux [mWb]", x=1.2) if i == 0 else None, size=8, 
            #     cmin=global_cmin, cmax=global_cmax
            # ),
            opacity=0.7
        ))        
    fig.show()
    
    
    
    
def plot_delta_elongation_phase(shots, t_lims=[0, 1000], sbc=Sbc(), save=False):
    t_min, t_max = t_lims[0], t_lims[1]
    
    # prepare and unpack crash data
    crash_df = load_crash_data(shots)
    
    # load and set up multishot dataframes
        # set up SBC and load data
    sbc.experiment.set("pi3b")
    df_multishot = sbc.get(asset="reconstruction", shot=shots, column=['li1', 'elong_lfs'], criteria="statistic = mean OR statistic = sigma")

    
    x_plotting = []
    y_plotting = []
    col_plotting = []
    shots_plotting = []
    times_plotting = []
    
    for s in range(len(shots)):
        df = df_multishot[df_multishot['shot'] == shots[s]]
        df = df[df.statistic == 'mean']
        crash_df_ss = crash_df[crash_df.shot == shots[s]]
        lifetime = max(df.t)
        df = df[(df.t >= t_min) & (df.t <= t_max)]
        # get times, li1, elong_lfs and plot
            # load scrambled arrays
        times = np.array(df.t)
        li1 = np.array(df.li1)
        elong_lfs = np.array(df.elong_lfs)
            # sort arrays
        times_ind = np.argsort(times)
        times = np.array([times[i] for i in times_ind])
        li1 = np.array([li1[i] for i in times_ind])
        elong_lfs = np.array([elong_lfs[i] for i in times_ind])
        
        d_times = np.array([(times[i]+times[i+1])/2 for i in range(len(times)-1)])
        d_li1 = np.array([(li1[i]+li1[i+1])/2 for i in range(len(times)-1)])
        d_elong_lfs = np.array([(elong_lfs[i+1]-elong_lfs[i]) for i in range(len(times)-1)])
        d_phases = np.array([time_to_phase.get_phase_from_time_singlepoint_crash(crash_df_ss, i, lifetime) for i in d_times])
        
        if min(len(li1), len(elong_lfs), len(times)) > 0:
            x_plotting.append(d_phases)
            y_plotting.append(d_elong_lfs)
            shots_plotting.append(shots[s])
            times_plotting.append(d_times)



    fig = go.Figure()

    for i in range(len(shots_plotting)):
        fig.add_trace(go.Scatter(
            x=x_plotting[i],
            y=y_plotting[i],
            name=f"Shot: {shots_plotting[i]}",
            mode='lines',
            hovertext=[f"Shot: {shots_plotting[i]}, Time: {t*1000:.3f}ms" for t in times_plotting[i]],
            hoverinfo="x+y+text",
            # line=dict(color=colormap_lines[i]),
            opacity=0.3
        ))
        
        
        fig.add_trace(go.Scatter(
            x=x_plotting[i],
            y=y_plotting[i],
            name=f"Shot: {shots_plotting[i]}",
            mode='markers',
            hovertext=[f"Shot: {shots_plotting[i]}, Time: {t*1000:.3f}ms" for t in times_plotting[i]],
            hoverinfo="x+y+text",
            # marker=dict(
            #     color=col_plotting[i], colorscale="Viridis", showscale=True if i == 0 else False, 
            #     colorbar=dict(title="Gun Flux [mWb]", x=1.2) if i == 0 else None, size=8, 
            #     cmin=global_cmin, cmax=global_cmax
            # ),
            opacity=0.7
        ))
        
    fig.update_layout(
        title=dict(
            text=f"Plot of Change in LCFS Elongation over Phase<br>{len(shots)} Shots",  # Dynamic title
            x=0.5,  # Center title
            y=0.95,
            font=dict(size=28)
        ),
        xaxis_title=r"Time [ms]",
        yaxis_title=f"Change in Elongation of LCFS",  # Replace with actual column name if needed
        legend_title="Shot"
    )
    fig.show()