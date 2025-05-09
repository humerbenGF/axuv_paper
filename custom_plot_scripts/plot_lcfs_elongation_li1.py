# import standard libraries
###################################################################################################
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.stats import norm
from matplotlib import cm
from matplotlib.colors import Normalize


# import sandbox client
###################################################################################################
from gf_sandbox_client.sbc import Sbc

# import personal files
###################################################################################################
from data_loaders.load_crash_data import load_crash_data
from processing_tools.interpolation import linear_interp_extrap as lin_interp


def plot_elongation_li1_at_crashes(shots, high_res_shots, t_lims=[0,1000], crash_indices=[0], max_crash_time=0.01, plotly=False, plot_all_lines=True, name_extension=""):
        # unpack t limits
    t_min, t_max = t_lims[0], t_lims[1]
    
    # load crash data
    crash_df = load_crash_data(list(np.append(shots, [21441])))
    
    # load and set up multishot dataframes
        # set up SBC and load data for normal time resolution data
    sbc=Sbc()
    sbc.experiment.set("pi3b")
    df_multishot = sbc.get(asset="reconstruction", shot=shots, column=['li1', 'elong_lfs'], criteria="statistic = mean OR statistic = sigma")
    df_multishot = df_multishot[(df_multishot.t >= t_min) & (df_multishot.t <= t_max)]
        # set up SBC and load data for high time resolution data
    sbc.project.set("high_time_res_recon_crashes")
    df_multishot_high_res = sbc.get(asset="reconstruction", shot=high_res_shots, column=['li1', 'elong_lfs'], criteria="statistic = mean OR statistic = sigma")
    df_multishot_high_res = df_multishot_high_res[(df_multishot_high_res.t >= t_min) & (df_multishot_high_res.t <= t_max)]

    # save plotting arrays for low time res data
        # set up arrays
    x_plotting = []
    y_plotting = []
    shots_plotting = []
    times_plotting = []
    crash_plotting_x = []
    crash_plotting_y = []
    crash_plotting_shots = []
    crash_plotting_times = []
        # iterate and save data
    for s in range(len(shots)):
        df = df_multishot[df_multishot['shot'] == shots[s]]
        df = df[df.statistic == 'mean']
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
        # gun_flux_arr = np.array([gun_flux[s] for i in times])
        
        if min(len(li1), len(elong_lfs), len(times)) > 0:   # len(gun_flux_arr)
            x_plotting.append(li1)
            y_plotting.append(elong_lfs)
            shots_plotting.append(shots[s])
            times_plotting.append(times)
            # col_plotting.append(gun_flux_arr)
            
            # now find crash times
            df = crash_df[crash_df.shot == shots[s]]
            df = df[(df.t <= t_max) & (df.t >= t_min)]
            
            # unpack crash times
            crash_times = np.array(df.t)
            crash_t_sorting_ind = np.argsort(crash_times)
            crash_times = np.array([crash_times[i] for i in crash_t_sorting_ind])
            
            #
            for crash_time_ind in range(len(crash_times)):
                if crash_time_ind in crash_indices and crash_times[crash_time_ind] < max_crash_time:
                    crash_plotting_x.append(lin_interp(times, li1, crash_times[crash_time_ind]))
                    crash_plotting_y.append(lin_interp(times, elong_lfs, crash_times[crash_time_ind]))
                    crash_plotting_shots.append(shots[s])
                    crash_plotting_times.append(crash_times[crash_time_ind])
    
    # save plotting arrays for high res data
    # save plotting arrays for low time res data
        # set up arrays
    x_plotting_high_res = []
    y_plotting_high_res = []
    shots_plotting_high_res = []
    times_plotting_high_res = []
        # iterate and save data
    if len(high_res_shots) >= 2:
        for s in range(len(high_res_shots)):
            df = df_multishot_high_res[df_multishot_high_res['shot'] == high_res_shots[s]]
            df = df[df.statistic == 'mean']
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
            # gun_flux_arr = np.array([gun_flux[s] for i in times])
            
            if min(len(li1), len(elong_lfs), len(times)) > 0:   # len(gun_flux_arr)
                x_plotting_high_res.append(li1)
                y_plotting_high_res.append(elong_lfs)
                shots_plotting_high_res.append(high_res_shots[s])
                times_plotting_high_res.append(times)
                # col_plotting.append(gun_flux_arr)

    
    if not plotly:
        # set size of figure
        fig = plt.figure(figsize=(5,4), dpi=250)
        
        # plot traces and scatter
        for i in range(len(x_plotting_high_res)):
            plt.plot(x_plotting_high_res[i], y_plotting_high_res[i], alpha=0.4, color='k', zorder=0, label='Example Traces' if i == 0 and not plot_all_lines else None)
        plt.scatter(crash_plotting_x, crash_plotting_y, marker='x', color='r', label='Crashes', zorder=1)
        
        if plot_all_lines:
            for i in range(len(x_plotting)):
                if shots_plotting[i] not in shots_plotting_high_res:
                    plt.plot(x_plotting[i], y_plotting[i], alpha=0.4, color='k', zorder=0, label='Traces' if i == 0 else None)
        # add labels to plot
            # title and axes
        plt.title(r"Plot of $\kappa$ over $\ell_{i1}$ for Chosen Non-Sustain Shots")
        plt.xlabel(r"$\ell_{i1}$")
        plt.ylabel(r"$\kappa$")
            # legend
        plt.legend(fontsize=10)
        plt.xlim(0.4,0.9)
        plt.ylim(1.65,1.95)
            # layout
        plt.tight_layout()
        
        # save figure
        plt.savefig(f"/home/jupyter-humerben/axuv_paper/plot_outputs/elongation_li1/elongation_li1_crashes_{name_extension}.png")
    
    if plotly:
        # Create figure
        fig = go.Figure()

        # Add high-resolution example traces
        for i in range(len(x_plotting_high_res)):
            fig.add_trace(go.Scatter(
                x=x_plotting_high_res[i],
                y=y_plotting_high_res[i],
                mode='lines',
                line=dict(color='black', width=1),
                opacity=0.4,
                name='Example Traces' if i == 0 and not plot_all_lines else None,
                showlegend=(i == 0 and not plot_all_lines)
            ))


        # Add all other traces if enabled
        if plot_all_lines:
            for i in range(len(x_plotting)):
                if shots_plotting[i] not in shots_plotting_high_res:
                    fig.add_trace(go.Scatter(
                        x=x_plotting[i],
                        y=y_plotting[i],
                        mode='lines',
                        line=dict(color='black', width=1),
                        opacity=0.4,
                        name='Traces' if i == 0 else None,
                        showlegend=(i == 0)
                    ))
                    
        # Add crashes
        fig.add_trace(go.Scatter(
            x=crash_plotting_x,
            y=crash_plotting_y,
            mode='markers',
            marker=dict(color='red', symbol='x', size=8),
            name='Crashes',
            hovertext=[f'shot={crash_plotting_shots[i]}, crash_time={str(crash_plotting_times[i]*1000)[:4]}ms' for i in range(len(crash_plotting_x))],
            hoverinfo='x+y+text'
        ))



        # Update layout
        fig.update_layout(
            title="LCFS Elongation Over Normalized Internal Inductance",
            xaxis_title="Normalized Internal Inductance",
            yaxis_title="LCFS Elongation",
            legend=dict(),
        )

        fig.update_xaxes(range=[0.4, 0.9])
        fig.update_yaxes(range=[1.65, 1.95])

        # To display:
        fig.write_html(f"/home/jupyter-humerben/axuv_paper/plot_outputs/elongation_li1/elongation_li1_crashes_{name_extension}.html")
        fig.show()
    
    return



def plot_elongation_li1_triple_matplotlib(shots_1, shots_2, shots_3, t_lims=[0,1000], elong_lims=[1.65,1.95], li1_lims=[0.35,1.0], faded_traces=False, high_res=False):
    # append all shots into shots
    shots = [*shots_1, *shots_2, *shots_3]
    # unpack t limits
    t_min, t_max = t_lims[0], t_lims[1]
    
    # load crash data
    crash_df = load_crash_data(list(np.append(shots, [21441])))
    
    # load and set up multishot dataframes
        # set up SBC and load data for normal time resolution data
    sbc=Sbc()
    sbc.experiment.set("pi3b")
    df_multishot = sbc.get(asset="reconstruction", shots=shots, column=['li1', 'elong_lfs'], criteria="statistic = mean OR statistic = sigma")
    df_multishot = df_multishot[(df_multishot.t >= t_min) & (df_multishot.t <= t_max)]
        # set up SBC and load data for high time resolution data
    sbc.project.set("high_time_res_recon_crashes")
    df_multishot_high_res = sbc.get(asset="reconstruction", shot=shots, column=['li1', 'elong_lfs'], criteria="statistic = mean OR statistic = sigma")
    df_multishot_high_res = df_multishot_high_res[(df_multishot_high_res.t >= t_min) & (df_multishot_high_res.t <= t_max)]

    # save plotting arrays for low time res data
        # set up arrays
    x_plotting = {}
    y_plotting = {}
    shots_plotting = {}
    times_plotting = {}
        # iterate and save data
    for s in range(len(shots)):
        df = df_multishot[df_multishot['shot'] == shots[s]]
        df = df[df.statistic == 'mean']
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
        # gun_flux_arr = np.array([gun_flux[s] for i in times])
        
        if min(len(li1), len(elong_lfs), len(times)) > 0:   # len(gun_flux_arr)
            x_plotting[shots[s]] = li1
            y_plotting[shots[s]] = elong_lfs
            shots_plotting[shots[s]] = shots[s]
            times_plotting[shots[s]] = times
            # col_plotting.append(gun_flux_arr)
        
    
    # save plotting arrays for high res data
    # save plotting arrays for low time res data
        # set up arrays
    x_plotting_high_res = {}
    y_plotting_high_res = {}
    shots_plotting_high_res = {}
    times_plotting_high_res = {}
    crash_plotting_x = []
    crash_plotting_y = []
        # iterate and save data
    if high_res:
        for s in range(len(shots)):
            df = df_multishot_high_res[df_multishot_high_res['shot'] == shots[s]]
            df = df[df.statistic == 'mean']
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
            # gun_flux_arr = np.array([gun_flux[s] for i in times])
            
            if min(len(li1), len(elong_lfs), len(times)) > 0:   # len(gun_flux_arr)
                x_plotting_high_res[shots[s]] = li1
                y_plotting_high_res[shots[s]] = elong_lfs
                shots_plotting_high_res[shots[s]] = shots[s]
                times_plotting_high_res[shots[s]] = times
                # col_plotting.append(gun_flux_arr)
                
                # now find crash times
                df = crash_df[crash_df.shot == shots[s]]
                df = df[(df.t <= t_max) & (df.t >= t_min)]
                
                # unpack crash times
                crash_times = np.array(df.t)
                crash_t_sorting_ind = np.argsort(crash_times)
                crash_times = np.array([crash_times[i] for i in crash_t_sorting_ind])
                
                #
                for crash_time in crash_times:
                    crash_plotting_x.append(lin_interp(times, li1, crash_time))
                    crash_plotting_y.append(lin_interp(times, elong_lfs, crash_time))


    # generate colormaps for timing
        # set proxy array
    a=times_plotting[list(times_plotting.keys())[0]]
        # use proxy array to generate the colormap
    norm = Normalize(vmin=np.min(a), vmax=np.max(a))
    cmap = cm.get_cmap('plasma')
    colors = cmap(norm(a))
    
    # set size of figure
    fig, axs = plt.subplots(1, 3, figsize=(10,4), dpi=250)
    axs = np.array([axs])  # Convert 1D to 2D array for consistent indexing
    
    # plot traces and scatter
    shots_2D = [shots_1, shots_2, shots_3]
    group_descriptions = [r"a) $V_{form}=22kV$"+"\n"+r"$N_{caps}<=32$"+"\n"+r"$\psi_{gun}<100mWb$",
                          r"b) $V_{form}=22kV$"+"\n"+r"$N_{caps}=48$"+"\n"+r"$\psi_{gun}<100mWb$",
                          r"b) $V_{form}>=23kV$"+"\n"+r"$N_{caps}=48$"+"\n"+r"$\psi_{gun}>130mWb$"]
    fontsize=14
    
    for i in range(len(shots_2D)):
        shots_singlepanel = shots_2D[i]
        for shot in shots_singlepanel:
            if shot in x_plotting.keys():
                if high_res:
                    axs[0, i].plot(x_plotting_high_res[shot], y_plotting_high_res[shot], alpha=0.4, color='k', zorder=2)
                else:
                    axs[0, i].plot(x_plotting[shot], y_plotting[shot], alpha=0.4, color='k', zorder=2)
                axs[0, i].scatter(x_plotting[shot], y_plotting[shot], c=colors, zorder=3, marker='.')
                
        if faded_traces:
            for shot in shots:
                if shot in x_plotting.keys():
                    if high_res:
                        axs[0, i].plot(x_plotting_high_res[shot], y_plotting_high_res[shot], alpha=0.05, color='k', zorder=0)
                    else:
                        axs[0, i].plot(x_plotting[shot], y_plotting[shot], alpha=0.05, color='k', zorder=0)
                    axs[0, i].scatter(x_plotting[shot], y_plotting[shot], c=colors, alpha=0.2, zorder=1, marker='.')
                    

        # settings for all individual panels
            # limits
        axs[0, i].set_ylim(elong_lims[0], elong_lims[1])
        axs[0, i].set_xlim(li1_lims[0], li1_lims[1])
            # labels
        axs[0, i].text((li1_lims[1]-li1_lims[0])*0.025+li1_lims[0], (elong_lims[1]-elong_lims[0])*0.025+elong_lims[0], group_descriptions[i])
            # axis titles and tick labels
        axs[0, i].set_xlabel(r"$\ell_{i1}$", fontsize=fontsize)
        if i != 0:
            axs[0, i].set_yticklabels([])  # Hide tick labels but keep the ticks
            # grid
        axs[0, i].grid(True)
    
    # settings for the overall plot 
        # titles
    plt.suptitle(r"LCFS Elongation $\mathcal{K}$ over Internal Inductance $\ell_{i1}$", fontsize=fontsize+2)
    axs[0, 0].set_ylabel(r"$\mathcal{K}$", fontsize=fontsize)
        # legend
    for i in range(len(times_plotting[list(times_plotting.keys())[0]])):
        axs[0, 1].scatter([],[], color=colors[i], label=f"t={int(times_plotting[list(times_plotting.keys())[0]][i]*1000)}ms")
        
        
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    axs[0, 1].legend(loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.55), frameon=False, fontsize=10)
    
    
    # for i in range(len(x_plotting)):
    #     plt.plot(x_plotting_high_res[i], y_plotting_high_res[i], alpha=0.4, color='k', zorder=0)
    #     plt.scatter(x_plotting[i], y_plotting[i], c=colors, zorder=1)
        
    # plt.scatter(crash_plotting_x, crash_plotting_y, marker='x', color='r')
    # # add labels to plot

        # layout

    # save figure
    plt.savefig("/home/jupyter-humerben/axuv_paper/plot_outputs/elongation_li1/elongation_li1_triple_plot.png")
    
    return



def plot_elongation_li1_matplotlib(shots, t_lims=[0,1000]):
    # unpack t limits
    t_min, t_max = t_lims[0], t_lims[1]
    
    # load crash data
    crash_df = load_crash_data(list(np.append(shots, [21441])))
    
    # load and set up multishot dataframes
        # set up SBC and load data for normal time resolution data
    sbc=Sbc()
    sbc.experiment.set("pi3b")
    df_multishot = sbc.get(asset="reconstruction", shot=shots, column=['li1', 'elong_lfs'], criteria="statistic = mean OR statistic = sigma")
    df_multishot = df_multishot[(df_multishot.t >= t_min) & (df_multishot.t <= t_max)]
        # set up SBC and load data for high time resolution data
    sbc.project.set("high_time_res_recon_crashes")
    df_multishot_high_res = sbc.get(asset="reconstruction", shot=shots, column=['li1', 'elong_lfs'], criteria="statistic = mean OR statistic = sigma")
    df_multishot_high_res = df_multishot_high_res[(df_multishot_high_res.t >= t_min) & (df_multishot_high_res.t <= t_max)]

    # save plotting arrays for low time res data
        # set up arrays
    x_plotting = []
    y_plotting = []
    shots_plotting = []
    times_plotting = []
        # iterate and save data
    for s in range(len(shots)):
        df = df_multishot[df_multishot['shot'] == shots[s]]
        df = df[df.statistic == 'mean']
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
        # gun_flux_arr = np.array([gun_flux[s] for i in times])
        
        if min(len(li1), len(elong_lfs), len(times)) > 0:   # len(gun_flux_arr)
            x_plotting.append(li1)
            y_plotting.append(elong_lfs)
            shots_plotting.append(shots[s])
            times_plotting.append(times)
            # col_plotting.append(gun_flux_arr)
    
    # save plotting arrays for high res data
    # save plotting arrays for low time res data
        # set up arrays
    x_plotting_high_res = []
    y_plotting_high_res = []
    shots_plotting_high_res = []
    times_plotting_high_res = []
    crash_plotting_x = []
    crash_plotting_y = []
        # iterate and save data
    for s in range(len(shots)):
        df = df_multishot_high_res[df_multishot_high_res['shot'] == shots[s]]
        df = df[df.statistic == 'mean']
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
        # gun_flux_arr = np.array([gun_flux[s] for i in times])
        
        if min(len(li1), len(elong_lfs), len(times)) > 0:   # len(gun_flux_arr)
            x_plotting_high_res.append(li1)
            y_plotting_high_res.append(elong_lfs)
            shots_plotting_high_res.append(shots[s])
            times_plotting_high_res.append(times)
            # col_plotting.append(gun_flux_arr)
            
            # now find crash times
            df = crash_df[crash_df.shot == shots[s]]
            df = df[(df.t <= t_max) & (df.t >= t_min)]
            
            # unpack crash times
            crash_times = np.array(df.t)
            crash_t_sorting_ind = np.argsort(crash_times)
            crash_times = np.array([crash_times[i] for i in crash_t_sorting_ind])
            
            #
            for crash_time in crash_times:
                crash_plotting_x.append(lin_interp(times, li1, crash_time))
                crash_plotting_y.append(lin_interp(times, elong_lfs, crash_time))

        
        
        
    
    
    # generate colormaps for timing
        # set proxy array
    a=times_plotting[0]
        # use proxy array to generate the colormap
    norm = Normalize(vmin=np.min(a), vmax=np.max(a))
    cmap = cm.get_cmap('viridis')
    colors = cmap(norm(a))
    
    # set size of figure
    fig = plt.figure(figsize=(5,4), dpi=250)
    
    # plot traces and scatter
    for i in range(len(x_plotting)):
        plt.plot(x_plotting_high_res[i], y_plotting_high_res[i], alpha=0.4, color='k', zorder=0)
        plt.scatter(x_plotting[i], y_plotting[i], c=colors, zorder=1)
        
    plt.scatter(crash_plotting_x, crash_plotting_y, marker='x', color='r')
    # add labels to plot
        # title and axes
    plt.title(r"Plot of $\kappa$ over $\ell_{i1}$ for Chosen Non-Sustain Shots")
    plt.xlabel(r"$\ell_{i1}$")
    plt.ylabel(r"$\kappa$")
        # legend
    for i in range(len(times_plotting[0])):
        plt.scatter([],[], color=colors[i], label=f"t={int(times_plotting[0][i]*1000)}ms")
    plt.legend(title="time [ms]", ncol=2, fontsize=10, columnspacing=0.5, handletextpad=0.3, handlelength=1.5)
        # layout
    plt.tight_layout()
    
    # save figure
    plt.savefig("/home/jupyter-humerben/axuv_paper/plot_outputs/elongation_li1/elongation_li1.png")
    
    return
    



def plot_elongation_li1(shots, gun_flux, t_lims=[0, 1000], sbc=Sbc(), save=False):
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
        gun_flux_arr = np.array([gun_flux[s] for i in times])
        
        if min(len(gun_flux_arr), len(li1), len(elong_lfs), len(times)) > 0:
            x_plotting.append(li1)
            y_plotting.append(elong_lfs)
            shots_plotting.append(shots[s])
            times_plotting.append(times)
            col_plotting.append(gun_flux_arr)
    
    fig = go.Figure()
    
    # get colorbar limits
    global_cmin = np.min([np.min(arr) for arr in col_plotting])
    global_cmax = np.max([np.max(arr) for arr in col_plotting])
    colormap_lines = px.colors.sample_colorscale("Viridis", [((col_plotting[i][0] - global_cmin) / (global_cmax - global_cmin)) for i in range(len(col_plotting))])
    
    for i in range(len(shots_plotting)):
        fig.add_trace(go.Scatter(
            x=x_plotting[i],
            y=y_plotting[i],
            name=f"Shot: {shots_plotting[i]}",
            mode='lines',
            hovertext=[f"Shot: {shots_plotting[i]}, Time: {t*1000:.3f}ms" for t in times_plotting[i]],
            hoverinfo="x+y+text",
            line=dict(
                color=colormap_lines[i]
            ),
            opacity=0.3
        ))
        
        
        fig.add_trace(go.Scatter(
            x=x_plotting[i],
            y=y_plotting[i],
            name=f"Shot: {shots_plotting[i]}",
            mode='markers',
            hovertext=[f"Shot: {shots_plotting[i]}, Time: {t*1000:.3f}ms" for t in times_plotting[i]],
            hoverinfo="x+y+text",
            marker=dict(
                color=col_plotting[i],  # <-- This should be an array the same length as x/y
                colorscale="Viridis",   # Or any Plotly-supported colorscale
                showscale=True if i == 0 else False,  # Show colorbar only for the first trace
                colorbar=dict(title="Gun Flux [mWb]", x=1.2) if i == 0 else None,
                size=8,
                cmin=global_cmin,                # Global min
                cmax=global_cmax                 # Global max
            ),
            opacity=0.7
        ))
        


        
    # Customize layout
    fig.update_layout(
        title=dict(
            text=f"Plot of Elongation LFS Over Internal Inductance",  # Dynamic title
            x=0.5,  # Center title
            y=0.95,
            font=dict(size=22)
        ),
        xaxis_title="Internal Inductance",
        yaxis_title="Elongation LFS",  # Replace with actual column name if needed
        legend_title="Shots"
    )
    
    if save:
        fig.write_html("/home/jupyter-humerben/axuv_paper/plot_outputs/elongation_non_sustain.html")
    fig.show()
    
    return




    
    
def crash_between_times(times, ss_crash_df, t_ind, c_ind):
    
    if (between(times[t_ind - 1], ss_crash_df.pre_crash_t.iloc[c_ind], times[t_ind]) or 
            between(times[t_ind - 1], ss_crash_df.t.iloc[c_ind], times[t_ind]) or
            between(times[t_ind - 1], ss_crash_df.post_crash_t.iloc[c_ind], times[t_ind]) or
            between(times[t_ind], ss_crash_df.pre_crash_t.iloc[c_ind], times[t_ind+1]) or 
            between(times[t_ind], ss_crash_df.t.iloc[c_ind], times[t_ind+1]) or
            between(times[t_ind], ss_crash_df.post_crash_t.iloc[c_ind], times[t_ind+1])):
        return True
    else:
        return False
    
def between(a, x, b):
    if a < x and x < b:
        return True
    else:
        return False