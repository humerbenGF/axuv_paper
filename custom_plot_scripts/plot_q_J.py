# import standard libraries
###################################################################################################
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.container import ErrorbarContainer



# import sandbox client
###################################################################################################
from gf_sandbox_client.sbc import Sbc

# import personal files
###################################################################################################
from data_loaders.load_crash_data import load_crash_data

def plot_q_J_side_by_side_singleshot(shot, t_lims=[0, 1000], r_lims=[-0.025, 0.475], q_lims=[0, 4], J_lims=[0, 1], sbc=Sbc(), plot_crashes=False, invert_multi_plot=False, title=False):
    # unpack limits
    t_min, t_max = t_lims
    r_min, r_max = r_lims
    q_min, q_max = q_lims
    J_min, J_max = J_lims
    
    
    # load and set up multishot dataframes
        # set up SBC and load data
    sbc.experiment.set("pi3b")
    df_multishot = sbc.get(asset="reconstruction_profile", shot=shot, column=['q', 'Javg', "Rmin", "Rmax"], criteria="statistic = mean OR statistic = sigma")
        # trim times
    df_multishot = df_multishot[df_multishot['t'] >= t_min]
    df_multishot = df_multishot[df_multishot['t'] <= t_max]
        # separate into mean and errors
    df_multishot_err = df_multishot[df_multishot['statistic'] == 'sigma']
    df_multishot = df_multishot[df_multishot['statistic'] == 'mean']

    # iterate through shots
        # unpack single shot information from the multishot dataframes
    df = df_multishot
    df_err = df_multishot_err
        
    # find unique times
    t_vals = np.unique(df['t'])
    plotting_array_q = []
    plotting_error_q = []
    plotting_array_J = []
    plotting_error_J = []
    plotting_array_r = []
        
    for t in t_vals:
        # get values for this specific time
            # psibar
        psibars = list(df[df["t"] == t]['psibar'])
        psibar_errors = list(df_err[df_err["t"] == t]['psibar'])
            # q
        q_vals = list(df[df['t'] == t]['q'])
        q_error_vals = list(df_err[df_err['t'] == t]['q'])
            # J
        J_vals = list(df[df['t'] == t]['Javg'])
        J_error_vals = list(df_err[df_err['t'] == t]['Javg'])
            # r
        R_min_vals = list(df[df['t'] == t]['Rmin'])
        R_max_vals = list(df[df['t'] == t]['Rmax'])
        
        # sort values for the specific time
            # get indices for sorting
        sorted_psi_indices = np.argsort(psibars)
        sorted_psi_indices_errors = np.argsort(psibar_errors)
            # sort q
        q_vals_plotting = [q_vals[i] for i in sorted_psi_indices]
        q_errors_plotting = [q_error_vals[i] for i in sorted_psi_indices_errors]
            # sort J
        J_vals_plotting = [J_vals[i]/(10**6) for i in sorted_psi_indices]
        J_errors_plotting = [J_error_vals[i]/(10**6) for i in sorted_psi_indices_errors]
            # sort r
        r_plotting = [(R_max_vals[i] - R_min_vals[i])/2 for i in sorted_psi_indices]
        
        # append values to plotting arrays
            # q
        plotting_array_q.append(q_vals_plotting)
        plotting_error_q.append(q_errors_plotting)
            # J
        plotting_array_J.append(J_vals_plotting)
        plotting_error_J.append(J_errors_plotting)
            # r
        plotting_array_r.append(r_plotting)

        
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2), dpi=400)  # Adjust size based on n
    font_min = 12

    axs = np.array([axs])  # Convert 1D to 2D array for consistent indexing

    # make colormap
    cmap = colormaps["plasma"]
    colormap = [cmap(i) for i in np.linspace(0, 1, len(plotting_array_r))]
    
    # J plot
    for t in range(len(plotting_array_r)):
        # plot
        axs[0, 0].errorbar(x=plotting_array_r[t], y=plotting_array_J[t], yerr=plotting_error_J[t],
                            label=f"t={int(round(t_vals[t]*1000, 2))}ms", color=colormap[t], alpha=0.5, linestyle="-", marker="o")
        # handle axes
        axs[0, 0].set_xlim(r_min, r_max)
        axs[0, 0].set_ylim(J_min, J_max)
        axs[0, 0].set_ylabel(r"$J_{\phi}\,\mathrm{[MA/m^2]}$", fontsize=font_min+2)
        # add label
        
    # q plot
    for t in range(len(plotting_array_r)):
        axs[0, 1].errorbar(x=plotting_array_r[t], y=plotting_array_q[t], yerr=plotting_error_q[t],
                            label=f"t={int(round(t_vals[t]*1000, 2))}ms", color=colormap[t], alpha=0.5, linestyle="-", marker="o")
        # handle axes
        axs[0, 1].set_xlim(r_min, r_max)
        axs[0, 1].set_ylim(q_min, q_max)
        axs[0, 1].set_ylabel(r"$q$", fontsize=font_min+2)
        axs[0,1].yaxis.grid(True, zorder=3)
                    
    # general settings to add at the end
        # x label settings
    axs[0, 0].set_xlabel(r"$r\,\mathrm{[m]}$", fontsize=font_min+2)
    axs[0, 1].set_xlabel(r"$r\,\mathrm{[m]}$", fontsize=font_min+2)
        # tick params
    axs[0, 0].tick_params(axis='both', labelsize=font_min)
    axs[0, 1].tick_params(axis='both', labelsize=font_min)

    # get legend
    h, l = axs[0, 0].get_legend_handles_labels()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.26, top=0.9)
    fig.legend(h, l, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0), frameon=False, fontsize=font_min)
    
    fig.suptitle(f"Shot {shot} "+r"$J_\phi(r)$ and $q(r)$ Profiles" if title else None, fontsize=font_min+6)
    fig.savefig(f"/home/jupyter-humerben/axuv_paper/plot_outputs/J_q_r/J_q_r_{shot}.png", bbox_inches='tight')
    plt.close()

    return



def plot_q_J_side_by_side(shots, t_lims=[0, 1000], r_lims=[-0.025, 0.475], q_lims=[0, 4], J_lims=[0, 10**6], sbc=Sbc(), plot_crashes=False, invert_multi_plot=False):
    # unpack limits
    t_min, t_max = t_lims
    r_min, r_max = r_lims
    q_min, q_max = q_lims
    J_min, J_max = J_lims
    
    # prepare and unpack crash data
    crash_df = load_crash_data(shots)
    
    # load and set up multishot dataframes
        # set up SBC and load data
    sbc.experiment.set("pi3b")
    df_multishot = sbc.get(asset="reconstruction_profile", shot=shots, column=['q', 'Javg', "Rmin", "Rmax"], criteria="statistic = mean OR statistic = sigma")
        # trim times
    df_multishot = df_multishot[df_multishot['t'] >= t_min]
    df_multishot = df_multishot[df_multishot['t'] <= t_max]
        # separate into mean and errors
    df_multishot_err = df_multishot[df_multishot['statistic'] == 'sigma']
    df_multishot = df_multishot[df_multishot['statistic'] == 'mean']

    # set up dicts to store plotting data
    plotting_arrays_q = {}
    plotting_errors_q = {}
    plotting_arrays_J = {}
    plotting_errors_J = {}
    plotting_arrays_r = {}

    # iterate through shots
    for s in range(len(shots)):
        # unpack single shot information from the multishot dataframes
        df = df_multishot[df_multishot.shot == shots[s]]
        df_err = df_multishot_err[df_multishot_err.shot == shots[s]]
        
        # find unique times
        t_vals = np.unique(df['t'])
        plotting_arrays_q[shots[s]] = []
        plotting_errors_q[shots[s]] = []
        plotting_arrays_J[shots[s]] = []
        plotting_errors_J[shots[s]] = []
        plotting_arrays_r[shots[s]] = []
        
        for t in t_vals:
            # get values for this specific time
                # psibar
            psibars = list(df[df["t"] == t]['psibar'])
            psibar_errors = list(df_err[df_err["t"] == t]['psibar'])
                # q
            q_vals = list(df[df['t'] == t]['q'])
            q_error_vals = list(df_err[df_err['t'] == t]['q'])
                # J
            J_vals = list(df[df['t'] == t]['Javg'])
            J_error_vals = list(df_err[df_err['t'] == t]['Javg'])
                # r
            R_min_vals = list(df[df['t'] == t]['Rmin'])
            R_max_vals = list(df[df['t'] == t]['Rmax'])
            
            # sort values for the specific time
                # get indices for sorting
            sorted_psi_indices = np.argsort(psibars)
            sorted_psi_indices_errors = np.argsort(psibar_errors)
                # sort q
            q_vals_plotting = [q_vals[i] for i in sorted_psi_indices]
            q_errors_plotting = [q_error_vals[i] for i in sorted_psi_indices_errors]
                # sort J
            J_vals_plotting = [J_vals[i] for i in sorted_psi_indices]
            J_errors_plotting = [J_error_vals[i] for i in sorted_psi_indices_errors]
                # sort r
            r_plotting = [(R_max_vals[i] - R_min_vals[i])/2 for i in sorted_psi_indices]
            
            # append values to plotting arrays
                # q
            plotting_arrays_q[shots[s]].append(q_vals_plotting)
            plotting_errors_q[shots[s]].append(q_errors_plotting)
                # J
            plotting_arrays_J[shots[s]].append(J_vals_plotting)
            plotting_errors_J[shots[s]].append(J_errors_plotting)
                # r
            plotting_arrays_r[shots[s]].append(r_plotting)

        
    if plot_crashes:
        print("plotting crashes")
        
    else:
        
        
        
        
        fig, axs = plt.subplots(len(shots), 2, figsize=(10, 3 * len(shots)), dpi=400)  # Adjust size based on n

        # Ensure axs is always a 2D array
        if len(shots) == 1:
            axs = np.array([axs])  # Convert 1D to 2D array for consistent indexing
        
        for s in range(len(shots)):
            
            # make colormap
            cmap = colormaps["viridis"]
            colormap = [cmap(i) for i in np.linspace(0, 1, len(plotting_arrays_r[shots[s]]))]
            
            # J plot
            for t in range(len(plotting_arrays_r[shots[s]])):
                # plot
                axs[s, 0].errorbar(x=plotting_arrays_r[shots[s]][t], y=plotting_arrays_J[shots[s]][t], yerr=plotting_errors_J[shots[s]][t],
                                   label=f"t={int(round(t_vals[t]*1000, 2))}", color=colormap[t], alpha=0.5, linestyle="-", marker="o")
                # handle axes
                axs[s, 0].set_xlim(r_min, r_max)
                axs[s, 0].set_ylim(J_min, J_max)
                axs[s, 0].set_ylabel(r"$J_{\phi}$ $[A/m^2]$")
                # add label
                axs[s, 0].text(0.0, (J_max-J_min)*0.025, f"{s+1}.a) Shot {shots[s]}: "+r"$J_{\phi}(r)$")
                
            # q plot
            for t in range(len(plotting_arrays_r[shots[s]])):
                axs[s, 1].errorbar(x=plotting_arrays_r[shots[s]][t], y=plotting_arrays_q[shots[s]][t], yerr=plotting_errors_q[shots[s]][t],
                                   label=f"t={int(round(t_vals[t]*1000, 2))}", color=colormap[t], alpha=0.5, linestyle="-", marker="o")
                # handle axes
                axs[s, 1].set_xlim(r_min, r_max)
                axs[s, 1].set_ylim(q_min, q_max)
                axs[s, 1].set_ylabel(r"$q(r)$")
                
                # add label
                axs[s, 1].text(0.0, (q_max-q_min)*0.025, f"{s+1}.b) Shot {shots[s]}: "+r"$q(r)$")
                
        # general settings to add at the end
            # x label settings
        axs[s, 0].set_xlabel("r [m]")
        axs[s, 1].set_xlabel("r [m]")
        axs[0, 1].legend(title="time [ms]", ncol=4, fontsize=10, columnspacing=0.5, handletextpad=0.3, handlelength=1.5)
                
        shot_string = ""
        for i in shots:
            shot_string += ("_"+str(i))
            
        fig.tight_layout()            
        fig.savefig(f"/home/jupyter-humerben/axuv_paper/plot_outputs/J_q_r/J_q_r{shot_string}.png")



    '''
    # Add traces for each set of data
    plotted_crash_end_time_ind= -1
    for i in range(len(t_vals)):
        # if the last set of lines bracketed a crash, you don't need to fill this set in
        if i == plotted_crash_end_time_ind:
            continue
        
        # check if the current and next times bracket a crash
        if i < len(t_vals) - 1:
            is_crash, crash_ind, crash_start_time_ind, crash_end_time_ind = get_crash_index_bracket(crash_df, t_vals, i)
            if is_crash and crash_start_time_ind <= t_max and crash_end_time_ind >= t_min:
                plotted_crash_end_time_ind = crash_end_time_ind
                # add trace for before crash occurs
                fig.add_trace(go.Scatter(
                    x=plotting_arrays_x[crash_start_time_ind], 
                    y=plotting_arrays_y[crash_start_time_ind],
                    error_y=dict(
                        type="data",  # Absolute error values
                        array=plotting_errors_y[crash_start_time_ind],  # Upper error
                        symmetric=True,  # Symmetric error bars
                    ),
                    mode='lines',
                    name=f"t={str(t_vals[crash_start_time_ind]*1000)[:4]}ms, BEFORE a crash at {str(crash_df.t[crash_ind]*1000)[:4]}ms",
                    line=dict(color=colormap_crashes[crash_ind], width=2, dash='dash'),
                    opacity=0.8,  # Similar to alpha in Matplotlib
                    hovertext=f"t={str(t_vals[crash_start_time_ind]*1000)[:4]}ms, BEFORE a crash at {str(crash_df.t[crash_ind]*1000)[:4]}ms",
                    hoverinfo='text'
                ))
                # add trace for after a crash occurs
                fig.add_trace(go.Scatter(
                    x=plotting_arrays_x[crash_end_time_ind], 
                    y=plotting_arrays_y[crash_end_time_ind],
                    error_y=dict(
                        type="data",  # Absolute error values
                        array=plotting_errors_y[crash_end_time_ind],  # Upper error
                        symmetric=True,  # Symmetric error bars
                    ),
                    mode='lines',
                    name=f"t={str(t_vals[crash_end_time_ind]*1000)[:4]}ms, AFTER a crash at {str(crash_df.t[crash_ind]*1000)[:4]}ms",
                    line=dict(color=colormap_crashes[crash_ind], width=2, dash='solid'),
                    opacity=0.8,  # Similar to alpha in Matplotlib
                    hovertext=f"t={str(t_vals[crash_end_time_ind]*1000)[:4]}ms, AFTER a crash at {str(crash_df.t[crash_ind]*1000)[:4]}ms",
                    hoverinfo='text'
                ))
                continue
                
                
        fig.add_trace(go.Scatter(
            x=plotting_arrays_x[i], 
            y=plotting_arrays_y[i],
            mode='lines',
            name=f"t={str(t_vals[i])[:6]}",
            line=dict(color=colormap[i], width=2),
            opacity=0.2,  # Similar to alpha in Matplotlib
            hovertext=f"t={str(t_vals[i])[:6]}",  # Add custom hover text
            hoverinfo='text',
            showlegend=False
        ))
    
    
    t_vals_ms = np.zeros(len(t_vals))
    for i in range(len(t_vals)):
        t_vals_ms[i] = t_vals[i]*1000
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers', hoverinfo="none", showlegend=False,
        marker=dict(
            colorscale=colormap, cmin=min(t_vals_ms), cmax=max(t_vals_ms), color=t_vals_ms, showscale=True,
            colorbar=dict(title="Time [ms]", x=1.4)
        )
    ))
    
    # Customize layout
    fig.update_layout(
        title=dict(
            text=f"Plot of {column} over r<br>Shot {shot}",  # Dynamic title
            x=0.5,  # Center title
            y=0.95,
            font=dict(size=22)
        ),
        xaxis_title=r"r",
        yaxis_title=column,  # Replace with actual column name if needed
        showlegend=True
    )
    
    if jupyter:
        fig.update_layout(
            width=900,  # Set figure width
            height=600,  # Set figure height
        )
        
    
    # Show the figure
    fig.show()
    '''


    return



if __name__ == '__main__':
    plot_q_J_side_by_side_singleshot(22289)
    # plot_q_J_side_by_side([22287, 22289])