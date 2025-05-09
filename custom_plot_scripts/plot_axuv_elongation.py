# import standard libraries
###################################################################################################
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from matplotlib import cm
from matplotlib.colors import Normalize


# import sandbox client
###################################################################################################
from gf_sandbox_client.sbc import Sbc


def plot_axuv_and_elongation_dual_plot(shot, sensor_number=21, psibar=0, t_lims=[0, 1000], elong_lims=[1.6,2], li1_lims=[0.2,1], error=False, q_reduction_factor=4):
    t_min, t_max = t_lims
    
    sbc=Sbc()
    sbc.experiment.set("pi3b")
    # get axuv data
    axuv_df = Sbc().get(experiment='pi3b', asset="axuv_amps", shot=shot, criteria=f"sensor_number = {sensor_number}")
    axuv_df = axuv_df[(axuv_df.t >= t_min) & (axuv_df.t <=t_max)]
    axuv_t = np.array(axuv_df.t)
    axuv_A = np.array(axuv_df.A)
    axuv_t_sorting_indices = np.argsort(axuv_t)
    axuv_t = np.array([axuv_t[i]*1000 for i in axuv_t_sorting_indices])
    axuv_A = np.array([axuv_A[i] for i in axuv_t_sorting_indices])
    
    # load the data for both plots
    df = sbc.get(asset="reconstruction", shot=shot, column=['li1', 'elong_lfs'], criteria="statistic = mean OR statistic = sigma")
    df_high_res = sbc.get(asset="reconstruction", shot=shot, column=['li1', 'elong_lfs'], criteria="statistic = mean OR statistic = sigma", project='high_time_res_recon_crashes')
    df_special = sbc.get(asset="reconstruction_profile", shot=shot, column=['q'], criteria="statistic = mean OR statistic = sigma")
    df_special_high_res = sbc.get(asset="reconstruction_profile", shot=shot, column=['q'], criteria="statistic = mean OR statistic = sigma", project="high_time_res_recon_crashes")
    # trim times
        # all
    df = df[(df['t'] >= t_min) & (df['t'] <= t_max)]
    df_high_res = df_high_res[(df_high_res['t'] >= t_min) & (df_high_res['t'] <= t_max)]
        # special
    df_special = df_special[(df_special['t'] >= t_min) & (df_special['t'] <= t_max)]
    df_special_high_res = df_special_high_res[(df_special_high_res['t'] >= t_min) & (df_special_high_res['t'] <= t_max)]
    # seperate into mean and errors
        # all
    # df_err = df[df['statistic'] == 'sigma']
    df = df[df['statistic'] == 'mean']
    df_high_res = df_high_res[df_high_res['statistic'] == 'mean']
        # special
    # df_special_err = df_special[df_special['statistic'] == 'sigma']
    df_special = df_special[df_special['statistic'] == 'mean']
    df_special_high_res = df_special_high_res[df_special_high_res['statistic'] == 'mean']
    
    
    data_columns = ['li1', 'elong_lfs', 'q', 't']
    
    low_resolution_plotting_data = {}
    high_resolution_plotting_data = {}
    
    for col in data_columns:
        if col == 'q':
            df_singlecolumn = df_special[df_special.psibar == psibar]
            df_singlecolumn_high_res = df_special_high_res[df_special_high_res.psibar == psibar]
        else:
            df_singlecolumn = df
            df_singlecolumn_high_res = df_high_res
            
        # unpack the column and times
            # column
        column_values = np.array(df_singlecolumn[col])
        column_values_high_res = np.array(df_singlecolumn_high_res[col])
            # times
        times = np.array(df_singlecolumn.t)
        times_high_res = np.array(df_singlecolumn_high_res.t)
        
        # sort the column values
            # get sorting indices
        times_indices = np.argsort(times)
        times_high_res_indices = np.argsort(times_high_res)
            # sort
        column_values = np.array([column_values[i] for i in times_indices])
        column_values_high_res = np.array([column_values_high_res[i] for i in times_high_res_indices])
        
        low_resolution_plotting_data[col] = column_values
        high_resolution_plotting_data[col] = column_values_high_res
    
    low_resolution_plotting_data['t'] = [low_resolution_plotting_data['t'][i]*1000 for i in range(len(low_resolution_plotting_data['t']))]
    high_resolution_plotting_data['t'] = [high_resolution_plotting_data['t'][i]*1000 for i in range(len(high_resolution_plotting_data['t']))]
    
    low_resolution_plotting_data['q'] = [low_resolution_plotting_data['q'][i]/q_reduction_factor for i in range(len(low_resolution_plotting_data['q']))]
    high_resolution_plotting_data['q'] = [high_resolution_plotting_data['q'][i]/q_reduction_factor for i in range(len(high_resolution_plotting_data['q']))]

    
    # generate colormaps for timing
        # set proxy array
    a=low_resolution_plotting_data['t']
        # use proxy array to generate the colormap
    norm = Normalize(vmin=np.min(a), vmax=np.max(a))
    cmap = cm.get_cmap('plasma')
    colors_array = cmap(norm(a))
    traces_color='k'
    
    # Create 1 row, 2 column subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot with 2 y-axes
        # AXUV
    ax1.plot(axuv_t, axuv_A, alpha=0.5, zorder=0, label="AXUV Current")
        # twin axes
    ax1b = ax1.twinx()
        # li1
    ax1b.plot(high_resolution_plotting_data['t'], high_resolution_plotting_data['li1'], color=traces_color, alpha=0.7, zorder=1)
    ax1b.scatter(low_resolution_plotting_data['t'], low_resolution_plotting_data['li1'], c=colors_array, marker='s', zorder=2, label=r'$\ell_{i1}$')
        # elong lfs
    ax1b.plot(high_resolution_plotting_data['t'], high_resolution_plotting_data['elong_lfs'], color=traces_color, alpha=0.7, zorder=1)
    ax1b.scatter(low_resolution_plotting_data['t'], low_resolution_plotting_data['elong_lfs'], c=colors_array, marker='X', zorder=2, label=r'$\kappa$')
        # q
    ax1b.plot(high_resolution_plotting_data['t'], high_resolution_plotting_data['q'], color=traces_color, alpha=0.7, zorder=1)
    ax1b.scatter(low_resolution_plotting_data['t'], low_resolution_plotting_data['q'], c=colors_array, marker='o', zorder=2, label=r'$q_{95}/4$')

    # titles and legend for the left plot
        # axis labels
    ax1.set_ylabel(r'$I_{photodiode}$ [A]')
    ax1b.set_ylabel(r"$\ell_{i1}$, $\kappa$, $q_{95}/4$")
    ax1.set_xlabel('time [ms]')
        # title
    ax1.set_title(r"AXUV, $\ell_{i1}$, $\kappa$, $q_{95}$ over Time")
        # Get the line objects and labels from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
        # Combine them and create a single legend on ax1
    ax1.legend(lines1 + lines2, labels1 + labels2)

    # Right plot (single y-axis)
        # plot values
    ax2.plot(high_resolution_plotting_data['li1'], high_resolution_plotting_data['elong_lfs'], alpha=0.7, color=traces_color, zorder=0)
    ax2.scatter(low_resolution_plotting_data['li1'], low_resolution_plotting_data['elong_lfs'], color=colors_array, zorder=1)
    ax2.axvline(0.4, color='k', linestyle='--')
    ax2.axhline(1.935, color='k', linestyle='--')
    
    # titles and legend for the left plot
        # axis labels and limits
    ax2.set_ylabel(r'$\kappa$', fontsize=18)
    ax2.set_ylim(elong_lims[0], elong_lims[1])
    ax2.set_xlabel(r'$\ell_{i1}$', fontsize=18)
    ax2.set_xlim(li1_lims[0], li1_lims[1])
        # title
    ax2.set_title(r'$\kappa$ over $\ell_{i1}$')
        # legend
    for i in range(len(colors_array)):
        ax2.scatter([],[],color=colors_array[i], label=f"t={i+1}")
    ax2.legend(title="time [ms]", ncol=2, fontsize=10)

    plt.tight_layout()
    
    plt.savefig(f"/home/jupyter-humerben/axuv_paper/plot_outputs/axuv_and_li1_elongation/axuv_li1_elongation_q0_{shot}.png")
    
    return