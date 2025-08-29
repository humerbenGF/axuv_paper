# import standard libraries
###################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl
import pandas as pd


# import sandbox client
###################################################################################################
from gf_sandbox_client.sbc import Sbc


# import personal files
###################################################################################################
from data_loaders.load_crash_data import load_crash_data




def plot_q0_q95_t(shot, t_lims=[0,1000]):
    # unpack t limits
    t_min, t_max = t_lims[0], t_lims[1]
    
    # load crash data
    
    # load and set up multishot dataframes
        # set up SBC and load data for normal time resolution data
    sbc=Sbc()
    sbc.experiment.set("pi3b")
    df_multishot = sbc.get(asset="reconstruction_profile", shot=shot, column=['q'], criteria="statistic = mean OR statistic = sigma")
    df_multishot = df_multishot[(df_multishot.t >= t_min) & (df_multishot.t <= t_max)]
        # set up SBC and load data for high time resolution data
    sbc.project.set("high_time_res_recon_crashes")
    df_multishot_high_res = sbc.get(asset="reconstruction_profile", shot=shot, column=['q'], criteria="statistic = mean OR statistic = sigma")
    df_multishot_high_res = df_multishot_high_res[(df_multishot_high_res.t >= t_min) & (df_multishot_high_res.t <= t_max)]

    # save plotting arrays for low time res data
        # set up arrays
    x_plotting = []
    y_plotting = []
    shots_plotting = []
    times_plotting = []
    # iterate and save data
    df = df_multishot
    df = df[df.statistic == 'mean']
    # get times, q0, q95 and plot
        # load scrambled arrays
    times = np.array(np.unique(df.t))
    q0 = np.array(df[df.psibar==0].q)
    q95 = np.array(df[df.psibar==0.95].q)
        # sort arrays
    times_ind = np.argsort(times)
    times_ind_0 = np.argsort(np.array(df[df.psibar == 0].t))
    times_ind_95 = np.argsort(np.array(df[df.psibar == 0.95].t))
    times = np.array([times[i] for i in times_ind])
    q0 = np.array([q0[i] for i in times_ind_0])
    q95 = np.array([q95[i] for i in times_ind_95])
    # gun_flux_arr = np.array([gun_flux[s] for i in times])
    
    if min(len(q0), len(q95), len(times)) > 0:   # len(gun_flux_arr)
        x_plotting.append(q0)
        y_plotting.append(q95)
        shots_plotting.append(shot)
        times_plotting.append(times)
        # col_plotting.append(gun_flux_arr)
    
    # save plotting arrays for high res data
    # save plotting arrays for low time res data
        # set up arrays
    x_plotting_high_res = []
    y_plotting_high_res = []
    shots_plotting_high_res = []
    times_plotting_high_res = []
    # iterate and save data
    df = df_multishot_high_res
    df = df[df.statistic == 'mean']
    # get times, q0, q95 and plot
        # load scrambled arrays
    times = np.array(np.unique(df.t))
    q0 = np.array(df[df.psibar==0].q)
    q95 = np.array(df[df.psibar==0.95].q)
        # sort arrays
    times_ind = np.argsort(times)
    times_ind_0 = np.argsort(np.array(df[df.psibar == 0].t))
    times_ind_95 = np.argsort(np.array(df[df.psibar == 0.95].t))
    times = np.array([times[i] for i in times_ind])
    q0 = np.array([q0[i] for i in times_ind_0])
    q95 = np.array([q95[i] for i in times_ind_95])
    # gun_flux_arr = np.array([gun_flux[s] for i in times])
    
    if min(len(q0), len(q95), len(times)) > 0:   # len(gun_flux_arr)
        x_plotting_high_res.append(q0)
        y_plotting_high_res.append(q95)
        shots_plotting_high_res.append(shot)
        times_plotting_high_res.append(times)
        # col_plotting.append(gun_flux_arr)

    # generate colormaps for timing
        # set proxy array
    a=times_plotting[0]
        # use proxy array to generate the colormap
    norm = Normalize(vmin=np.min(a), vmax=np.max(a))
    cmap = mpl.colormaps.get_cmap('plasma')
    colors = cmap(norm(a))
    
    # set size of figure
    fig = plt.figure(figsize=(4.5,4.2), dpi=350)
    font_min=12
    
    # plot traces and scatter
    for i in range(len(x_plotting)):
        # plt.plot(x_plotting_high_res[i], y_plotting_high_res[i], alpha=0.4, color='k', zorder=0)
        plt.scatter(x_plotting_high_res[i], y_plotting_high_res[i], alpha=0.4, color='k', zorder=0, marker='.')
        plt.scatter(x_plotting[i], y_plotting[i], c=colors, zorder=1)
            
    # set limits
    # plt.xlim([0.35, 1])
    # plt.ylim([1.6,1.85])
    
    # add labels to plot
        # title and axes
    plt.xlabel(r"$q_0$", fontsize=font_min+2)
    plt.ylabel(r"$q_{95}$", fontsize=font_min+2)
    plt.tick_params(axis='both', labelsize=font_min)
    plt.grid(True, alpha=0.4)
        # legend
    for i in range(len(times_plotting[0])):
        plt.scatter([],[], color=colors[i], label=f"t={int(times_plotting[0][i]*1000)}ms")
    plt.legend(ncol=2, fontsize=font_min, columnspacing=0.5, handletextpad=0.3, handlelength=1.5)
        # layout
    plt.tight_layout()
    
    # save figure
    plt.savefig(f"/home/jupyter-humerben/axuv_paper/plot_outputs/q/q0_q95_{shot}.png")
    plt.close()
    
    return


def plot_qmin_t_multishot(shots_1, shots_2=[], t_lims=[0,1000], plot_by_shot=True, plot_all_together=True):
    shots = list(np.append(shots_1, shots_2))
    sbc=Sbc()
    sbc.experiment.set("pi3b")
    df_multishot = sbc.get(asset="reconstruction_profile", shot=shots, column=['q'], criteria="statistic = mean")
    df_multishot = df_multishot[(df_multishot.t >= t_lims[0]) & (df_multishot.t <= t_lims[1])]
    
    qmin_by_shot_1 = {}
    qmin_all_shot_1 = []
    t_all_shot_1 = []
    for shot in shots_1:
        df = df_multishot[df_multishot.shot == shot]
        times = np.sort(np.unique(df.t))
        qmin_singleshot = []
        for time in times:
            qmin_singletime = np.min(df[df.t==time].q)
            qmin_singleshot.append(qmin_singletime)
            qmin_all_shot_1.append(qmin_singletime)
            t_all_shot_1.append(time)
        
        qmin_by_shot_1[shot] = {'q':qmin_singleshot, 't':times}
    
    qmin_by_shot_2 = {}
    qmin_all_shot_2 = []
    t_all_shot_2= []
    for shot in shots_2:
        df = df_multishot[df_multishot.shot == shot]
        times = np.sort(np.unique(df.t))
        qmin_singleshot = []
        for time in times:
            qmin_singletime = np.min(df[df.t==time].q)
            qmin_singleshot.append(qmin_singletime)
            qmin_all_shot_2.append(qmin_singletime)
            t_all_shot_2.append(time)
        
        qmin_by_shot_2[shot] = {'q':qmin_singleshot, 't':times}
    
    # sort arrays
    t_sorting_1 = np.argsort(t_all_shot_1)
    q_1 = [qmin_all_shot_1[int(i)] for i in t_sorting_1]
    t_1 = [t_all_shot_1[int(i)] for i in t_sorting_1]
    t_sorting_2 = np.argsort(t_all_shot_2)       
    q_2 = [qmin_all_shot_2[int(i)] for i in t_sorting_2]
    t_2 = [t_all_shot_2[int(i)] for i in t_sorting_2]
    
    mean_1, std_1 = rolling_stats(q_1, t_1, 1/1000)
    mean_2, std_2 = rolling_stats(q_2, t_2, 1/1000)
    
    
    if plot_by_shot:
        plt.figure(dpi=250)
        for k in qmin_by_shot_1:
            plt.plot(qmin_by_shot_1[k]['t'], qmin_by_shot_1[k]['q'])
            plt.xlabel(r'$t\,\mathrm{[s]}$')
            plt.ylabel(r'$q_\mathrm{min}$')
        plt.savefig("/home/jupyter-humerben/axuv_paper/plot_outputs/q/qmin_t_multiline.png")
        plt.close()
    
    if plot_all_together:
        plt.figure(dpi=250)
        plt.plot(t_1, mean_1, label='Non-sustain', zorder=3)
        plt.fill_between(t_1, mean_1+std_1, mean_1-std_1, alpha=0.1, zorder=2)
        plt.plot(t_2, mean_2, label='Sustain', zorder=3)
        plt.fill_between(t_2, mean_2+std_2, mean_2-std_2, alpha=0.1, zorder=2)
        plt.xlabel(r'$t\,\mathrm{[s]}$')
        plt.ylabel(r'$q_\mathrm{min}$')
        plt.xlim(t_lims)
        plt.legend()
        plt.grid(True, alpha=0.4, zorder=1)
        plt.savefig("/home/jupyter-humerben/axuv_paper/plot_outputs/q/qmin_t_stats.png")
        plt.close()
    
    return
    


def rolling_stats(y, x, window_width):
    """
    Compute rolling mean and standard deviation of y(x) with a fixed window width in x.

    Parameters:
    - y: array-like, values of the function y(x)
    - x: array-like, corresponding x values (must be sorted in ascending order)
    - window_width: float, the width of the rolling window in x

    Returns:
    - y_mean: array of same shape as y, containing rolling mean
    - y_std: array of same shape as y, containing rolling std deviation
    """
    y = np.asarray(y)
    x = np.asarray(x)
    y_mean = np.full_like(y, np.nan, dtype=np.float64)
    y_std = np.full_like(y, np.nan, dtype=np.float64)

    n = len(x)
    for i in range(n):
        # Define window bounds
        x_left = x[i] - window_width / 2
        x_right = x[i] + window_width / 2
        
        # Find indices within window
        in_window = (x >= x_left) & (x <= x_right)
        y_window = y[in_window]

        if len(y_window) > 0:
            y_mean[i] = np.mean(y_window)
            y_std[i] = np.std(y_window)

    return y_mean, y_std

    

if __name__ == '__main__':
    # plot_q0_q95_t(22605, [0, 0.01])
    
    non_sustain = pd.read_csv("/home/jupyter-humerben/axuv_paper/datasets/non_sustain_dataset.csv")
    sustain = pd.read_csv("/home/jupyter-humerben/axuv_paper/datasets/sustain_dataset.csv")
    non_sustain_shots = list(np.array(non_sustain['shot']))
    sustain_shots = list(np.array(sustain['shot']))
    plot_qmin_t_multishot(non_sustain_shots, sustain_shots, [0,0.012])