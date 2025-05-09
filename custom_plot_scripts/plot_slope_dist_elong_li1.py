# import standard libraries
###################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib import cm
from matplotlib.colors import Normalize
import seaborn as sns



# import sandbox client
###################################################################################################
from gf_sandbox_client.sbc import Sbc

# import personal files
###################################################################################################
from data_loaders.load_crash_data import load_crash_data
from processing_tools.interpolation import linear_interp_extrap as lin_interp


def plot_slope_dist_elong_li1(shots, t_lims=[0,1000], gap_to_first_crash=0.001, slope_lims=[-4,4], shot_type="sustain"):
    # unpack t limits
    t_min, t_max = t_lims[0], t_lims[1]
    
    # load crash data
    crash_df_multishot = load_crash_data(list(np.append(shots, [21441])))
    
    # load and set up multishot dataframes
        # set up SBC and load data for normal time resolution data
    sbc=Sbc()
    sbc.experiment.set("pi3b")
    df_multishot = sbc.get(asset="reconstruction", shot=shots, column=['li1', 'elong_lfs'], criteria="statistic = mean")
    df_multishot = df_multishot[(df_multishot.t >= t_min) & (df_multishot.t <= t_max)]
        # set up SBC and load data for high time resolution data
    sbc.project.set("high_time_res_recon_crashes")
    df_multishot_high_res = sbc.get(asset="reconstruction", shot=shots, column=['li1', 'elong_lfs'], criteria="statistic = mean")
    df_multishot_high_res = df_multishot_high_res[(df_multishot_high_res.t >= t_min) & (df_multishot_high_res.t <= t_max)]
    df_multishot_high_res = df_multishot
    
    slopes = []
    
    for s in range(len(shots)):
        df = df_multishot_high_res[df_multishot_high_res.shot == shots[s]]
        crash_df = crash_df_multishot[crash_df_multishot.shot == shots[s]]
        if len(crash_df.t) == 0 or len(df.t) == 0:
            continue
        
        times = np.sort(np.unique(np.array(df.t)))
        start_time = calc_nearest_time(times, 0.0015)
        end_time = calc_nearest_time(times, min([min(np.array(crash_df.t))-gap_to_first_crash, 0.007])), 
        d_li1 = df[df.t == end_time].li1.iloc[0] - df[df.t == start_time].li1.iloc[0]
        d_elong = df[df.t == end_time].elong_lfs.iloc[0] - df[df.t == start_time].elong_lfs.iloc[0]
        slope = d_elong/d_li1
        if slope > slope_lims[0] and slope < slope_lims[1]:
            slopes.append(slope)
    
    sns.histplot(slopes, stat="density", bins=20, alpha=0.5, label="Histogram")
    sns.kdeplot(slopes, fill=False, color='k', linewidth=3, label='KDE')
    plt.title(r"Distribution of Slopes $d\kappa/d\ell_{i1}$ Before the First Crash"+f"\n{len(slopes)} Sustain Shots")
    plt.xlabel(r"$d\kappa/d\ell_{i1}$")
    print("mean:", np.mean(slopes), "std:", np.std(slopes))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"/home/jupyter-humerben/axuv_paper/plot_outputs/slope_dist_elong_li1_{shot_type}.png")
        
        
def calc_nearest_time(times, time):
    """
    Find the value in 'times' that is closest to 'time'.

    Parameters:
        times (array-like): Array of time values.
        time (float): Target time.

    Returns:
        float: The nearest value in 'times' to 'time'.
    """
    times = np.asarray(times)
    idx = np.abs(times - time).argmin()
    return times[idx]