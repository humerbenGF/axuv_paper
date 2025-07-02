# import standard files
###################################################################################################
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


# import personal files
###################################################################################################
from data_loaders.load_crash_data import load_crash_data


def generate_first_crash_time_stats_for_shot_list(shots, printouts=True):
    df = load_crash_data(shots)
    
    first_crash_times = []
    for shot in shots:
        df_ss = df[df.shot == shot]
        if len(df.shot) > 0:
            first_crash_t = min(np.array(df_ss.t))
            first_crash_times.append(first_crash_t)
            
    mean = np.mean(first_crash_times)
    std = np.std(first_crash_times)
    
    if printouts:
        print("mean:", mean, "stdev:", std)
    
    return

def generate_first_crash_time_stats_for_multiple_groups(shots_multi_group, group_names, printouts=True, plots=False, title=False):
    first_crashes_multi_group = []
    stats_multi_group = {'mean':[], 'std':[]}
    for shots in shots_multi_group:
        df = load_crash_data(shots)
    
        first_crash_times = []
        for shot in shots:
            df_ss = df[df.shot == shot]
            if len(df_ss.t) > 0:
                first_crash_t = min(np.array(df_ss.t))*1000
                first_crash_times.append(first_crash_t)
        
        first_crashes_multi_group.append(first_crash_times)
        stats_multi_group['mean'].append(np.mean(first_crash_times))
        stats_multi_group['std'].append(np.std(first_crash_times))

    if printouts:
        for g in range(len(shots_multi_group)):
            print(f"group {g+1} ({group_names[g]})\n\tmean: {str(stats_multi_group['mean'][g])[:5]}ms, stdev: {str(stats_multi_group['std'][g])[:5]}ms")

    if plots:
        fig = plt.figure(figsize=(6.75,4), dpi=400)
        font_min=12
        # set up colormap
        N = len(first_crashes_multi_group)
        cmap = plt.get_cmap('jet')
        colors = [cmap(i / (N - 1)) for i in range(N)]
        
        for g in range(len(first_crashes_multi_group)):
            # seaborn.ecdfplot(first_crashes_multi_group[g], color=colors[g], label=group_names[g])
            seaborn.kdeplot(first_crashes_multi_group[g], color=colors[g], label=group_names[g], fill=True, alpha=0.1, zorder=(g+1))
        
        # labels
        plt.grid(True, zorder=0, alpha=0.4)
        plt.title(r"Probability Density Function $f(t)$ of First Crash Occurance" if title else None, fontsize=font_min+6)
        plt.xlabel(r"$t\,\mathrm{[ms]}$", fontsize=font_min+2)
        plt.ylabel(r"$f(t)\,\mathrm{[ms^{-1}]}$", fontsize=font_min+2)
        plt.legend(loc='upper left', fontsize=font_min)
        plt.tick_params(axis='both', labelsize=font_min)
        plt.tight_layout()
            
        plt.savefig("/home/jupyter-humerben/axuv_paper/plot_outputs/crashes/kde_of_crashes_multigroup.png")
    
    return