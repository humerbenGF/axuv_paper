# import standard libraries
###################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


# import sandbox client
###################################################################################################
from gf_sandbox_client.sbc import Sbc

# import personal files
###################################################################################################
from data_loaders.load_shots_since_data import load_shots_since_into_df as load_shots_since
from processing_tools.crash_timings import check_is_crash_before_time as is_early_crash
from data_loaders.load_crash_data import load_crash_data as load_crash


def plot_percent_chance_crash_before_time_over_ss_li(shots, times_of_interest, sustain=True, min_sustain=6000, max_ss_li=200):
    # load in shots since df
    shots_since_df = load_shots_since()
    crash_df = load_crash(shots)
    
    sustain_df = Sbc().get(project='default', user='humerben', shot=shots, columns='sustain', asset='shot_log', experiment='pi3b')
    
    data_dict = {"shot":[], "shots_since_li":[], "is_crash":[], "time_of_interest":[]}
    
    for t in range(len(times_of_interest)):
        for shot in shots:
            if shot in np.array(sustain_df.shot) and shot in np.array(crash_df.shot):
                if sustain and sustain_df[sustain_df.shot == shot].sustain.iloc[0] <= min_sustain:
                    continue
                elif not sustain and sustain_df[sustain_df.shot == shot].sustain.iloc[0] > 0:
                    continue
                elif shots_since_df[shots_since_df.shot == shot].ss_li_pot.iloc[0] > max_ss_li:
                    continue
            else:
                continue
            
            data_dict['shot'].append(shot)
            data_dict['is_crash'].append(is_early_crash(crash_df[crash_df.shot == shot], times_of_interest[t]))
            data_dict['shots_since_li'].append(shots_since_df[shots_since_df.shot == shot].ss_li_pot.iloc[0])
            data_dict['time_of_interest'].append(times_of_interest[t])
            
    data_df_multitime = pd.DataFrame(data_dict)
    
    # make figure
    plt.figure(figsize=(6, 4.5), dpi=300)
    
    # final processing and plotting of loaded data
    for t in range(len(times_of_interest)):
        data_df = data_df_multitime[data_df_multitime.time_of_interest == times_of_interest[t]]
        # Combine values and labels
        x = np.concatenate([np.array(data_df[data_df.is_crash == True].shots_since_li), np.array(data_df[data_df.is_crash == False].shots_since_li)])
        y = np.concatenate([np.ones_like(np.array(data_df[data_df.is_crash == True].shots_since_li)), 
                            np.zeros_like(np.array(data_df[data_df.is_crash == False].shots_since_li))])

        # Sort by x
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        # Smooth percentage using moving average or Gaussian
        window_size = int(len(x_sorted)/10)
        window_sigma = window_size/2
        min_samples = 5
        num_bins = 25
        bin_width = (max(x)-min(x))/num_bins
        smoothed = gaussian_filter1d(y_sorted, sigma=window_sigma)
        
        x_smoothed, smoothed = rolling_mean_by_x(x_sorted, smoothed, bin_width, min_samples=0)

        smoothed_pct = [smoothed[i]*100 for i in range(len(smoothed))]

        plt.plot(x_smoothed, smoothed_pct, label=f'% Chance of Crash at t<{round(times_of_interest[t]*1000, 1)}ms')

    fontsize=12
    # Plotting
    plt.xlabel('Shots Since Lithium Pot Coat', fontsize=fontsize+2)
    plt.ylabel('Percentage Chance of\nCrash Occurance', fontsize=fontsize+2)
    # plt.title(f'Percentage Chance of Crash Occurance Over Shots Since Lithium\n{len(np.unique(data_df_multitime.shot))} Shots, Sustain > {min_sustain}V' 
    #           if sustain else f'Percentage Chance of Crash Occurance Over Shots Since Lithium\n{len(np.unique(data_df_multitime.shot))} Shots, Sustain = 0V', fontsize=fontsize+4)
    plt.grid(True)
    plt.legend(loc='lower right', fontsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.tight_layout()
    plt.savefig("/home/jupyter-humerben/axuv_paper/plot_outputs/crashes/percentage_crash_before_t_ss_li_sustain.png"
                if sustain else "/home/jupyter-humerben/axuv_paper/plot_outputs/crashes/percentage_crash_before_t_ss_li_non_sustain.png")
    
    return


def rolling_mean_by_x(x, event, x_window=5.0, min_samples=5):
    x = np.asarray(x)
    event = np.asarray(event)
    x_sorted_idx = np.argsort(x)
    x_sorted = x[x_sorted_idx]
    event_sorted = event[x_sorted_idx]

    smoothed_prob = np.empty_like(x_sorted, dtype=float)
    smoothed_prob[:] = np.nan  # Default to NaN

    half_window = x_window / 2

    for i, xi in enumerate(x_sorted):
        # Find all points within [xi - half_window, xi + half_window]
        mask = (x_sorted >= xi - half_window) & (x_sorted <= xi + half_window)
        sample_count = np.sum(mask)
        if sample_count >= min_samples:
            smoothed_prob[i] = np.mean(event_sorted[mask])  # % crash

    return x_sorted, smoothed_prob


if __name__ == '__main__':
    shot_list = [i+20000 for i in range(2900)]
    plot_percent_chance_crash_before_time_over_ss_li(shot_list, [0.003, 0.005, 0.007, 0.009, 0.011], sustain=True, max_ss_li=200)