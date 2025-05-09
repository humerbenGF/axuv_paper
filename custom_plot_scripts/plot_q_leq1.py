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


def plot_q_leq_1_ov_crash_timing(shots, shots2, sbc=Sbc(), first_n_crashes=1):
    # prepare and unpack crash data
    crash_df = load_crash_data(shots)
    crash_df_2 = load_crash_data(shots2)
    
    # load and set up multishot dataframes
        # set up SBC and load data
    sbc.experiment.set("pi3b")
    df_multishot = sbc.get(asset="reconstruction_profile", shot=shots, column=['q'], criteria="statistic = mean OR statistic = sigma")
    df_multishot_2 = sbc.get(asset="reconstruction_profile", shot=shots2, column=['q'], criteria="statistic = mean OR statistic = sigma")
    
    # shot list 1
    x_plotting = []
    y_plotting = []
    crash_numbers_plotting = []
    crash_times_plotting = []
    shot_numbers_plotting = []
    
    for shot in shots:
        # set up single shot information
        df = df_multishot[df_multishot["shot"] == shot]
        ss_crash_df = crash_df[crash_df["shot"] == shot]
        times = np.sort(np.unique(df.t))
        crash_times = np.sort(np.unique(ss_crash_df.t))
        
        # check time to see if there are any crashes
        for t_ind in range(len(times) - 1):
            for c_ind in range(len(crash_times)):
                if times[t_ind] < crash_times[c_ind] and times[t_ind+1] > crash_times[c_ind]  and c_ind < first_n_crashes:
                    q_mean = df[(df.statistic == 'mean') & (df.t == times[t_ind]) & (df.psibar == 0)].q.iloc[0]
                    q_sig = df[(df.statistic == 'sigma') & (df.t == times[t_ind]) & (df.psibar == 0)].q.iloc[0]
                    q_prob_weight = norm.cdf(1, loc=q_mean, scale=q_sig)
                    amplitude = ss_crash_df[ss_crash_df.t == crash_times[c_ind]].amplitude.iloc[0]
                    
                    x_plotting.append(amplitude)
                    y_plotting.append(q_prob_weight)
                    crash_numbers_plotting.append(c_ind+1)
                    crash_times_plotting.append(crash_times[c_ind])
                    shot_numbers_plotting.append(shot)
    
    # shot list 2
    x_plotting_2 = []
    y_plotting_2 = []
    crash_numbers_plotting_2 = []
    crash_times_plotting_2 = []
    shot_numbers_plotting_2 = []
    
    for shot in shots2:
        # set up single shot information
        df = df_multishot_2[df_multishot_2["shot"] == shot]
        ss_crash_df = crash_df_2[crash_df_2["shot"] == shot]
        times = np.sort(np.unique(df.t))
        crash_times = np.sort(np.unique(ss_crash_df.t))
        
        # check time to see if there are any crashes
        for t_ind in range(len(times) - 1):
            for c_ind in range(len(crash_times)):
                if times[t_ind] < crash_times[c_ind] and times[t_ind+1] > crash_times[c_ind]  and c_ind < first_n_crashes:
                    q_mean = df[(df.statistic == 'mean') & (df.t == times[t_ind]) & (df.psibar == 0)].q.iloc[0]
                    q_sig = df[(df.statistic == 'sigma') & (df.t == times[t_ind]) & (df.psibar == 0)].q.iloc[0]
                    q_prob_weight = norm.cdf(1, loc=q_mean, scale=q_sig)
                    amplitude = ss_crash_df[ss_crash_df.t == crash_times[c_ind]].amplitude.iloc[0]
                    
                    x_plotting_2.append(amplitude)
                    y_plotting_2.append(q_prob_weight)
                    crash_numbers_plotting_2.append(c_ind+1)
                    crash_times_plotting_2.append(crash_times[c_ind])
                    shot_numbers_plotting_2.append(shot)
    
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=x_plotting, y=y_plotting, 
                             mode='markers', marker=dict(color=crash_times_plotting, colorscale="Viridis", showscale=True, 
                                                         colorbar=dict(title="Crash Time Group 1", x=1.2)), 
                             hovertext=shot_numbers_plotting, hoverinfo="text+x+y", name="Shot Group 1", showlegend=True)
                  )

    fig.add_trace(go.Scatter(x=x_plotting_2, y=y_plotting_2, 
                             mode='markers', marker=dict(color=crash_times_plotting_2, colorscale="Viridis", showscale=True, 
                                                         colorbar=dict(title="Crash Time Group 2", x=1.35), symbol='x'), 
                             hovertext=shot_numbers_plotting_2, hoverinfo="text+x+y", name="Shot Group 2", showlegend=True)
                  )
    
    fig.update_layout(
        title=dict(
            text=f"Plot of P(q<1) Over Amplitude {len(x_plotting)} Crashes<br>",  # Dynamic title
            x=0.5,  # Center title
            y=0.95,
            font=dict(size=22)
        ),
        xaxis_title=r"Crash Amplitude [nA]",
        yaxis_title=f"P(q<1)"  # Replace with actual column name if needed
    )
    
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
    
    
    
    
if __name__ == '__main__':
    andrea_non_sustain_curated = [21262, 21263, 21265, 21268, 21271, 21273, 21282, 21283, 21290, 21347, 21571, 
                                  21572, 21584, 21587, 21588, 21591, 21592, 21593, 21594, 21623, 21624, 21626, 
                                  22285, 22286, 22287, 22288, 22289, 22290, 22293, 22294, 22295, 22304, 22305, 
                                  22306, 22307, 22576, 22591, 22595, 22597, 22599, 22622, 22625, 22656, 22657, 
                                  22662, 22664, 22666, 22669, 22670, 22676, 22678, 22695, 22696, 22697, 22698, 
                                  22699, 22708, 22710, 22712, 22713, 22714, 22715, 22716, 22717, 22718, 22726, 
                                  22727, 22728, 22729, 22731, 22732, 22733, 22734, 22735, 22736, 22737, 22742, 
                                  22743, 22744, 22745, 22746, 22747, 22750, 22751, 22752, 22753, 22754, 22755, 
                                  22800, 22801, 22804, 22805, 22806, 22807, 22808, 22809, 22811, 22812, 22817, 
                                  22818, 22820, 22821, 22822, 22823, 22824, 22826, 22840, 22842, 22843, 22844, 
                                  22881, 22884, 22885, 22886, 22915, 22990]
    
    plot_q_leq_1_ov_crash_timing(andrea_non_sustain_curated, first_n_crashes=1000)