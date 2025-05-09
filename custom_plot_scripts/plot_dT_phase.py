# import standard libraries
###################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import sandbox client
###################################################################################################
from gf_sandbox_client.sbc import Sbc

# import personal files
###################################################################################################
from data_loaders.load_crash_data import load_crash_data as get_crash_df
from processing_tools.turn_time_to_crash_phase import get_phase_from_time_singlepoint_crash as get_phase


def plot_dT_phase(shots):
    thomson_R_positions = [600, 730, 900]
    
    # get thomson data
    thomson_df_multishot = Sbc().get(experiment='pi3b', asset="thomson_temperature", shot=shots)
    general_df = Sbc().get(experiment='pi3b', asset="reconstruction", shot=shots, column='elong_lfs', criteria='statistic = mean')
    
    crash_df_multishot = get_crash_df(shots)
    max_value = 0
    min_value = 10000000
    thomson_data_multishot = {}
    for s in range(len(shots)):
        thomson_data = {}
        lifetime = max(np.unique(general_df.t))+1/1000
        crash_df = crash_df_multishot[crash_df_multishot.shot == shots[s]]
        thomson_df = thomson_df_multishot[thomson_df_multishot.shot == shots[s]]
        for pos in thomson_R_positions:
            if pos in np.array(thomson_df.R):
                # load data
                T = thomson_df[thomson_df.R == pos].Te
                T_err = thomson_df[thomson_df.R == pos].Te_err
                t = thomson_df[thomson_df.R == pos].t
                t_ind_sort = np.argsort(t)
                T = [T.iloc[i] for i in t_ind_sort]
                T_err = [T_err.iloc[i] for i in t_ind_sort]
                t = [t.iloc[i] if t.iloc[i] > 0 else 0 for i in t_ind_sort]
                phase = [get_phase(crash_df, (t[i]), lifetime) for i in range(len(t))]
                dphase = [get_phase(crash_df, (t[i]+t[i+1])/2, lifetime) for i in range(len(t)-1)]
                dT = [(T[i+1]-T[i])/(phase[i+1]-phase[i]) for i in range(len(t)-1)]
                # save data to dict
                thomson_data[pos] = {'dTe': dT, 'phase':dphase}
                
                if len(dT) > 0:
                    if max(dT) > max_value:
                        max_value = max(dT)
                    if min(dT) < min_value:
                        min_value = min(dT)
        
        thomson_data_multishot[shots[s]] = thomson_data
        
    
    fig, axs = plt.subplots(1, 3, figsize=(10,5))
    
    thomson_colors = ['red', 'blue', 'black']
    
    for shot in thomson_data_multishot.keys():
        for i in range(len(thomson_R_positions)):
            if thomson_R_positions[i] in thomson_data_multishot[shot].keys():
                axs[i].scatter(x=thomson_data_multishot[shot][thomson_R_positions[i]]['phase'], y=thomson_data_multishot[shot][thomson_R_positions[i]]['dTe'],
                            color=thomson_colors[i], zorder=1)
                axs[i].plot(thomson_data_multishot[shot][thomson_R_positions[i]]['phase'], thomson_data_multishot[shot][thomson_R_positions[i]]['dTe'],
                            color='grey', linewidth=0.5, zorder=0)
                
    for i in range(len(thomson_R_positions)):
        axs[i].set_title(r"Thomson Scattering $T_e$"+f"\nR={thomson_R_positions[i]}mm")
        axs[i].set_xlabel(r"Phase")
        axs[i].set_ylim(1.1*min_value, 1.1*max_value)
        
    axs[0].set_ylabel(r"$dT_e/dPhase$")
    
    for i, ax in enumerate(axs):
        if i > 0:
            ax.set_yticklabels([])   # Hide tick labels
        
    plt.tight_layout()
    plt.savefig("/home/jupyter-humerben/axuv_paper/plot_outputs/thomson_slope_sustain_early_crash.png")
    
    
    return
    
    
if __name__ == '__main__':
    plot_dT_phase([22603, 22605])