from gf_sandbox_client.sbc import Sbc

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

from data_loaders.load_crash_data import load_crash_data


def calc_dw_over_crash_multishot(shots, high_res=False):
    crash_df = load_crash_data(shots)
    recon_df = Sbc().get(project='default', shot=shots, experiment='pi3b', asset='reconstruction', columns=['li1'], criteria='statistic = mean and t = 0.001')
    crash_shots = np.unique(crash_df.shot)
    recon_shots = np.unique(recon_df.shot)
    dw_multishot = []
    t_avg_multishot = []
    shot_multishot = []
    for shot in shots:
        if shot not in crash_shots or shot not in recon_shots:
            continue
        dw, t_pre, t_post = calc_dw_over_crash_singleshot(shot, high_res)
        dw_multishot.append(dw)
        t_avg_multishot.append((t_post+t_pre)/2)
        shot_multishot.append(shot)
        
    df = pd.DataFrame({'shot':shot_multishot, 'dW':dw_multishot})
    
    for i in range(len(dw_multishot)):
        if dw_multishot[i] > 0:
            print(shot_multishot[i])
    

    plt.scatter(t_avg_multishot, dw_multishot)
    plt.grid(True)
    plt.savefig("test.png")
    
    return


def calc_dw_over_crash_singleshot(shot, high_res=False):
    # get recon data
    sbc = Sbc()
    sbc.project.set("high_time_res_recon_crashes") if high_res else None
    sbc.experiment.set('pi3b')
    sbc.asset.set('reconstruction_profile')
    df = sbc.get(shots=shot, columns=['Rmax', 'Rmin', 'Javg', 'Bpol', 'Volume'], criteria='statistic = mean')
    df = df[df.psibar > 0]
    
    # get crash data
    crash_df = load_crash_data([shot])
    pre_crash_time = min(crash_df.pre_crash_t)
    post_crash_time = min(crash_df.post_crash_t)
    
    t_recon = np.sort(np.unique(df.t))
    post_crash_recon_t = max(t_recon)
    for time in t_recon:
        if time < pre_crash_time:
            pre_crash_recon_t = time
            
        if time > post_crash_time and time <= post_crash_recon_t:
            post_crash_recon_t = time
    
    pre_crash_recon_df = df[df.t == pre_crash_recon_t].sort_values(by='psibar').reset_index(drop=True)
    post_crash_recon_df = df[df.t == post_crash_recon_t].sort_values(by='psibar').reset_index(drop=True)
    
    dw = calc_dw_from_pre_post_dfs(pre_crash_recon_df, post_crash_recon_df)
    
    print(dw)
    
    
    
    
    return dw, pre_crash_time, post_crash_time


def calc_dw_from_pre_post_dfs(pre_crash_recon_df, post_crash_recon_df):
    dw_test_arr = []
    dw_tot = 0
    for i in range(len(pre_crash_recon_df)):
        if i == 0:
            dr = (post_crash_recon_df.Rmax.iloc[i] - post_crash_recon_df.Rmin.iloc[i])/2 - (pre_crash_recon_df.Rmax.iloc[i] - pre_crash_recon_df.Rmin.iloc[i])/2
            jxB = -1*((post_crash_recon_df.Javg.iloc[i] * pre_crash_recon_df.Bpol.iloc[i]) + (pre_crash_recon_df.Javg.iloc[i] * post_crash_recon_df.Bpol.iloc[i]))
            dV = ((post_crash_recon_df.Volume.iloc[i] - 0) + (pre_crash_recon_df.Volume.iloc[i] - 0)) / 2
            dw = -0.5 * dr * jxB * dV

        else:
            dr = (post_crash_recon_df.Rmax.iloc[i] - post_crash_recon_df.Rmin.iloc[i])/2 - (pre_crash_recon_df.Rmax.iloc[i] - pre_crash_recon_df.Rmin.iloc[i])/2
            jxB = -1*((post_crash_recon_df.Javg.iloc[i] * pre_crash_recon_df.Bpol.iloc[i]) + (pre_crash_recon_df.Javg.iloc[i] * post_crash_recon_df.Bpol.iloc[i]))
            dV = ((post_crash_recon_df.Volume.iloc[i] - post_crash_recon_df.Volume.iloc[i-1]) + (pre_crash_recon_df.Volume.iloc[i] - pre_crash_recon_df.Volume.iloc[i-1])) / 2
            dw = -0.5 * dr * jxB * dV   

        dw_tot += dw
        
        dw_test_arr.append(dw)
    
    return dw_tot



if __name__ == '__main__':
    non_sustain = pd.read_csv("/home/jupyter-humerben/axuv_paper/datasets/non_sustain_dataset.csv")
    
    non_sustain_shots = list(np.array(non_sustain['shot']))
    
    calc_dw_over_crash_multishot(non_sustain_shots)
    
    # calc_dw_over_crash_singleshot(22289)