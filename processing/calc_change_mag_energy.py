from gf_sandbox_client.sbc import Sbc

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import pickle as pkl

from data_loaders.load_crash_data import load_crash_data


def check_if_crash_is_IRE():
    dW_df = pd.read_csv("/home/jupyter-humerben/axuv_paper/datasets/dW_dataset.csv")
    
    with open('/home/jupyter-humerben/axuv_paper/IRE_exploration/IRE_Bprobe_detection/data/output.pkl', 'rb') as f:
        bprobe_df = pkl.load(f)
        
    print(bprobe_df)
        
    bprobe_unique = np.unique(bprobe_df.shot)
    
    stats_dict = {'overall_shots':0, 'overall_confirmed':0, 'sustain_shots':0, 'sustain_confirmed':0, 'non_sustain_shots':0, 'non_sustain_confirmed':0,
                  'pressure_driven_shots':0, 'pressure_driven_confirmed':0, 'magnetically_driven_shots':0, 'magnetically_driven_confirmed':0}
    for shot in np.unique(dW_df.shot):
        if shot in bprobe_unique:
            ire_confirmed = False
            for i in range(len(bprobe_df[bprobe_df.shot == shot])):
                if (bprobe_df[bprobe_df.shot == shot].omega.iloc[i] < -1000 and 
                    bprobe_df[bprobe_df.shot == shot].num_points.iloc[i] > 2 and 
                    bprobe_df[bprobe_df.shot == shot].omega_R2.iloc[i] > 0.95):
                    # print(bprobe_df[bprobe_df.shot == shot].omega.iloc[i], bprobe_df[bprobe_df.shot == shot].omega_R2.iloc[i])
                    ire_confirmed = True

            stats_dict['overall_confirmed'] += 1 if ire_confirmed else 0
            stats_dict['overall_shots'] += 1

            if dW_df[dW_df.shot == shot].sustain.iloc[0] > 0:
                stats_dict['sustain_confirmed'] += 1 if ire_confirmed else 0
                stats_dict['sustain_shots'] += 1

            if dW_df[dW_df.shot == shot].sustain.iloc[0] == 0:
                stats_dict['non_sustain_confirmed'] += 1 if ire_confirmed else 0
                stats_dict['non_sustain_shots'] += 1
                
            if dW_df[dW_df.shot == shot].dW.iloc[0] > 0:
                stats_dict['pressure_driven_confirmed'] += 1 if ire_confirmed else 0
                stats_dict['pressure_driven_shots'] += 1
                
            if dW_df[dW_df.shot == shot].dW.iloc[0] < 0:
                stats_dict['magnetically_driven_confirmed'] += 1 if ire_confirmed else 0
                stats_dict['magnetically_driven_shots'] += 1
            
        
    keys = list(stats_dict.keys())
    for i in range(int(len(keys)/2)):
        print(f"{stats_dict[keys[2*i+1]]}/{stats_dict[keys[2*i]]}: {str(stats_dict[keys[2*i+1]]/stats_dict[keys[2*i]]*100)[:4]}% // {keys[2*i+1]}, {keys[2*i]}")
    
    return


def generate_kde_of_dW():
    df = pd.read_csv("/home/jupyter-humerben/axuv_paper/datasets/dW_dataset.csv")
    
    dW_sustain = np.array(df[df.sustain > 0].dW)
    dW_non_sustain = np.array(df[df.sustain == 0].dW)
    
    plt.figure(dpi=300)
    
    fontsize=12
    
    sb.kdeplot(dW_sustain, label="sustain", fill=True, zorder=3)
    sb.kdeplot(dW_non_sustain, label="non-sustain", fill=True, zorder=3)
    
    plt.xlim(-6,3)
    plt.xlabel(r'$\delta W_\mathrm{mag}\,\mathrm{[kJ]}$', fontsize=fontsize)
    plt.ylabel(r"Density $\mathrm{[kJ]^{-1}}$", fontsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)
    
    plt.grid(True, alpha=0.5, zorder=1)
    plt.legend(fontsize=fontsize-2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig("/home/jupyter-humerben/axuv_paper/plot_outputs/energy_principle/energy_principle_kde.png")
    plt.close()

    return


def calc_dw_over_crash_multishot(shots, min_shot=0, high_res=False):
    crash_df = load_crash_data(shots)
    recon_df = Sbc().get(project='default', shot=shots, experiment='pi3b', asset='reconstruction', columns=['Ipl'], criteria='statistic = mean and t = 0.001')
    sustain_df = Sbc().get(project='default', user='humerben', shot=shots, columns='sustain', asset='shot_log', experiment='pi3b')
    crash_shots = np.unique(crash_df.shot)
    recon_shots = np.unique(recon_df.shot)
    dw_multishot = []
    dli1_multishot = []
    dkappa_multishot = []
    sustain_multishot = []
    t_avg_multishot = []
    shot_multishot = []
    for shot in shots:
        if shot not in crash_shots or shot not in recon_shots or shot <= min_shot:
            continue
        dw, dli1, dkappa, t_pre, t_post = calc_dw_over_crash_singleshot(shot, high_res)
        dw_multishot.append(dw/1000)
        dli1_multishot.append(dli1)
        dkappa_multishot.append(dkappa)
        sustain_multishot.append(sustain_df[sustain_df.shot == shot].sustain.iloc[0]/1000)
        t_avg_multishot.append((t_post+t_pre)/2*1000)
        shot_multishot.append(shot)
        
    df = pd.DataFrame({'shot':shot_multishot, 'dW':dw_multishot, 'dli1':dli1_multishot,
                       'dkappa':dkappa_multishot, 't':t_avg_multishot, 'sustain':sustain_multishot})
    df.to_csv("/home/jupyter-humerben/axuv_paper/datasets/dW_dataset.csv", index=False)
    
    print_potential_IRE=False
    if print_potential_IRE:
        for i in range(len(dw_multishot)):
            if dw_multishot[i] > 0:
                print(shot_multishot[i])

    
    fontsize=12
    # dw
    ###################################
    plt.figure(dpi=300)
    plt.scatter(t_avg_multishot, dw_multishot, c=sustain_multishot, cmap='viridis', marker='.', zorder=2)
    cbar = plt.colorbar()
    cbar.set_label(r"$V_\mathrm{sust}\,\mathrm{[kV]}$", fontsize=fontsize+2)
    cbar.ax.tick_params(axis='both', labelsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)
    
    plt.ylim(-4, 2)
    plt.ylabel(r'$\delta W_\mathrm{mag}\,\mathrm{[kJ]}$', fontsize=fontsize+2)
    plt.xlabel(r'$t\,\mathrm{[ms]}$', fontsize=fontsize+2)
    plt.grid(True, zorder=0, alpha=0.7)
    plt.tight_layout()
    plt.savefig("/home/jupyter-humerben/axuv_paper/plot_outputs/energy_principle/energy_principle_multishot.png")
    plt.close()
    
    
    # li1
    ###################################
    plt.figure(dpi=300)
    plt.scatter(t_avg_multishot, dli1_multishot, c=sustain_multishot, cmap='viridis', marker='.', zorder=2)
    cbar = plt.colorbar()
    cbar.set_label(r"$V_\mathrm{sust}\,\mathrm{[kV]}$", fontsize=fontsize+2)
    cbar.ax.tick_params(axis='both', labelsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)

    plt.ylabel(r'$\Delta \ell_\mathrm{i1}$', fontsize=fontsize+2)
    plt.xlabel(r'$t\,\mathrm{[ms]}$', fontsize=fontsize+2)
    plt.grid(True, zorder=0, alpha=0.7)
    plt.tight_layout()
    plt.savefig("/home/jupyter-humerben/axuv_paper/plot_outputs/energy_principle/enery_principle_li1_multishot.png")
    plt.close()
    
    # elongation
    ###################################
    plt.figure(dpi=300)
    plt.scatter(t_avg_multishot, dkappa_multishot, c=sustain_multishot, cmap='viridis', marker='.', zorder=2)
    cbar = plt.colorbar()
    cbar.set_label(r"$V_\mathrm{sust}\,\mathrm{[kV]}$", fontsize=fontsize+2)
    cbar.ax.tick_params(axis='both', labelsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)

    plt.ylabel(r'$\Delta \kappa$', fontsize=fontsize+2)
    plt.xlabel(r'$t\,\mathrm{[ms]}$', fontsize=fontsize+2)
    plt.grid(True, zorder=0, alpha=0.7)
    plt.tight_layout()
    plt.savefig("/home/jupyter-humerben/axuv_paper/plot_outputs/energy_principle/enery_principle_kappa_multishot.png")
    plt.close()
    
    # k-li1
    ###################################
    plt.figure(dpi=300)
    plt.scatter(dli1_multishot, dkappa_multishot, c=sustain_multishot, cmap='viridis', marker='.', zorder=2)
    cbar = plt.colorbar()
    cbar.set_label(r"$V_\mathrm{sust}\,\mathrm{[kV]}$", fontsize=fontsize+2)
    cbar.ax.tick_params(axis='both', labelsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)

    plt.ylabel(r'$\Delta \kappa$', fontsize=fontsize+2)
    plt.xlabel(r'$\Delta \ell_\mathrm{i1}$', fontsize=fontsize+2)
    plt.grid(True, zorder=0, alpha=0.7)
    plt.tight_layout()
    plt.savefig("/home/jupyter-humerben/axuv_paper/plot_outputs/energy_principle/dli1_dkappa_multishot.png")
    plt.close()
    
    return


def calc_dw_over_crash_singleshot(shot, high_res=False):
    # get recon data
    sbc = Sbc()
    sbc.project.set("high_time_res_recon_crashes") if high_res else sbc.project.set("default")
    sbc.experiment.set('pi3b')
    sbc.asset.set('reconstruction_profile')
    df = sbc.get(shots=shot, columns=['Rmax', 'Rmin', 'Javg', 'Bpol', 'Volume'], criteria='statistic = mean')
    df_li1 = sbc.get(shots=shot, asset='reconstruction', columns=['li1', 'elong_lfs'], criteria='statistic = mean')
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
    pre_crash_recon_li1 = df_li1[df_li1.t == pre_crash_recon_t].li1.iloc[0]
    pre_crash_recon_kappa = df_li1[df_li1.t == pre_crash_recon_t].elong_lfs.iloc[0]
    post_crash_recon_df = df[df.t == post_crash_recon_t].sort_values(by='psibar').reset_index(drop=True)
    post_crash_recon_li1 = df_li1[df_li1.t == post_crash_recon_t].li1.iloc[0]
    post_crash_recon_kappa = df_li1[df_li1.t == post_crash_recon_t].elong_lfs.iloc[0]
    
    dw = calc_dw_from_pre_post_dfs(pre_crash_recon_df, post_crash_recon_df)
    dli1 = post_crash_recon_li1-pre_crash_recon_li1
    dkappa = post_crash_recon_kappa-pre_crash_recon_kappa
    
    return dw, dli1, dkappa, pre_crash_time, post_crash_time



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
    sustain = pd.read_csv("/home/jupyter-humerben/axuv_paper/datasets/sustain_dataset.csv")
    
    non_sustain_shots = list(np.array(non_sustain['shot']))
    sustain_shots = list(np.array(sustain['shot']))
    
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif']
    
    # calc_dw_over_crash_singleshot(22289)
    # calc_dw_over_crash_multishot(list(np.append(non_sustain_shots, sustain_shots)))
    
    # generate_kde_of_dW()
    check_if_crash_is_IRE()