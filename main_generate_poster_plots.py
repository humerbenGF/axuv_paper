# import standard libraries
###################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import sandbox client
###################################################################################################
from gf_sandbox_client.sbc import Sbc

# import plotting scripts
###################################################################################################
    # methods
import custom_plot_scripts.plot_axuv_crash_calcs as plot_axuv_crash_calcs
    # dtm
import custom_plot_scripts.plot_q_J as plot_q_J
import custom_plot_scripts.plot_density_thomson as plot_n_Te
import custom_plot_scripts.plot_lcfs_elongation_li1 as plot_elongation
import custom_plot_scripts.plot_dIpl_li1 as plot_dIpl_dli1
import processing.first_crash_stats_for_shot_grouping as first_crash_stats
    # ire
import custom_plot_scripts.plot_IRE_multipanel_poster as plot_IRE_multipanel
import custom_plot_scripts.plot_lcfs_elongation_phase as plot_elongation_phase
import custom_plot_scripts.plot_crashes_ss_li as plot_crashes_ss_li

# import other scripts
###################################################################################################
    # filtering
from filtering_tools.filter_by_lifetime import filter_by_lifetime
from filtering_tools.filter_by_sustain import filter_by_sustain



def make_shot_list_edges(min_shot, max_shot):
    L = []
    while min_shot <= max_shot:
        L.append(min_shot)
        min_shot += 1

    return L


if __name__ == '__main__':
    # load required shot lists
    shot_list=make_shot_list_edges(19718, 23064)
    
    non_sustain = pd.read_csv("/home/jupyter-humerben/axuv_paper/datasets/non_sustain_dataset.csv")
    sustain = pd.read_csv("/home/jupyter-humerben/axuv_paper/datasets/sustain_dataset.csv")
    
    non_sustain_shots = list(np.array(non_sustain['shot']))
    non_sustain_high_gun_flux_and_formation = [22729,22731,22732,22733,22734,22735,22736,22737,22743,22744,22745,22754,22755]
    non_sustain_low_gun_flux_high_formation = [22285,22286,22287,22288,22289,22290,22622,22625,22626,22628,22629,22630,22631]
    non_sustain_low_gun_flux_and_formation = [22293,22294,22295,22296,22297,22298,22634,22637,22645,22646]
    non_sustain_gun_flux = list(np.array(non_sustain['gun_flux_mWb']))
    sustain_shots = list(np.array(sustain['shot']))
    sustain_shots_small_crash = list(np.array(sustain[sustain.note == 'small_crash'].shot))
    sustain_shots_early_crash = list(np.array(sustain[sustain.note == 'early_or_normal_crash'].shot))
    high_res_shots = [20335, 20346, 21282, 21284, 21314, 21441, 22289, 22603, 22605, 22708, 22885]
    
    
    # plots for poster
        # font
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif']
        # methods
    plot_axuv_crash_calcs.plot_axuv_norm_non_norm(22289, [0,0.018])
    plot_axuv_crash_calcs.plot_axuv_norm_non_norm_combined(22289, [0,0.018])
        # results
            # DTM
    plot_q_J.plot_q_J_side_by_side_singleshot(22289, [0, 0.012], q_lims=[0.5,4.5])
    plot_n_Te.plot_density_thomson_singleshot(22289, ['221', '239', '258'])
    plot_dIpl_dli1.plot_dIpl_dli1_colored_by_gun_flux_grouped(non_sustain_shots, non_sustain_gun_flux,
                                    non_sustain_low_gun_flux_and_formation, non_sustain_low_gun_flux_high_formation, non_sustain_high_gun_flux_and_formation)
    plot_elongation.plot_elongation_li1_triple_matplotlib(non_sustain_low_gun_flux_and_formation, non_sustain_low_gun_flux_high_formation,
                                    non_sustain_high_gun_flux_and_formation, t_lims=[0, 0.012], faded_traces=True, high_res=False)
    first_crash_stats.generate_first_crash_time_stats_for_multiple_groups(
        [non_sustain_low_gun_flux_and_formation, non_sustain_low_gun_flux_high_formation, non_sustain_high_gun_flux_and_formation], 
        [r"$U_{\mathrm{form}}\leq445\,\mathrm{kJ}$"+"\n"+r"$\psi_{\mathrm{inj}}<100\,\mathrm{mWb}$", 
         r"$U_{\mathrm{form}}=647\,\mathrm{kJ}$"+"\n"+r"$\psi_{\mathrm{inj}}<100\,\mathrm{mWb}$", 
         r"$U_{\mathrm{form}}\geq703\,\mathrm{kJ}$"+"\n"+r"$\psi_{\mathrm{inj}}>130\,\mathrm{mWb}$"], plots=True)

            # IRE
    plot_IRE_multipanel.plot_IRE_multipanel(22605, ['221', '239', '258', '276'])
    plot_elongation_phase.plot_elongation_phase_slope_matplotlib(sustain_shots_early_crash)
    plot_crashes_ss_li.plot_percent_chance_crash_before_time_over_ss_li(shot_list, [0.003, 0.005, 0.007, 0.009, 0.011], True)