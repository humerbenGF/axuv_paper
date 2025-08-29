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
import custom_plot_scripts.plot_q_J as plot_q_J
import custom_plot_scripts.plot_axuv_crash_calcs as plot_axuv_crash_calcs
import custom_plot_scripts.plot_q_leq1 as plot_q_leq1
import custom_plot_scripts.plot_lcfs_elongation_li1 as plot_elongation
import custom_plot_scripts.plot_lcfs_elongation_phase as plot_elongation_phase
import custom_plot_scripts.plot_J_r as plot_J_r
import custom_plot_scripts.plot_axuv_elongation as plot_axuv_elong
import custom_plot_scripts.plot_axuv_energy_principle as plot_energy_principle
import custom_plot_scripts.plot_dT_phase as plot_thomson
import custom_plot_scripts.plot_slope_dist_elong_li1 as plot_slope_dist_elong_li1
import custom_plot_scripts.plot_dIpl_li1 as plot_dIpl_dli1
import custom_plot_scripts.plot_crashes_ss_li as plot_crashes_ss_li
# import custom_plot_scripts.plot_inductance_gun_flux as plot_ind_gun_flux
    # non specific plotting scripts
import plot_tools.plot_over_phase as plot_phase
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
    shot_list=make_shot_list_edges(19718, 23064)
    high_time_res=False
    
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
        
    sbc = Sbc()

    if high_time_res:
        sbc.user.set("humerben")
        sbc.project.set("high_time_res_recon_crashes")
        sbc.experiment.set("pi3b")
    
    # theory plots
    # plot_J_r.plot_J_r_matplotlib(22289)
    
    # methods plots
        # plot norm and non norm signals
    # plot_axuv_crash_calcs.plot_axuv_norm_non_norm(22289, [0, 0.018])
        # plot phase
    # plot_axuv_crash_calcs.plot_axuv_phase_calc(22289, [0, 0.020])
    
    # results plots
        # multi q plot
    # plot_q_J.plot_q_J_side_by_side([21119, 22289], [0, 0.012])
    # plot_q_J.plot_q_J_side_by_side_singleshot(22289, [0, 0.012], q_lims=[0.5,4.5])
    # plot_q_J.plot_q_J_side_by_side_singleshot(22605, [0, 0.012], q_lims=[0.5,4.5])

        # elongation plots
            # elongation li1
    # plot_elongation.plot_elongation_li1(non_sustain_shots, non_sustain_gun_flux, t_lims=[0, 0.01], save=True)
    # plot_elongation.plot_elongation_li1_matplotlib([22603, 22605], [0, 0.012])

    # plot_elongation.plot_elongation_li1_matplotlib_singleshot(22605, [0, 0.01])
            # elongation phase
    # plot_elongation_phase.plot_elongation_phase_slope_matplotlib(sustain_shots_early_crash, [0, 0.012])
    # plot_elongation_phase.plot_elongation_phase(sustain_shots_early_crash, [0, 0.2])
    # plot_elongation_phase.plot_delta_elongation_phase(sustain_shots_early_crash, [0, 0.012])
    
    
    # misc plots
        # phase
    # plot_phase.plot_column_over_phase(sustain_shots, 'elong_lfs', [0, 0.012])
    # plot_phase.plot_dColumn_dPhase_over_phase(sustain_shots_early_crash, 'li1', [0, 0.012])
    # plot_phase.plot_dColumn_dPhase_over_phase([22603, 22605], 'elong_lfs', [0, 0.012], sbc=sbc)
    # plot_phase.plot_column_over_phase(sustain_shots_early_crash, 'n', t_lims=[0, 0.012], psibar=0)
    # plot_phase.plot_dColumn_dPhase_over_phase(sustain_shots_early_crash, 'n', t_lims=[0, 0.012], psibar=0)
    # plot_phase.plot_dColumn_dPhase_over_phase(sustain_shots_early_crash, 'q', t_lims=[0, 0.012], psibar=0.95)
    # plot_phase.plot_dColumn_dPhase_over_phase(sustain_shots_early_crash, 'Ipl', psibar=1, t_lims=[0, 0.012])
    # plot_phase.plot_dColumn_dPhase_over_phase(sustain_shots_early_crash, 'ParticleInventory', psibar=1, t_lims=[0, 0.012])
    # plot_phase.plot_column_over_phase(sustain_shots_early_crash, 'Ipl', psibar=1, t_lims=[0, 0.012])
    # plot_phase.plot_column_over_phase([22603, 22605], 'Ipl', psibar=1, t_lims=[0, 0.012], sbc=sbc)
    # plot_dIpl_dli1.plot_dIpl_dli1_colored_by_gun_flux(non_sustain_shots, non_sustain_gun_flux)
    
    
    # plots for poster
    poster=False
    if poster:
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
        plot_elongation.plot_elongation_li1_triple_matplotlib(non_sustain_low_gun_flux_and_formation, non_sustain_low_gun_flux_high_formation,
                                        non_sustain_high_gun_flux_and_formation, t_lims=[0, 0.012], faded_traces=True, high_res=False)
        plot_dIpl_dli1.plot_dIpl_dli1_colored_by_gun_flux_grouped(non_sustain_shots, non_sustain_gun_flux,
                                        non_sustain_low_gun_flux_and_formation, non_sustain_low_gun_flux_high_formation, non_sustain_high_gun_flux_and_formation)
                # IRE
        plot_elongation_phase.plot_elongation_phase_slope_matplotlib(sustain_shots_early_crash)
        plot_crashes_ss_li.plot_percent_chance_crash_before_time_over_ss_li(shot_list, [0.003, 0.005, 0.007, 0.009, 0.011], True)
    
    
        # thomson
    # plot_thomson.plot_dT_phase(sustain_shots_early_crash)
    
        # energy principle
    energy_principle = False
    if energy_principle:
        for i in [22289, 21441, 22605, 22744]:
            plot_energy_principle.plot_axuv_energy_principle(i, t_lims=[0, 0.012], sbc=sbc)
    
    # print(len(shot_list))
    # shot_list = filter_by_lifetime(shot_list, 0, 0.015)
    # print(len(shot_list))
    shot_list = filter_by_sustain(shot_list, 6000, 10000)
    # print(len(shot_list))
    
    
    # plot_q_leq1.plot_q_leq_1_ov_crash_timing(shots=non_sustain_shots, shots2=sustain_shots)
    # plot_axuv_elong.plot_axuv_and_elongation_dual_plot(22744, t_lims=[0, 0.012], psibar=0.95)
    
    # plot_slope_dist_elong_li1.plot_slope_dist_elong_li1(sustain_shots, [0, 0.012], shot_type="sustain")
    
    # plot_ind_gun_flux.plot_inductance_over_gun_flux(non_sustain_shots, non_sustain_gun_flux, 0.003)
    
    # plot_elongation.plot_elongation_li1_at_crashes(non_sustain_high_gun_flux_and_formation, [22744], [0, 0.012], name_extension="high_flux")
    # plot_elongation.plot_elongation_li1_at_crashes(non_sustain_low_gun_flux_high_formation, [22289], [0, 0.012], name_extension="low_flux")
    # plot_elongation.plot_elongation_li1_at_crashes([*non_sustain_high_gun_flux_and_formation, *non_sustain_low_gun_flux_high_formation], [22289], [0, 0.012], plotly=True)
    
    # plot_axuv_elong.plot_axuv_and_elongation_dual_plot(22602, t_lims=[0,0.012])
    # plot_elongation.plot_elongation_li1_matplotlib([21112, 22605])
    plot_elongation_phase.plot_q0_q95_phase_slope_matplotlib(sustain_shots_early_crash)