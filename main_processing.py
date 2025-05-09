# import standard libraries
###################################################################################################
import pandas as pd
import numpy as np

# import sandbox client
###################################################################################################
from gf_sandbox_client.sbc import Sbc

# import personal scripts
###################################################################################################
    # processing
from processing.generate_percentage_crashes_with_dtm import get_percentage_double_surface_crossings_leading_to_crash
import processing.first_crash_stats_for_shot_grouping as first_crash_stats

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
    high_time_res=True
    
    non_sustain = pd.read_csv("/home/jupyter-humerben/axuv_paper/datasets/non_sustain_dataset.csv")
    sustain = pd.read_csv("/home/jupyter-humerben/axuv_paper/datasets/sustain_dataset.csv")
    
    non_sustain_shots = list(np.array(non_sustain['shot']))
    non_sustain_gun_flux = list(np.array(non_sustain['gun_flux_mWb']))
    non_sustain_high_gun_flux_and_formation = [22729,22731,22732,22733,22734,22735,22736,22737,22743,22744,22745,22754,22755]
    non_sustain_low_gun_flux_high_formation = [22285,22286,22287,22288,22289,22290,22622,22625,22626,22628,22629,22630,22631]
    non_sustain_low_gun_flux_and_formation = [22293,22294,22295,22296,22297,22298,22634,22637,22645,22646]
    sustain_shots = list(np.array(sustain['shot']))
    sustain_shots_small_crash = list(np.array(sustain[sustain.note == 'small_crash'].shot))
    sustain_shots_early_crash = list(np.array(sustain[sustain.note == 'early_or_normal_crash'].shot))
    high_res_shots = [20335, 20346, 21282, 21284, 21314, 21441, 22289, 22603, 22605, 22708, 22885]
        
    sbc = Sbc()

    if high_time_res:
        sbc.user.set("humerben")
        sbc.project.set("high_time_res_recon_crashes")
        sbc.experiment.set("pi3b")
        
    # get_percentage_double_surface_crossings_leading_to_crash(non_sustain_shots)
    # generate_first_crash_stats_for_shot_list(non_sustain_high_gun_flux_and_formation)
    # generate_first_crash_stats_for_shot_list(non_sustain_low_gun_flux_high_formation)
    
    first_crash_stats.generate_first_crash_time_stats_for_multiple_groups(
        [non_sustain_low_gun_flux_and_formation, non_sustain_low_gun_flux_high_formation, non_sustain_high_gun_flux_and_formation], 
        [r"$V_{form}=22kV$, $N_{caps}<=32$"+"\n"+"$\psi_{gun}<100mWb$", r"$V_{form}=22kV$, $N_{caps}=48$"+"\n"+"$\psi_{gun}<100mWb$", 
         r"$V_{form}>=23kV$, $N_{caps}=48$"+"\n"+"$\psi_{gun}>130mWb$"], plots=True)