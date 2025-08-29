import pandas as pd

import GF_data_tools as gdt
import matplotlib.pyplot as plt
import numpy as np
from IRE_Bprobe_detection import detect_IREs, load_IRE_data

from processing.calc_IRE_toroidal_angle import calculate_toroidal_angle_multicoil as calc_toroidal_angle

from plotting.omega_t_scatter import plot_omega_t_scatter as plot_w_t




def make_shot_list_edges(min_shot, max_shot):
    L = []
    while min_shot <= max_shot:
        L.append(min_shot)
        min_shot += 1

    return L


if __name__ == '__main__':
    richmond_shots=make_shot_list_edges(19718, 23064)
    
    shot=21441
    coils_list = ['z211_t*_r100']
    coils_list = ['z211_t*_r100', 'z291_t*_r060', 'z161_t*_r087', 'z261_t*_r087']

    non_sustain = pd.read_csv("/home/jupyter-humerben/axuv_paper/datasets/non_sustain_dataset.csv")
    sustain = pd.read_csv("/home/jupyter-humerben/axuv_paper/datasets/sustain_dataset.csv")
    
    non_sustain_shots = list(np.array(non_sustain['shot']))
    sustain_shots = list(np.array(sustain['shot']))

    shots = list(np.append(non_sustain_shots, sustain_shots))
    shots = list(np.array(sustain[sustain.note == 'early_or_normal_crash'].shot))

    detect_IREs(shots)
    # df = load_IRE_data()
    # plot_w_t()
    
    # calc_toroidal_angle(21112, coil_names=['z211_t*_r100', 'z291_t*_r060', 'z161_t*_r087', 'z261_t*_r087'], diagnostic_plot=True, output_plot=True)
    # df = load_IRE_data()
    # print(df[df.shot==21240])