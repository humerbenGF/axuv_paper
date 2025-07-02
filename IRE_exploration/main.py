import GF_data_tools as gdt
import matplotlib.pyplot as plt
import numpy as np

from processing.calc_IRE_toroidal_angle import calculate_toroidal_angle_multicoil as calc_toroidal_angle

if __name__ == '__main__':
    shot=21441
    coils_list = ['z211_t*_r100']
    # coils_list = ['z215_t*_r008']
    coils_list = ['z211_t*_r100', 'z291_t*_r060', 'z161_t*_r087', 'z261_t*_r087']
    calc_toroidal_angle(shot, coils_list, diagnostic_plot=True, crash_index=0)
