import GF_data_tools as gdt
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import json
import os
import data_loaders.load_crash_data as load_crash


def plot_magnetic_data_singleplane(shot_number, coil_name='z211_t*_r100', diagnostic_plot=True):
    if not os.path.isdir(f"/home/jupyter-humerben/axuv_paper/IRE_exploration/plots/{shot_number}"):
        os.mkdir(f"/home/jupyter-humerben/axuv_paper/IRE_exploration/plots/{shot_number}")
    
    crash_data = load_crash.load_crash_data([shot_number])
    first_crash_time = min(crash_data.pre_crash_t)
    first_crash_end_time = min(crash_data.post_crash_t)
    
    q_options = {"experiment": "pi3b",
             "manifest": "default",
             "shot": int(shot_number),
             "layers": 'mirnov/' + coil_name + '/poloidal/cross_talk_subtracted', 
             "nodisplay": True}
    
    data = gdt.fetch_data.run_query(q_options) 
    waves = data['waves']

    times, toroidal_angles = [], []

    cutoff = 2e4
    wave_max, wave_min = 0, 1000
    for w in waves:
        smoothed_wave = gdt.wave_utils.low_pass_filter(w, cutoff)
        waves_list = gdt.plotting.collect_waves(data)[0]
        scalars = data['scalars']
        t_lims = gdt.plotting.calc_xlims(waves_list, scalars)
        t = np.linspace(t_lims[0]*1000, t_lims[1]*1000, len(waves[0]))
        wave_max = max(wave_max, max(smoothed_wave))
        wave_min = min(wave_min, min(smoothed_wave))
        
        plt.plot(t, smoothed_wave, label=w.meta['wave_label'][7:21]) if diagnostic_plot else None
    plt.legend() if diagnostic_plot else None
    plt.title(r"$B_{\mathrm{pol}}(t)$ at Different Values of Toroidal Angle"+f"\nShot {shot_number} // Coil {coil_name}") if diagnostic_plot else None
    plt.xlabel(r"$t\,\mathrm{[ms]}$") if diagnostic_plot else None
    plt.ylabel(r"$B_{\mathrm{pol}}\,\mathrm{[T]}$") if diagnostic_plot else None
    plt.xlim(first_crash_time*1000-2,first_crash_time*1000+2)
    # plt.ylim(wave_max/2, wave_max)
    plt.vlines([first_crash_time*1000, first_crash_end_time*1000], wave_min, wave_max, 'r')
    plt.savefig(f"/home/jupyter-humerben/axuv_paper/IRE_exploration/plots/{shot_number}/singleplane_{coil_name}.png") if diagnostic_plot else None
    plt.close() if diagnostic_plot else None
    
    return


if __name__ == '__main__':
    plot_magnetic_data_singleplane(22756)