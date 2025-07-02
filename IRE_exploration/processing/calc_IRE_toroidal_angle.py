import GF_data_tools as gdt
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import json
import os

from processing.calc_phase_over_poloidal_angle import calc_phase_over_poloidal_angle

def calculate_toroidal_angle_multicoil(shot_number, coil_names, crash_index=0, dt=1.5, threshold=0, output_plot=True, diagnostic_plot=False, plot_phase_pol_angle=False):
    if not os.path.isdir(f"/home/jupyter-humerben/axuv_paper/IRE_exploration/plots/{shot_number}"):
        os.mkdir(f"/home/jupyter-humerben/axuv_paper/IRE_exploration/plots/{shot_number}")
    with open('/home/jupyter-humerben/axuv_paper/IRE_exploration/data/crash_info_with_hardware_error.json', 'r') as f:
        crash_info_dict = json.load(f)
    crash_info_dict_ss = crash_info_dict[str(shot_number)]
    crash_time = crash_info_dict_ss['times'][crash_index]*1000
    
    tab_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    data_dict = {}
    for coil_name in coil_names:
        times, toroidal_angles, poly_coeffs, rsquared = calculate_toroidal_angle_singlecoil(shot_number, crash_time, coil_name, dt, threshold, diagnostic_plot)
        data_dict[coil_name] = times, toroidal_angles, poly_coeffs, rsquared
    
    if output_plot:
        i=0
        for coil_name in coil_names:
            times, toroidal_angles, poly_coeffs, rsquared = data_dict[coil_name]
            poly_func = np.poly1d(poly_coeffs)
            plt.plot(times, poly_func(times), alpha=0.5, color=tab_colors[i], label=r"Linear Fit: $\mathrm{R^2}$="+f"{str(rsquared)[:5]}", zorder=1)
            plt.scatter(times, toroidal_angles, marker='x', color=tab_colors[i], label=f"{coil_name}", zorder=2)
            i += 1
        plt.title(r"$\phi(t)$ for an IRE"+f"\nShot {shot_number}")
        plt.ylabel(r"$\phi\,\mathrm{[rad]}$")
        plt.xlabel(r"$t\,\mathrm{[ms]}$")
        plt.legend()
        plt.savefig(f"/home/jupyter-humerben/axuv_paper/IRE_exploration/plots/{shot_number}/phi_t.png")
        plt.close()
        
    if plot_phase_pol_angle:
        calc_phase_over_poloidal_angle(shot_number, data_dict=data_dict)
    
    return

def calculate_toroidal_angle_singlecoil(shot_number, crash_time, coil_name, dt, threshold, diagnostic_plot):
    q_options = {"experiment": "pi3b",
             "manifest": "default",
             "shot": int(shot_number),
             "layers": 'mirnov/' + coil_name + '/poloidal/cross_talk_subtracted', 
             "nodisplay": True}
    
    data = gdt.fetch_data.run_query(q_options) 
    waves = data['waves']

    times, toroidal_angles = [], []

    cutoff = 2e4
    for w in waves:
        smoothed_wave = gdt.wave_utils.low_pass_filter(w, cutoff)
        waves_list = gdt.plotting.collect_waves(data)[0]
        scalars = data['scalars']
        t_lims = gdt.plotting.calc_xlims(waves_list, scalars)
        t = np.linspace(t_lims[0]*1000, t_lims[1]*1000, len(waves[0]))
        x_trimmed, w_trimmed = trim_xy_by_range(t, smoothed_wave, crash_time-dt, crash_time+dt)
        x_std, w_std = trim_xy_by_range(t, smoothed_wave, crash_time-dt-0.25, crash_time-dt)
        std = np.std(w_std)
        if threshold == 0:
            threshold = 10*std
        peaks = signal.find_peaks(w_trimmed, prominence=threshold)[0]
        
        for peak in peaks:
            times.append(x_trimmed[peak])
            toroidal_angles.append(int(w.meta['wave_label'][13:16])*2*np.pi/360)
            
        
        plt.plot(x_trimmed, w_trimmed, label=w.meta['wave_label'][7:21]) if diagnostic_plot else None
    plt.legend() if diagnostic_plot else None
    plt.title(r"$B_{\mathrm{pol}}(t)$ at Different Values of Toroidal Angle"+f"\nShot {shot_number} // Coil {coil_name}") if diagnostic_plot else None
    plt.xlabel(r"$t\,\mathrm{[ms]}$") if diagnostic_plot else None
    plt.ylabel(r"$B_{\mathrm{pol}}\,\mathrm{[T]}$") if diagnostic_plot else None
    plt.savefig(f"/home/jupyter-humerben/axuv_paper/IRE_exploration/plots/{shot_number}/{coil_name}.png") if diagnostic_plot else None
    plt.close() if diagnostic_plot else None
        
    t_inds = np.argsort(times)
    times = [times[i] for i in t_inds]
    toroidal_angles = [toroidal_angles[i] for i in t_inds]
    
    toroidal_angles_new = [toroidal_angles[0]]
    num_rotations = 0
    for i in range(len(toroidal_angles)-1):
        if toroidal_angles[i] < toroidal_angles[i+1]:
            num_rotations += 1
        toroidal_angles_new.append((toroidal_angles[i+1]-2*np.pi*num_rotations))
    
    toroidal_angles = toroidal_angles_new

    coeffs = np.polyfit(times, toroidal_angles, 1)
    poly_func = np.poly1d(coeffs)
    rsquared = compute_r_squared(times, toroidal_angles, poly_func)

    print(f"angular velocity of {coil_name}: {int(coeffs[0]*1000)} [rad/s]")
    
    return times, toroidal_angles, coeffs, rsquared



        
def compute_r_squared(x, y, poly):
    y_fit = poly(x)
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot
        
def trim_xy_by_range(x, y, x0, x1):
    """
    Trim arrays x and y such that only values where x0 < x < x1 are retained.

    Parameters:
    - x: 1D array-like of x-values
    - y: 1D array-like of y-values (same shape as x)
    - x0: lower bound (exclusive)
    - x1: upper bound (exclusive)

    Returns:
    - x_trimmed, y_trimmed: filtered arrays
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    mask = (x > x0) & (x < x1)
    return x[mask], y[mask]