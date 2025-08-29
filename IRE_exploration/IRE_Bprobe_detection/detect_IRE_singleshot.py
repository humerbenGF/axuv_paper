# import standard libraries
###################################################################################################
import numpy as np
import json
import scipy.signal as signal

# import GF libraries
###################################################################################################
import GF_data_tools as gdt



def detect_IRE_singleshot(shot_number, coil_names, crash_info_dict, crash_index=0):
    # make singleshot crash info and crash times
    if str(shot_number) not in crash_info_dict.keys():
        return [], [], [], [], [], [], []
    crash_info_dict_ss = crash_info_dict[str(shot_number)]
    if type(crash_info_dict_ss) == type(str()):
        return [], [], [], [], [], [], []
    if len(crash_info_dict_ss['times']) == 0:
        return [], [], [], [], [], [], []
    crash_times = np.sort([crash_info_dict_ss['times'][i]*1000 for i in range(1)]) # range(len(crash_info_dict_ss['times']))])
    if len(crash_info_dict_ss['times']) > crash_index + 1:
        next_crash_start = crash_info_dict_ss['pre_crash_times'][crash_index+1]*1000
    else:
        next_crash_start = 100000000
    
    # initialize constant settings
    dt = 1.5 # ms
    threshold = 10 # *sigma
    
    # initialize lists
    shots_singleshot, crash_t_singleshot, probe_z_singleshot, probe_r_singleshot, omega_singleshot, omega_R2_singleshot, num_points_singleshot = [], [], [], [], [], [], []
    
    for crash_time in [crash_times[crash_index]]:
        for coil_name in coil_names:
            try:
                times = []
                times, toroidal_angles, poly_coeffs, rsquared = calculate_toroidal_angle_singlecoil(shot_number, crash_time, next_crash_start, coil_name, dt, threshold)
                
                shots_singleshot.append(shot_number)
                crash_t_singleshot.append(crash_time)
                probe_z_singleshot.append(int(coil_name[1:4]))
                probe_r_singleshot.append(int(coil_name[9:12]))
                omega_singleshot.append(poly_coeffs[0]*1000)
                omega_R2_singleshot.append(rsquared)
                num_points_singleshot.append(len(times))
            except:
                print(f"Shot {shot_number} was not successfully processed")
    
    return shots_singleshot, crash_t_singleshot, probe_z_singleshot, probe_r_singleshot, omega_singleshot, omega_R2_singleshot, num_points_singleshot

def calculate_toroidal_angle_singlecoil(shot_number, crash_time, next_crash_start_time, coil_name, dt, threshold):
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
        x_std, w_std = trim_xy_by_range(t, smoothed_wave, crash_time-dt-0.25, min(crash_time-dt,next_crash_start_time))
        std = np.std(w_std)
        if threshold == 0:
            threshold = 10
        peaks = signal.find_peaks(w_trimmed, prominence=threshold*std)[0]
        
        gap=False
        for p, peak in enumerate(peaks):
            if p > 0:
                if (x_trimmed[peaks[p]] - x_trimmed[peaks[p-1]]) > 0.5:
                    gap=True
            times.append(x_trimmed[peak]) if not gap else None
            toroidal_angles.append(int(w.meta['wave_label'][13:16])*2*np.pi/360) if not gap else None
        
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