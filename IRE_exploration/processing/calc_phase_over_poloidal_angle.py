import GF_data_tools as gdt
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
import numpy as np



def calc_phase_over_poloidal_angle(shot_number, output_plot=False, num_timeslices=5, data_dict={}):
    q_options = {"experiment": "pi3b",
        "manifest": "default",
        "shot": int(shot_number),
        "layers": 'reconstruction/raxis_mean', 
        "nodisplay": True}
    
    data = gdt.fetch_data.run_query(q_options) 
    raxis_arr = data['waves'][0]
    times_axis = data['waves'][0].x_axis()
    
    q_options['layers'] = 'reconstruction/zaxis_mean'
    data = gdt.fetch_data.run_query(q_options)
    zaxis_arr = data['waves'][0]
    zmid_plane=2.11
    
    min_time, max_time = 10000, -10000
    for k in data_dict.keys():
        times, toroidal_angles, poly_coeffs, rsquared = data_dict[k]
        poly_func = np.poly1d(poly_coeffs)
        if min(times) < min_time:
            min_time = min(times)
        if max(times) > max_time:
            max_time = max(times)
            
    timeslices = np.linspace(min_time, max_time, num_timeslices)
    
    multitime_pol_angle = []
    multitime_tor_phase = []
    for j in range(len(timeslices)):
        r_axis = np.interp([timeslices[j]], times_axis, raxis_arr)[0]
        z_axis = np.interp([timeslices[j]], times_axis, zaxis_arr)[0]
        
        multitime_pol_angle.append([])
        multitime_tor_phase.append([])
        
        for i in range(len(data_dict)):
            key = str(list(data_dict.keys())[i])
            poly_coeffs = data_dict[key][2]
            z = int(key[1:4])/100 - zmid_plane
            r = int(key[9:12])/100
            
            r_vec = r - r_axis
            z_vec = z - z_axis

            pol_angle = np.arccos((r_vec)/(np.sqrt(r_vec**2 + z_vec**2))) * z_vec/np.abs(z_vec)
            poly_func = np.poly1d(poly_coeffs)
            tor_phase = poly_func(timeslices[j])
            
            multitime_pol_angle[j].append(pol_angle)
            multitime_tor_phase[j].append(tor_phase)
            
        min_pol = np.argmin(np.abs(multitime_pol_angle[j]))

        for i in range(len(multitime_tor_phase[j])):
            if i != min_pol:
                multitime_tor_phase[j][i] = multitime_tor_phase[j][i] - multitime_tor_phase[j][min_pol]
                if multitime_tor_phase[j][i] < 0:
                    multitime_tor_phase[j][i] += 2*np.pi
            
            if multitime_pol_angle[j][i] < -min(np.abs(multitime_pol_angle[j])):
                multitime_pol_angle[j][i] += 2*np.pi
                    
        multitime_tor_phase[j][min_pol] = 0
    
    
    colormap = get_cmap("plasma")  # You can replace 'viridis' with any Matplotlib colormap
    # Linearly spaced values between 0 and 1 for the colormap
    colors = [colormap(i / (len(timeslices) - 1)) for i in range(len(timeslices))]
    
    if output_plot:
        for j in range(len(timeslices)):   
            plt.scatter(multitime_pol_angle[j], multitime_tor_phase[j], label=f't={round(timeslices[j], 2)}ms', color=colors[j])
        plt.title(r"Relative Toroidal Phase $\Delta\phi$ Over Poloidal Angle $\theta$"+f"\nShot {shot_number}")
        plt.ylabel(r"Relative Toroidal Phase $\Delta\phi\,\mathrm{[rad]}$")
        plt.xlabel(r"Poloidal Angle $\theta\,\mathrm{[rad]}$")
        plt.legend()
        plt.savefig(f"/home/jupyter-humerben/axuv_paper/IRE_exploration/plots/{shot_number}/phase_pol_angle.png")
        plt.close()
    
    return