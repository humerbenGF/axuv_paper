# import standard libraries
###################################################################################################
import matplotlib.pyplot as plt
import numpy as np

# import sandbox client
###################################################################################################
from gf_sandbox_client.sbc import Sbc

# import personal files
###################################################################################################
from processing_tools.energy_principle import calculate_energy_principle_singleshot as calc_energy_principle


def plot_axuv_energy_principle(shot, sensor_number=21, t_lims=[0, 1000], sbc=Sbc()):
    t_min, t_max = t_lims
    
    df_energy_principle = calc_energy_principle(shot, sbc)
    
    sbc=Sbc()
    sbc.experiment.set("pi3b")
    # get axuv data
    axuv_df = Sbc().get(experiment='pi3b', asset="axuv_amps", shot=shot, criteria=f"sensor_number = {sensor_number}")
    axuv_df = axuv_df[(axuv_df.t >= t_min) & (axuv_df.t <=t_max)]
    axuv_t = np.array(axuv_df.t)
    axuv_A = np.array(axuv_df.A)
    axuv_t_sorting_indices = np.argsort(axuv_t)
    axuv_t = np.array([axuv_t[i] for i in axuv_t_sorting_indices])
    axuv_A = np.array([axuv_A[i] for i in axuv_t_sorting_indices])
    
    # Create the figure and the primary axis
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot on the left y-axis
    line1, = ax1.plot(axuv_t, axuv_A, zorder=0, alpha=0.6, label='AXUV Current')
    ax1.set_ylabel(r'$I_{photodiode}$ [A]')

    # Create a secondary y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    line2, = ax2.plot(df_energy_principle.t, df_energy_principle.dW_dt, zorder=1, color='r', label=r'$\Delta W / \Delta t$')
    ax2.set_ylabel(r'$\Delta W / \Delta t$')


    # Add x-axis label and title
    ax1.set_xlabel('time [s]')
    ax1.set_title(f'AXUV and Time Derivative of Energy Principle over Time\nShot {shot}')
    
    # Combine legends from both axes
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.tight_layout()
    plt.savefig(f"/home/jupyter-humerben/axuv_paper/plot_outputs/energy_principle/axuv_energy_principle_{shot}.png")