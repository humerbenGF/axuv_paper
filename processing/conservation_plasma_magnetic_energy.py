# import standard libraries
###################################################################################################
import pandas as pd
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt

# import sandbox client
###################################################################################################
from gf_sandbox_client.sbc import Sbc

# import sandbox client
###################################################################################################
from data_loaders.load_crash_data import load_crash_data as load_crash


def calculate_plasma_magnetic_energy_singleshot(shot, sbc=Sbc(), maj_rad=0.65):
    # standard data
    df = sbc.get(user='humerben', experiment='pi3b', asset='reconstruction', shot=shot, columns=['li1', 'Ipl'], criteria='statistic = mean or statistic = sigma')
    df_err = df[df.statistic == 'sigma']
    df = df[df.statistic == 'mean']
    times = np.sort(np.unique(df.t))
    
    # axuv data
    df_axuv = Sbc().get(user='humerben', experiment='pi3b', asset='axuv_amps', shot=shot)
    df_axuv = df_axuv[(df_axuv.sensor_number == 21) & (df_axuv.t < max(times))]
    axuv_times_sorting = np.argsort(df_axuv.t)
    axuv_A = np.array(df_axuv.A)
    axuv_A = [10**9*axuv_A[i] for i in axuv_times_sorting]
    axuv_t = np.array(df_axuv.t)
    axuv_t = [10**3*axuv_t[i] for i in axuv_times_sorting]
        
    # crash data
    crash_df = load_crash(shot)
    crash_df = crash_df[crash_df.t < max(times)]
    crash_times = [t*1000 for t in np.sort(np.unique(crash_df.t))]
    crash_beginning_times = [t*1000 for t in np.sort(np.unique(crash_df.pre_crash_t))]
    crash_end_times = [t*1000 for t in np.sort(np.unique(crash_df.post_crash_t))]


    Upl_array = []
    Upl_error_array = []
    Ipl_array = []
    Ipl_error_array = []
    times_plotting = []
    for t in range(len(times)):
        # load data
        li1 = df[df.t == times[t]].li1.iloc[0]
        li1_error = df_err[df_err.t == times[t]].li1.iloc[0]
        Ipl = df[df.t == times[t]].Ipl.iloc[0]
        Ipl_error = df_err[df_err.t == times[t]].Ipl.iloc[0]
        
        # calculate potentials
        Upl = 1/4 * (const.mu_0 * li1 * maj_rad) * (Ipl)**2
        Upl_error = Upl * np.sqrt((li1_error/li1)**2 + (2*Ipl_error/Ipl)**2)
        
        # append to arrays
        Ipl_array.append(Ipl/1000)
        Ipl_error_array.append(Ipl_error/1000)
        Upl_array.append(Upl/1000)
        Upl_error_array.append(Upl_error/1000)
        times_plotting.append(times[t]*1000)
        
    times_plotting = np.array(times_plotting)
    
    
    # Create figure and primary axis
    fig, [ax1,ax3] = plt.subplots(1,2, figsize=(12,5))

    # Plot on primary y-axis
        # plot mean
    ax1.plot(times_plotting, Ipl_array, color='blue', label='Plasma Current')
        # plot error
    y1 = np.array([Ipl_array[i] - Ipl_error_array[i] for i in range(len(Ipl_array))])
    y2 = np.array([Ipl_array[i] + Ipl_error_array[i] for i in range(len(Ipl_array))])
    ax1.fill_between(x=times_plotting, y1=y1, y2=y2, color='blue', alpha=0.2)
    ax1.set_ylabel(r'$Ipl$ [kA]', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(ymin=0)

    # Create secondary y-axis that shares the same x-axis
    ax2 = ax1.twinx()
        # plot mean
    ax2.plot(times_plotting, Upl_array, color='red', label='Internal Plasma Magnetic Energy')
        # plot error
    y1 = [Upl_array[i] - Upl_error_array[i] for i in range(len(Upl_array))]
    y2 = [Upl_array[i] + Upl_error_array[i] for i in range(len(Upl_array))]
    ax2.fill_between(x=times_plotting, y1=y1, y2=y2, color='red', alpha=0.2)
    ax2.set_ylabel(r'$Upl$ [kJ]', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(ymin=0)

    # second panel in the plot
    ax3.plot(axuv_t, axuv_A)
    ax3.set_ylabel(r"$I_{photodiode}$ [nA]")
    ax3.set_xlabel("Time [ms]")


    # vertical lines for crashes
    ax2.vlines(crash_times, 0, max(y2), color='k', label='Crash Times')
    ax3.vlines(crash_times, 0, max(axuv_A), color='k', label='Crash Times')
    for i in range(len(crash_times)):
        ax2.fill_betweenx([0, max(y2)], crash_beginning_times[i], crash_end_times[i], color='gray', alpha=0.3)
        ax3.fill_betweenx([0, max(axuv_A)], crash_beginning_times[i], crash_end_times[i], color='gray', alpha=0.3)
        





    # Common x-axis label
    ax1.set_xlabel('time [ms]')
    

    # Optional: add legends
    # ax1.legend(loc='upper left')
    ax2.legend(loc='lower left')
    # fig.legend()

    fig.suptitle(f'Magnetic Energy Conservation\nShot {shot}')
    fig.tight_layout()
    
    fig.savefig("test.png")
    
    
    
    
    
    
    return



if __name__ == '__main__':
    sbc = Sbc()
    sbc.project.set("high_time_res_recon_crashes")
    calculate_plasma_magnetic_energy_singleshot(22744, sbc)