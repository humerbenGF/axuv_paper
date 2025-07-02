# import standard libraries
###################################################################################################
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter


# import sandbox client
###################################################################################################
from gf_sandbox_client.sbc import Sbc

# import personal files
###################################################################################################
from data_loaders.load_crash_data import load_crash_data


def plot_axuv_phase_calc(shot, t_lims=[0, 1000], sbc=Sbc()):
    # unpack limits
    t_min, t_max = t_lims
    
    # prepare and unpack crash data
    crash_df = load_crash_data(shot)
    df_for_lifetime = sbc.get(asset="reconstruction_profile", columns=['q'], shot=shot, criteria='statistic = mean', experiment='pi3b')
    lifetime = max(np.array(df_for_lifetime.t))
    crash_times = np.array([i*1000 for i in np.array(crash_df.t)])
    phase_integers = np.append(np.append([t_min*1000], crash_times), [lifetime*1000])
    t_max=lifetime
    
    # set up SBC and load data
        # load data
    sbc.experiment.set("pi3b")
    df_axuv = sbc.get(asset="axuv_amps", columns=['t', 'A'], shot=shot)
        # select sensor
    df_axuv = df_axuv[df_axuv.sensor_number == 21]
        # trim times
    df_axuv = df_axuv[df_axuv['t'] >= t_min]
    df_axuv = df_axuv[df_axuv['t'] <= t_max]
    
    # unpack and reorder arrays
        # unpack arrays
    A = np.array(df_axuv['A'])
    t = np.array(df_axuv['t'])
        # get indices for sorting
    t_indices = np.argsort(t)
        # sort arrays
    A_raw = np.array([A[i]*10**9 for i in t_indices])
    t = np.array([t[i]*1000 for i in t_indices])
        
    # filters and pre-processing
    A_lp = lowpass_fft(A_raw, cutoff=100000, fs=1/abs(t[1]-t[0]))
    A_sg = savgol_filter(A_lp, int(1/abs(t[1]-t[0])), 3)
    
    
    # plot
        # main lines
    plt.plot(t, A_raw, alpha=0.5, linewidth=0.5, label="Unfiltered Soft X-ray Data")
    plt.plot(t, A_sg, alpha=0.9, label="Filtered Soft X-ray Data")
        # vertical lines
    plt.vlines(phase_integers, min(A_sg), max(A_sg), color='k', label="Integer Values of Crash Phase")
    
    # labels/titles
        # title
    plt.title("Soft X-ray Photodiode Current over Time with Crash Phase")
        # axes
    plt.ylabel(r'$I_{photodiode}$ [nA]')
    plt.xlabel('time [ms]')
        # legend
    plt.tight_layout()
    plt.legend(fontsize=10, columnspacing=0.5, handletextpad=0.3, handlelength=1.5, loc='upper center', bbox_to_anchor=(0.5, -0.25), frameon=False)
    plt.tight_layout()
    
    # save figure
    plt.savefig(f"/home/jupyter-humerben/axuv_paper/plot_outputs/phase_integers_{shot}.png")
    
    return
    
    


def plot_axuv_norm_non_norm(shot, t_lims=[0, 1000], sbc=Sbc()):
    # unpack limits
    t_min, t_max = t_lims
    
    # prepare and unpack crash data
    crash_df = load_crash_data(shot)
    
    # set up SBC and load data
        # load data
    sbc.experiment.set("pi3b")
    df_axuv = sbc.get(asset="axuv_amps", columns=['t', 'A'], shot=shot)
        # select sensor
    df_axuv = df_axuv[df_axuv.sensor_number == 21]
        # trim times
    df_axuv = df_axuv[df_axuv['t'] >= t_min]
    df_axuv = df_axuv[df_axuv['t'] <= t_max]
    
    # unpack and reorder arrays
        # unpack arrays
    A = np.array(df_axuv['A'])
    t = np.array(df_axuv['t'])
        # get indices for sorting
    t_indices = np.argsort(t)
        # sort arrays
    A_raw = np.array([A[i]*10**9 for i in t_indices])
    t = np.array([t[i]*1000 for i in t_indices])
        
    # filters and pre-processing
    A_lp = lowpass_fft(A_raw, cutoff=100000, fs=1/abs(t[1]-t[0]))
    A_sg = savgol_filter(A_lp, int(1/abs(t[1]-t[0])), 3)
    dA = np.gradient(A_sg, abs(t[1] - t[0]))
    dA_sg = savgol_filter(dA, int(1/abs(t[1]-t[0])), 3)
    ddA = np.gradient(dA_sg, abs(t[1]-t[0]))
    ddA_sg = savgol_filter(ddA, int(1/abs(t[1]-t[0])), 3)
    
    # get normalized values for plotting
    A_norm = normalize(A_sg)
    dA_norm = normalize(dA_sg)
    ddA_norm = normalize(ddA_sg)
    
    
    # make plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=400)
    
    font_min=12

    # Ensure axs is always a 2D array
    axs = np.array([axs])  # Convert 1D to 2D array for consistent indexing
    
    # add in background crash info
    for i in range(len(crash_df.t)):
        crash_boundaries = [crash_df.pre_crash_t[i]*1000, crash_df.t[i]*1000, crash_df.post_crash_t[i]*1000]
        axs[0, 0].vlines([crash_boundaries[0], crash_boundaries[2]], min(A_sg), max(A_sg), linewidth=0.5, color='r', alpha=0.8)
        axs[0, 0].fill_between(x=[crash_boundaries[0], crash_boundaries[2]], y1=min(A_sg), y2=max(A_sg), color='r', alpha=0.2)
        axs[0, 0].vlines([crash_boundaries[1]], min(A_sg), max(A_sg), color='k', zorder=10)
    axs[0, 0].vlines([t_lims[0]], min(A_sg), max(A_sg), color='k', zorder=10)
    axs[0, 0].vlines([t_lims[1]*1000], min(A_sg), max(A_sg), color='k', zorder=10)
        
        
    # make false labels for non-norm crash data
    axs[0, 0].fill_between(x=[0], y1=[0], color='r', alpha=0.2, label='Soft X-ray Crash Region')
    axs[0, 0].vlines([], min(A_sg), max(A_sg), color='k', label='Integer Values of Crash Phase')
    axs[0, 0].scatter([],[], marker='x', color='k')

    
        
    # make non normalized plot
        # plot Primary traces
    axs[0, 0].plot(t, A_raw, alpha=0.5, linewidth=0.5, label="Unfiltered Soft X-ray Data")
    axs[0, 0].plot(t, A_sg, alpha=0.9, label="Filtered Soft X-ray Data")
        # set labels and make legend
    axs[0, 0].set_ylabel(r"$I_{photodiode}$ [nA]", fontsize=font_min+2)
    axs[0, 0].set_xlabel(r"time [ms]", fontsize=font_min+2)
    
    # make normalized plot
        # plot primary traces
    axs[0, 1].plot(t, A_norm, label="Filtered Soft X-ray Data")
    axs[0, 1].plot(t, dA_norm, label="Filtered Soft X-ray First Time Derivative Data")
    axs[0, 1].plot(t, ddA_norm, label="Filtered Soft X-ray Second Time Derivative Data")
        # make legend
    axs[0, 1].set_ylabel(r"Normalized Soft X-ray Data", fontsize=font_min+2)
    axs[0, 1].set_xlabel(r"time [ms]", fontsize=font_min+2)
    
    # add foreground crash info
    for i in range(len(crash_df.t)):
        crash_boundaries = [crash_df.pre_crash_t[i]*1000, crash_df.t[i]*1000, crash_df.post_crash_t[i]*1000]
        axs[0, 1].vlines([crash_boundaries[0], crash_boundaries[2]], -1, 1, linewidth=0.5, color='r', alpha=0.8)
        axs[0, 1].vlines([crash_boundaries[1]], -1, 1, linewidth=0.5, color='k', alpha=0.8)
        
        
    axs[0, 1].vlines([crash_boundaries[0], crash_boundaries[2]], -1, 1, linewidth=0.5, color='r', alpha=0.8, label="Crash Boundaries")
    axs[0, 1].vlines([crash_boundaries[1]], -1, 1, linewidth=0.5, color='k', alpha=0.8, label="Peak of Soft X-ray Crash")
    
    # misc settings for both axes
        # grid
    axs[0, 0].grid(True, alpha=0.4)
    axs[0, 1].grid(True, alpha=0.4)
        # tick params
    axs[0, 0].tick_params(axis='both', labelsize=font_min)
    axs[0, 1].tick_params(axis='both', labelsize=font_min)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)
    
    # set legends for both plots
    axs[0, 0].legend(fontsize=font_min, columnspacing=0.5, handletextpad=0.3, handlelength=1.5, loc='upper center', bbox_to_anchor=(0.5, -0.125), frameon=False)
    axs[0, 1].legend(fontsize=font_min, columnspacing=0.5, handletextpad=0.3, handlelength=1.5, loc='upper center', bbox_to_anchor=(0.5, -0.125), frameon=False)
    
    # Add captions below subplots (adjust y position as needed)
    # axs[0, 0].text(0.5, -0.90, "a) Soft X-ray Data", ha='center', va='top', transform=axs[0, 0].transAxes, fontsize=12)
    # axs[0, 1].text(0.5, -0.90, "b) Normalized & Derivative Soft X-ray Data", ha='center', va='top', transform=axs[0, 1].transAxes, fontsize=12)
    
    # save plot
    plt.savefig(f"/home/jupyter-humerben/axuv_paper/plot_outputs/crash_detection/axuv_norm_non-norm_{shot}.png")

    
    
    return




def plot_axuv_norm_non_norm_combined(shot, t_lims=[0, 1000], sbc=Sbc()):
    # unpack limits
    t_min, t_max = t_lims
    
    # prepare and unpack crash data
    crash_df = load_crash_data(shot)
    
    # set up SBC and load data
        # load data
    sbc.experiment.set("pi3b")
    df_axuv = sbc.get(asset="axuv_amps", columns=['t', 'A'], shot=shot)
        # select sensor
    df_axuv = df_axuv[df_axuv.sensor_number == 21]
        # trim times
    df_axuv = df_axuv[df_axuv['t'] >= t_min]
    df_axuv = df_axuv[df_axuv['t'] <= t_max]
    
    # unpack and reorder arrays
        # unpack arrays
    A = np.array(df_axuv['A'])
    t = np.array(df_axuv['t'])
        # get indices for sorting
    t_indices = np.argsort(t)
        # sort arrays
    A_raw = np.array([A[i]*10**9 for i in t_indices])
    t = np.array([t[i]*1000 for i in t_indices])
        
    # filters and pre-processing
    A_lp = lowpass_fft(A_raw, cutoff=100000, fs=1/abs(t[1]-t[0]))
    A_sg = savgol_filter(A_lp, int(1/abs(t[1]-t[0])), 3)
    dA = np.gradient(A_sg, abs(t[1] - t[0]))
    dA_sg = savgol_filter(dA, int(1/abs(t[1]-t[0])), 3)
    ddA = np.gradient(dA_sg, abs(t[1]-t[0]))
    ddA_sg = savgol_filter(ddA, int(1/abs(t[1]-t[0])), 3)
    
    # get normalized values for plotting
    A_norm = normalize(A_sg)
    A_sg = A_norm
    dA_norm = normalize(dA_sg)
    ddA_norm = normalize(ddA_sg)
    
    
    # make plot
    fig = plt.figure(figsize=(5, 5), dpi=400)
    
    font_min=12

    # Ensure axs is always a 2D array
    
    # add in background crash info
    for i in range(len(crash_df.t)):
        crash_boundaries = [crash_df.pre_crash_t[i]*1000, crash_df.t[i]*1000, crash_df.post_crash_t[i]*1000]
        plt.vlines([crash_boundaries[0], crash_boundaries[2]], -1, 1, linewidth=0.5, color='r', alpha=0.8)
        plt.fill_between(x=[crash_boundaries[0], crash_boundaries[2]], y1=-1, y2=1, color='r', alpha=0.1)
        plt.vlines([crash_boundaries[1]], -1, 1, color='k', zorder=10)
    plt.vlines([t_lims[0]], -1, 1, color='k', zorder=10)
    plt.vlines([t_lims[1]*1000], -1, 1, color='k', zorder=10)
        
        
    # make false labels for non-norm crash data
    plt.fill_between(x=[0], y1=[0], color='r', alpha=0.2, label='Soft X-ray Crash Region')
    plt.vlines([], -1, 1, color='k', label='Integer Values of Crash Phase')
    
    # make normalized plot
        # plot primary traces
    plt.plot(t, A_norm, label="Filtered Soft X-ray Data")
    plt.plot(t, dA_norm, label="Filtered Soft X-ray First Time Derivative Data")
    plt.plot(t, ddA_norm, label="Filtered Soft X-ray Second Time Derivative Data")
        # make legend
    plt.ylabel(r"Normalized Soft X-ray Data", fontsize=font_min+2)
    plt.xlabel(r"time [ms]", fontsize=font_min+2)
    
    # misc settings for both axes
        # grid
    plt.grid(True, alpha=0.4)
        # tick params
    plt.tick_params(axis='both', labelsize=font_min)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)
    
    # set legends for both plots
    plt.legend(fontsize=font_min, columnspacing=0.5, handletextpad=0.3, handlelength=1.5, loc='upper center', bbox_to_anchor=(0.4, -0.125), frameon=False)
    
    
    # save plot
    plt.savefig(f"/home/jupyter-humerben/axuv_paper/plot_outputs/crash_detection/axuv_norm_non-norm_combined_{shot}.png")

    
    
    return


    
    
def lowpass_fft(data, cutoff, fs=1.0):
    """
    Apply a low-pass filter using FFT by zeroing out high frequencies.

    Parameters:
        data (array-like): Input signal (1D NumPy array).
        cutoff (float): Cutoff frequency in Hz (removes all frequencies above this).
        fs (float): Sampling frequency in Hz (default = 1.0).

    Returns:
        np.ndarray: Low-pass filtered signal.
    """
    N = len(data)  
    freqs = np.fft.fftfreq(N, d=1/fs)  # Frequency components
    fft_vals = np.fft.fft(data)  # Compute FFT
    
    # Zero out frequencies above the cutoff
    fft_vals[np.abs(freqs) > cutoff] = 0  

    return np.fft.ifft(fft_vals).real  # Inverse FFT to return to time domain


def normalize(arr):
    min_val, max_val = np.min(arr), np.max(arr)
    norm_coeff = max(abs(min_val), max_val)
    
    return np.array([i/norm_coeff for i in arr])
    
    
    
if __name__ == '__main__':
    plot_axuv_norm_non_norm(22289, [0, 0.018])