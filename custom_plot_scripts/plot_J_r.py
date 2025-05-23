# import standard libraries
###################################################################################################
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# import sandbox client
###################################################################################################
from gf_sandbox_client.sbc import Sbc

# import personal files
###################################################################################################
from data_loaders.load_crash_data import load_crash_data



def plot_J_r_waterfall(shot, t_lims=[0, 1000], r_lims=[-0.025, 0.475], J_lims=[0, 1.2], sbc=Sbc(), times_list=[]):
        # unpack limits
    t_min, t_max = t_lims
    r_min, r_max = r_lims
    J_min, J_max = J_lims
    
    
    # load and set up multishot dataframes
        # set up SBC and load data
    sbc.experiment.set("pi3b")
    sbc.user.set("humerben")
    sbc.project.set("high_time_res_recon_crashes")
    df = sbc.get(asset="reconstruction_profile", shot=shot, column=['Javg', "Rmin", "Rmax"], criteria="statistic = mean OR statistic = sigma")
    df_Jaxis = sbc.get(asset="reconstruction", shot=shot, column=['Jaxis'], criteria="statistic = mean")
    df = df.dropna()
        # trim times
    df = df[df['t'] >= t_min]
    df = df[df['t'] <= t_max]
        # separate into mean and errors
    df_err = df[df['statistic'] == 'sigma']
    df = df[df['statistic'] == 'mean']

    # find unique times
    if len(times_list) > 0:
        t_vals = []
        for i in range(len(times_list)):
            if times_list[i] in np.unique(df['t']):
                t_vals.append(times_list[i])
        np.array(t_vals)
    else:
        t_vals = np.unique(df['t'])
    plotting_arrays_J = []
    plotting_arrays_r = []
    
    print(t_vals)
    for t in t_vals:
        # get values for this specific time
            # psibar
        psibars = list(df[df["t"] == t]['psibar'])
            # J
        J_vals = list(df[df['t'] == t]['Javg'])
            # r
        R_min_vals = list(df[df['t'] == t]['Rmin'])
        R_max_vals = list(df[df['t'] == t]['Rmax'])
        
        # sort values for the specific time
            # get indices for sorting
        sorted_psi_indices = np.argsort(psibars)
            # sort J
        J_vals_plotting = [J_vals[i]/(10**6) for i in sorted_psi_indices]
            # sort r
        r_plotting = [(R_max_vals[i] - R_min_vals[i])/2 for i in sorted_psi_indices]
        
        # append values to plotting arrays
            # J
        plotting_arrays_J.append([0] + [df_Jaxis[(df_Jaxis.t == t)].Jaxis.iloc[0]/(10**6)]+ J_vals_plotting + [0])
            # r
        plotting_arrays_r.append([0,0] + r_plotting + [r_plotting[-1]])
    
    
    
    
    
    
    
    
    times = np.array([i*1000 for i in t_vals])
    r_list = plotting_arrays_r
    J_list = plotting_arrays_J


    # Plot
    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(111, projection='3d')

    i=len(times)
    for r, J, t in zip(r_list, J_list, times):
        verts = []

        # Construct the ribbon surface (r, t, J) + bottom line (r, t, 0)
        x = r
        y = np.full_like(r, t)
        z = J

        # Create a 3D polygon for the surface from curve to zero
        verts.append(list(zip(x, y, z)))
        verts.append(list(zip(x[::-1], y[::-1], np.zeros_like(z))))  # bottom face

        # Add as a closed surface

        poly = Poly3DCollection(
            verts,
            facecolor='white',
            edgecolor='black',
            linewidths=0.5
        )
        ax.add_collection3d(poly)

    # Aesthetics
    ax.set_xlabel(r'$r$ $[m]$')
    ax.set_ylabel(r'$t$ $[ms]$')
    ax.set_zlabel(r'$J$ $[MA/m^2]$')
    ax.set_xlim(r_min, r_max)
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_zlim(J_min, J_max)
    ax.view_init(elev=25, azim=75)
    plt.tight_layout()
    # ax.grid(False)
    plt.savefig(f"/home/jupyter-humerben/axuv_paper/plot_outputs/J_q_r/{shot}_J_r_waterfall.png")
    
    return


def plot_J_r_matplotlib(shot, t_lims=[0, 1000], r_lims=[-0.025, 0.475], J_lims=[0, 1.2*10**6], sbc=Sbc()):
    # unpack limits
    t_min, t_max = t_lims
    r_min, r_max = r_lims
    J_min, J_max = J_lims
    
    # prepare and unpack crash data
    crash_df = load_crash_data([shot])
    
    # load and set up multishot dataframes
        # set up SBC and load data
    sbc.experiment.set("pi3b")
    df = sbc.get(asset="reconstruction_profile", shot=shot, column=['Javg', "Rmin", "Rmax"], criteria="statistic = mean OR statistic = sigma")
        # trim times
    df = df[df['t'] >= t_min]
    df = df[df['t'] <= t_max]
        # separate into mean and errors
    df_err = df[df['statistic'] == 'sigma']
    df = df[df['statistic'] == 'mean']

    # set up dicts to store plotting data
    plotting_arrays_J = {}
    plotting_errors_J = {}
    plotting_arrays_r = {}

        
    # find unique times
    t_vals = np.unique(df['t'])
    plotting_arrays_J[shot] = []
    plotting_errors_J[shot] = []
    plotting_arrays_r[shot] = []
    
    for t in t_vals:
        # get values for this specific time
            # psibar
        psibars = list(df[df["t"] == t]['psibar'])
        psibar_errors = list(df_err[df_err["t"] == t]['psibar'])
            # J
        J_vals = list(df[df['t'] == t]['Javg'])
        J_error_vals = list(df_err[df_err['t'] == t]['Javg'])
            # r
        R_min_vals = list(df[df['t'] == t]['Rmin'])
        R_max_vals = list(df[df['t'] == t]['Rmax'])
        
        # sort values for the specific time
            # get indices for sorting
        sorted_psi_indices = np.argsort(psibars)
        sorted_psi_indices_errors = np.argsort(psibar_errors)
            # sort J
        J_vals_plotting = [J_vals[i] for i in sorted_psi_indices]
        J_errors_plotting = [J_error_vals[i] for i in sorted_psi_indices_errors]
            # sort r
        r_plotting = [(R_max_vals[i] - R_min_vals[i])/2 for i in sorted_psi_indices]
        
        # append values to plotting arrays
            # J
        plotting_arrays_J[shot].append(J_vals_plotting)
        plotting_errors_J[shot].append(J_errors_plotting)
            # r
        plotting_arrays_r[shot].append(r_plotting)
        
    # make colormap
    cmap = colormaps["viridis"]
    colormap = [cmap(i) for i in np.linspace(0, 1, len(plotting_arrays_r[shot]))]
    
    # J plot
    for t in range(len(plotting_arrays_r[shot])):
        # plot
        plt.errorbar(x=plotting_arrays_r[shot][t], y=plotting_arrays_J[shot][t], yerr=plotting_errors_J[shot][t],
                            label=f"t={int(round(t_vals[t]*1000, 2))}", color=colormap[t], alpha=0.5, linestyle="-", marker="o")
        
    # handle axes
    plt.xlim(r_min, r_max)
    plt.ylim(J_min, J_max)
    plt.ylabel(r"$J_{\phi}$ $[A/m^2]$")
    plt.xlabel(r"r [m]")
    
    plt.legend(title="time [ms]", ncol=4, fontsize=10, columnspacing=0.5, handletextpad=0.3, handlelength=1.5)
    
    plt.savefig(f"/home/jupyter-humerben/axuv_paper/plot_outputs/J_q_r/{shot}_J_r.png")
    
    return



if __name__ == '__main__':
    plot_J_r_waterfall(22289, [0.007,0.008]) #, times_list=[0.006, 0.00625, 0.0065, 0.00675, 0.007, 0.00725, 0.0075, 0.00775, 0.008, 0.00825, 0.0085, 0.00875])