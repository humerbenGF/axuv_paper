# import gf data tools and sandbox client
#################################################################
import GF_data_tools as gdt
from GF_data_tools import plotting
from gf_sandbox_client.sbc import Sbc


# import standard python libraries
#################################################################
import matplotlib.pyplot as plt
import numpy as np
import pickle

# import personal files
#################################################################
from BH_axuv_Te.axuv_Te_function_with_uncertainty_gftools_noplot_v2_202408 import axuv_Te_with_error    # type: ignore

def plot_IRE_multipanel(shot, chords_to_include):
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif']
    
    # load in current
    Ipl, Ipl_t = load_Ipl(shot)
    
    # load in AXUV
    axuv_df = Sbc().get(experiment='pi3b', asset="axuv_amps", shot=shot, criteria=f"sensor_number = {21}")
    axuv_df = axuv_df[(axuv_df.t >= 0) & (axuv_df.t <= 0.01)]
    axuv_t = np.array(axuv_df.t)
    axuv_A = np.array(axuv_df.A)
    axuv_t_sorting_indices = np.argsort(axuv_t)
    axuv_t = np.array([axuv_t[i]*1000 for i in axuv_t_sorting_indices])
    axuv_A = np.array([axuv_A[i]*(10**9) for i in axuv_t_sorting_indices])

    # load in axuv Te
    axuv_Te_data = axuv_Te_with_error(shot)
    axuv_Te_t = axuv_Te_data['axuv_Te_time (ms)']
    axuv_Te = axuv_Te_data['Mylar21&6.2_axuv_Te (eV)']
    axuv_Te_plus = axuv_Te_data['AXUV_Te2 plus uncertainty']
    axuv_Te_minus = axuv_Te_data['AXUV_Te2 minus uncertainty']
    
    # get thomson data
    thomson_df = Sbc().get(experiment='pi3b', asset="thomson_temperature", shot=shot)
    thomson_R_positions = [600, 730, 900]
    thomson_data = {}
    for pos in thomson_R_positions:
        # load data
        T = thomson_df[thomson_df.R == pos].Te
        T_err = thomson_df[thomson_df.R == pos].Te_err
        t = thomson_df[thomson_df.R == pos].t
        t_ind_sort = np.argsort(t)
        T = [T.iloc[i] for i in t_ind_sort]
        T_err = [T_err.iloc[i] for i in t_ind_sort]
        t = [t.iloc[i]*1000 for i in t_ind_sort]
        # save data to dict
        thomson_data[pos] = {'Te': T, 'Te_err':T_err, 't':t}
        
    # load in particle inventory
    sbc=Sbc()
    pi_df_mean = sbc.get(project='high_time_res_recon_crashes', experiment='pi3b', shots=[shot], asset='reconstruction', columns=['ParticleInventory'], criteria='statistic=mean')
    pi_mean = np.array(pi_df_mean.ParticleInventory)
    pi_df_std = sbc.get(project='high_time_res_recon_crashes', experiment='pi3b', shots=[shot], asset='reconstruction', columns=['ParticleInventory'], criteria='statistic=sigma')
    pi_std = np.array(pi_df_std.ParticleInventory)
    pi_t = [i*1000 for i in np.sort(np.unique(pi_df_mean.t))]
    
    # load in density
    dens_data = load_density(shot, chords_to_include)
    dens_t = dens_data['t']
    keys = list(dens_data.keys())
    for k in keys:
        if k not in chords_to_include:
            del dens_data[k]
            


    fig, axs = plt.subplots(3,1,figsize=(6.2,7.66),dpi=300)
    font_min=12
    tab_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    
    # Ipl and axuv plot
        # Ipl
    axs[0].plot(Ipl_t, Ipl, color='r')
    axs[0].set_ylabel(r"$I_\mathrm{pl}\,\mathrm{[MA]}$", fontsize=font_min+2)
    axs[0].tick_params(axis='both', labelsize=font_min)
        # axuv
    ax0_twin = axs[0].twinx()
    ax0_twin.plot(axuv_t, axuv_A, color='b')
    ax0_twin.set_xlim(0, 10)
    ax0_twin.set_ylim(0, 2.5e2)
    ax0_twin.set_ylabel(r"AXUV Current $\mathrm{[nA]}$", fontsize=font_min+2, color='b')
    ax0_twin.tick_params(axis='both', labelsize=font_min)
    
    axs[0].set_xticklabels([])
    axs[0].grid(True, alpha=0.4)


    # temp and part inv plot
        # axuv_Te
    axs[1].plot(axuv_Te_t, axuv_Te, color=tab_colors[0], label=r"AXUV $\mathrm{T_e}$", zorder=2)
    axs[1].fill_between(axuv_Te_t, axuv_Te+axuv_Te_plus, axuv_Te-axuv_Te_minus, color=tab_colors[0], alpha=0.3, zorder=2)
    axs[1].set_ylabel(r"AXUV $T_\mathrm{e}\,\mathrm{[eV]}$", fontsize=font_min+2)
    axs[1].set_ylim(ymin=0, ymax=600)
    axs[1].set_xlim(0, 10)
    # axs[1].legend(loc='lower right', fontsize=font_min)
    axs[1].tick_params(axis='both', labelsize=font_min)
        # thomson_Te
    axs[1].scatter(thomson_data[600]['t'], thomson_data[600]['Te'], color='r', zorder=5)
    axs[1].scatter(thomson_data[730]['t'], thomson_data[730]['Te'], color='b', zorder=4)
    axs[1].scatter(thomson_data[900]['t'], thomson_data[900]['Te'], color='k', zorder=3)

        # particle inv
    ax1_twin = axs[1].twinx()
    ax1_twin.plot(pi_t, pi_mean, color=tab_colors[1], zorder=2)
    ax1_twin.fill_between(pi_t, pi_mean+pi_std, pi_mean-pi_std, color=tab_colors[1], alpha=0.3, zorder=2)
    ax1_twin.set_ylabel(r"$N_p\,\mathrm{[m^{-3}]}$", fontsize=font_min+2, color=tab_colors[1])
    ax1_twin.set_xlim(0, 10)
    ax1_twin.set_ylim(ymin=0)
        
        # general settings
    axs[1].set_xticklabels([])
    axs[1].grid(True, alpha=0.4, zorder=0)
    
    # density plot
    for k in dens_data.keys():
        axs[2].plot(dens_t, dens_data[k]*10**6, linewidth=1, label=f"z={k}cm")
    axs[2].set_ylabel(r'IF $n_\mathrm{{e}}\,\mathrm{[m^{-3}]}$', fontsize=font_min+2)
    axs[2].set_xlabel(r"time $\mathrm{[ms]}$", fontsize=font_min+2)
    axs[2].set_ylim(ymin=0, ymax=5e19)
    axs[2].set_xlim(0, 10)
    axs[2].legend(loc='upper right', fontsize=font_min, ncols=2)
    axs[2].tick_params(axis='both', labelsize=font_min)
    axs[2].grid(True, alpha=0.4)

    
    # both axes
    plt.tight_layout()
    
    
    plt.savefig(f"/home/jupyter-humerben/axuv_paper/plot_outputs/density/IRE_multipanel_{shot}.png")
    plt.close()
    
    return


def load_density(shot, chords_to_include):
    dens_options = {'experiment':'pi3b',
                    'manifest': 'default',
                    'shot': shot,
                    'layers':'dual_interferometer/*',
                    'nodisplay': True}

    try:
        data = gdt.fetch_data.run_query(dens_options)
        dens_options['layers'] = 'heterodyne_interferometer/*'
        data_for_t = gdt.fetch_data.run_query(dens_options)
    except:
        return "no_density_data"
    
    singleshot_densities = {}
    for i in range(len(data['waves'])):
        if data['waves'][i].meta['chord'] in chords_to_include:
            singleshot_densities[data['waves'][i].meta['chord']] = gdt.wave_utils.low_pass_filter(data['waves'][i], 40e3)
        
    # get time data
    waves_list = plotting.collect_waves(data)[0]
    scalars = data['scalars']
    t_lims = plotting.calc_xlims(waves_list, scalars)
    t = np.linspace(t_lims[0]*1000, t_lims[1]*1000, len(singleshot_densities[list(singleshot_densities.keys())[0]]))
    
    singleshot_densities['t'] = t
    
    return singleshot_densities


def load_Ipl(shot):
    Ipl_options = {'experiment':'pi3b',
                    'manifest': 'default',
                    'shot': shot,
                    'layers':'mirnov_combined/plasma_current_avg',
                    'nodisplay': True}

    try:
        data = gdt.fetch_data.run_query(Ipl_options)
    except:
        return "no_density_data"
    
    Ipl = [i/(10**6) for i in data['waves'][0]]
    Ipl_t = [i*1000 for i in data['waves'][0].x_axis()]
    
    return Ipl, Ipl_t


if __name__ == '__main__':
    plot_IRE_multipanel(22605, ['221', '239', '258', '276'])
    # sbc=Sbc()
    # sbc.asset.set('reconstruction')
    # print(sbc.columns.describe())
