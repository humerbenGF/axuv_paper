# import gf data tools
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

def plot_density_thomson_singleshot(shot, chords_to_include, x_lims_ms=[0, 12]):
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif']
    
    # load in density
    dens_data = load_density(shot, chords_to_include)
    dens_t = dens_data['t']
    keys = list(dens_data.keys())
    for k in keys:
        if k not in chords_to_include:
            del dens_data[k]
            
    # load in temperature
    axuv_Te_data = axuv_Te_with_error(shot)
    axuv_t = axuv_Te_data['axuv_Te_time (ms)']
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

        
    fig, ax1 = plt.subplots(figsize=(6.4,4.8),dpi=300)
    font_min=12
    
    # left axis
    for k in dens_data.keys():
        ax1.plot(dens_t, dens_data[k]*10**6, linewidth=1, label=f"IF Chord at z={k}cm")
    ax1.set_ylabel(r'IF $\mathrm{n_{e}}\,\mathrm{[m^{-3}]}$', fontsize=font_min+2)
    ax1.set_xlabel(r"time $\mathrm{[ms]}$", fontsize=font_min+2)
    ax1.set_ylim(ymin=0)
    ax1.legend(loc='lower left', fontsize=font_min)
    ax1.tick_params(axis='both', labelsize=font_min)

    # right axis
    ax2 = ax1.twinx()
    ax2.plot(axuv_t, axuv_Te, color='r', label=r"AXUV $\mathrm{T_e}$")
    ax2.fill_between(axuv_t, axuv_Te+axuv_Te_plus, axuv_Te-axuv_Te_minus, color='r', alpha=0.2)
    ax2.set_ylabel(r"AXUV $\mathrm{T_e}\,\mathrm{[eV]}$", fontsize=font_min+2)
    ax2.set_ylim(ymin=0, ymax=500)
    ax2.legend(loc='lower right', fontsize=font_min)
    ax2.tick_params(axis='both', labelsize=font_min)

        # thomson_Te
    ax2.scatter(thomson_data[600]['t'], thomson_data[600]['Te'], color='r', zorder=5)
    ax2.scatter(thomson_data[730]['t'], thomson_data[730]['Te'], color='b', zorder=4)
    ax2.scatter(thomson_data[900]['t'], thomson_data[900]['Te'], color='k', zorder=3)
    
    # both axes
    plt.xlim(x_lims_ms)
    ax1.grid(True, alpha=0.4)
    plt.tight_layout()
    
    
    plt.savefig(f"/home/jupyter-humerben/axuv_paper/plot_outputs/density/density_temp_{shot}.png")
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
        return "no_recon_data"
    
    
    singleshot_densities = {}
    for i in range(len(data['waves'])):
        if data['waves'][i].meta['chord'] in chords_to_include:
            singleshot_densities[data['waves'][i].meta['chord']] = data['waves'][i]
        
    # get time data
    waves_list = plotting.collect_waves(data)[0]
    scalars = data['scalars']
    t_lims = plotting.calc_xlims(waves_list, scalars)
    t = np.linspace(t_lims[0]*1000, t_lims[1]*1000, len(singleshot_densities[list(singleshot_densities.keys())[0]]))
    
    singleshot_densities['t'] = t
    
    return singleshot_densities
    
    
def load_thomson(shot):
    return


if __name__ == '__main__':
    plot_density_thomson_singleshot(22744, ['221', '239', '258'])
