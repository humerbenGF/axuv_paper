# import standard libraries
###################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import sandbox client
###################################################################################################
from gf_sandbox_client.sbc import Sbc

# import personal files
###################################################################################################
from processing.calculate_plasma_current_slope import calculate_plasma_current_slopes as calc_Ipl_dt



def plot_dIpl_dli1_colored_by_gun_flux(shots, gun_fluxes, sbc=Sbc(), title=False):
    df = calc_Ipl_dt(shots, gun_fluxes, sbc)
    
    fontmin=12
    # scatter
    plt.scatter(df.slope, df.li1, c=df.gun_flux, cmap='viridis')
    
    # labels
    plt.xlabel(r'$dI_{\phi}/dt$ $[MA/s]$', fontsize=fontmin+2)
    plt.ylabel(r'$\ell_{i1}$', fontsize=fontmin+2)
    plt.title(r"Internal Inductance $\ell_{i1}$ Over"+"\n"+"Plasma Current Ramp Rate $dI_{\phi}/dt$" if title else None, fontsize=fontmin+6)
    
    # change tick parameters and add grid
    plt.tick_params(axis='both', labelsize=fontmin)
    plt.grid(True, alpha=0.3)
    
    # colorbar
    cbar = plt.colorbar()
    cbar.set_label(label=r'$\psi_{gun}$', fontsize=fontmin+2)
    cbar.ax.tick_params(axis='both', labelsize=fontmin)
    
    plt.tight_layout()
    
    plt.savefig("/home/jupyter-humerben/axuv_paper/plot_outputs/current_ramp/dli1_dIpl_all.png")
    
    return


def plot_dIpl_dli1_colored_by_gun_flux_grouped(shots, gun_fluxes, group_1, group_2, group_3, sbc=Sbc(), title=False):
    # group_labels = {1:r"$V_{form}=22kV$, $N_{caps}<=32$"+"\n"+"$\psi_{gun}<100mWb$", 2:r"$V_{form}=22kV$, $N_{caps}=48$"+"\n"+"$\psi_{gun}<100mWb$", 
    #      3:r"$V_{form}>=23kV$, $N_{caps}=48$"+"\n"+"$\psi_{gun}>130mWb$"}
    group_labels = {1:"Shots from figure 12.a", 2:"Shots from figure 12.b", 3:"Shots from figure 12.c"}
    df = calc_Ipl_dt(shots, gun_fluxes, sbc)
    
    # make array of markers
    shots_in_df = np.array(df.shot)
    markers_plotting = {1:'P',2:'s',3:'d'}
    slopes_plotting = {1:[],2:[],3:[]}
    li1_plotting = {1:[],2:[],3:[]}
    gun_flux_plotting = {1:[],2:[],3:[]}
    for shot in shots_in_df:
        if shot in group_1:
            slopes_plotting[1].append(df.slope[df.shot == shot].iloc[0])
            li1_plotting[1].append(df.li1[df.shot == shot].iloc[0])
            gun_flux_plotting[1].append(df.gun_flux[df.shot == shot].iloc[0])
        elif shot in group_2:
            slopes_plotting[2].append(df.slope[df.shot == shot].iloc[0])
            li1_plotting[2].append(df.li1[df.shot == shot].iloc[0])
            gun_flux_plotting[2].append(df.gun_flux[df.shot == shot].iloc[0])
        elif shot in group_3:
            slopes_plotting[3].append(df.slope[df.shot == shot].iloc[0])
            li1_plotting[3].append(df.li1[df.shot == shot].iloc[0])
            gun_flux_plotting[3].append(df.gun_flux[df.shot == shot].iloc[0])
    
    fig = plt.figure(dpi=400)
    
    fontmin=12
    # scatter
    plt.scatter(df.slope, df.li1, c=df.gun_flux, cmap='viridis', marker='.', label='Shots not in Figure 12')
    for k in markers_plotting.keys():
        plt.scatter(slopes_plotting[k], li1_plotting[k], c=gun_flux_plotting[k], marker=markers_plotting[k], s=36 if markers_plotting[k] == 's' else 75,
                    cmap='viridis', vmin=min(df.gun_flux), vmax=max(df.gun_flux), edgecolors='black', linewidths=0.5, label=group_labels[k])
    
    # labels
    plt.xlabel(r'$dI_{\phi}/dt\,\mathrm{[MA/s]}$', fontsize=fontmin+2)
    plt.ylabel(r'$\ell_{i1}$', fontsize=fontmin+2)
    plt.title(r"Internal Inductance $\ell_{i1}$ Over"+"\n"+r"Plasma Current Ramp Rate $dI_{\phi}/dt$" if title else None, fontsize=fontmin+6)
    
    # change tick parameters and add grid
    plt.tick_params(axis='both', labelsize=fontmin)
    plt.grid(True, alpha=0.3)
    
    # colorbar
    cbar = plt.colorbar()
    cbar.set_label(label=r'$\psi_{\mathrm{inj}}$', fontsize=fontmin+2)
    cbar.ax.tick_params(axis='both', labelsize=fontmin)
    
    plt.legend(fontsize=fontmin)
    
    plt.tight_layout()
    
    plt.savefig("/home/jupyter-humerben/axuv_paper/plot_outputs/current_ramp/dli1_dIpl_grouped.png")
    plt.close()
    
    
    
    
    return