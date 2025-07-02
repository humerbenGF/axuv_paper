from gf_sandbox_client.sbc import Sbc

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import scipy.constants as const
from scipy.optimize import curve_fit


from data_loaders.load_crash_data import load_crash_data


def dIsh_dli1_ov_li1(shots):
    sbc = Sbc()
    sbc.experiment.set('pi3b')
    sbc.asset.set('reconstruction')
    
    df = sbc.get(shots=shots, columns=['li1', 'Ishaft', "Ipl"], criteria='statistic = mean')
    
    
    fig = go.Figure()

    for shot in shots:
        dIsh_dli1_plot = []
        li1_plot = []
        df_ss = df[df.shot == shot]
        

        max_Ipl_index = np.argmax(np.array(df_ss.Ipl))
        max_Ipl_time = np.array(df_ss.t)[max_Ipl_index]
        df_ss = df_ss[df_ss.t <= max_Ipl_time]
        Ish, Ipl, li1, t = np.array(df_ss.Ishaft), np.array(df_ss.Ipl), np.array(df_ss.li1), np.array(df_ss.t)
        

        for i in range(len(li1)-1):
            dIsh_dli1_plot.append((Ish[i+1] - Ish[i]) / (li1[i+1]-li1[i]) / 10**6)
            li1_plot.append((li1[i+1]+li1[i])/2)
            
        fig.add_trace(go.Scatter(x=li1_plot, y=dIsh_dli1_plot, mode='markers', showlegend=False))
    
    fig.update_layout(
        title=dict(
            text=r"Plot of $dI_{\mathrm{sh}}/d\ell_{i1}$ over $\ell_{i1}$",  # Dynamic title
            x=0.5,
            y=0.95,
            font=dict(size=28)
        ),
        xaxis_title=r"$\ell_{i1}$",
        yaxis_title=r"$dI_{\mathrm{sh}}/d\ell_{i1}\,\mathrm{[MA]}$",  # Replace with actual column name if needed
    )
    fig.show()
    
    return

def dIsh_over_dli1(shots):
    sbc = Sbc()
    sbc.experiment.set('pi3b')
    sbc.asset.set('reconstruction')
    
    df = sbc.get(shots=shots, columns=['li1', 'Ishaft', "Ipl"], criteria='statistic = mean')
    
    fig = go.Figure()

    for shot in shots:
        dIsh_dt_plot = []
        dli1_dt_plot = []
        df_ss = df[df.shot == shot]
        

        max_Ipl_index = np.argmax(np.array(df_ss.Ipl))
        max_Ipl_time = np.array(df_ss.t)[max_Ipl_index]
        df_ss = df_ss[df_ss.t <= max_Ipl_time]
        Ish, Ipl, li1, t = np.array(df_ss.Ishaft), np.array(df_ss.Ipl), np.array(df_ss.li1), np.array(df_ss.t)
        

        for i in range(len(li1)-1):
            dIsh_dt_plot.append((Ish[i+1]-Ish[i])/(t[i+1]-t[i])/10**6)
            dli1_dt_plot.append((li1[i+1]-li1[i])/(t[i+1]-t[i]))
            
        fig.add_trace(go.Scatter(x=dli1_dt_plot, y=dIsh_dt_plot, mode='markers', showlegend=False))
    
    fig.update_layout(
        title=dict(
            text=r"Plot of $dI_{\mathrm{sh}}/dt$ over $\ell_{i1}$",  # Dynamic title
            x=0.5,
            y=0.95,
            font=dict(size=28)
        ),
        xaxis_title=r"$d\ell_{i1}/dt\,\mathrm{[s^{-1}]}$",
        yaxis_title=r"$dI_{\mathrm{sh}}/dt\,\mathrm{[MA/s]}$",  # Replace with actual column name if needed
    )
    fig.show() 
            
            
    return

def dIsh_over_dli1_matplotlib(shots, gun_fluxes, use_crash_as_max=True):
    sbc = Sbc()
    sbc.project.set("default")
    sbc.experiment.set('pi3b')
    sbc.asset.set('reconstruction')
    
    df = sbc.get(shots=shots, columns=['li1', 'Ishaft', "Ipl"], criteria='statistic = mean')
    sbc.asset.set('shot_log')
    df_sustain = sbc.get(shots=shots, columns=['sustain'])
    
    crash_df = load_crash_data(shots)
    
    dIsh_dt_plot = []
    dli1_dt_plot = []
    gun_flux_plot = []
    sustain_plot = []
    li1_plot = []
    for s in range(len(shots)):
        shot = shots[s]
        gun_flux = gun_fluxes[s]
        sustain = df_sustain[df_sustain.shot == shot].sustain.iloc[0]/1000

        df_ss = df[df.shot == shot]
        if len(df_ss) == 0:
            continue
        
        max_Ipl_index = np.argmax(np.array(df_ss.Ipl))
        max_Ipl_time = np.array(df_ss.t)[max_Ipl_index]
        if use_crash_as_max:
            crash_times_ss = crash_df[crash_df.shot == shot].pre_crash_t
            if len(crash_times_ss)>0:
                max_Ipl_time = min(crash_times_ss)
            else:
                continue
        df_ss = df_ss[df_ss.t <= max_Ipl_time]
        Ish, Ipl, li1, t = np.array(df_ss.Ishaft), np.array(df_ss.Ipl), np.array(df_ss.li1), np.array(df_ss.t)
        
        if len(li1) < 2:
            continue

        dIsh_dt_plot.append((Ish[-1]-Ish[0])/(t[-1]-t[0])/10**6)
        dli1_dt_plot.append((li1[-1]-li1[0])/(t[-1]-t[0]))
        gun_flux_plot.append(gun_flux)
        sustain_plot.append(sustain)
        li1_plot.append((li1[-1]+li1[0])/2)
    
    plt.figure(figsize=(10,6), dpi=300)
    plt.grid(True, alpha=0.5, zorder=0)
    cbar_array=sustain_plot
    plt.scatter(dIsh_dt_plot, dli1_dt_plot, c=cbar_array, marker='o', cmap='viridis',
                vmin=min(cbar_array), vmax=max(cbar_array), edgecolors='black', linewidths=0.25, zorder=3)
    
    cbar = plt.colorbar()
    cbar.set_label(r"$V_{sust}\,\mathrm{[kV]}$")
    plt.xlabel(r"$\Delta I_{\mathrm{sh}}/\Delta t\,\mathrm{[MA/s]}$")
    plt.ylabel(r"$\Delta \ell_{i1}/\Delta t\,\mathrm{[s^{-1}]}$")
    plt.title(r"$\Delta \ell_{i1}/\Delta t$ over $\Delta I_{\mathrm{sh}}/\Delta t$")
    plt.tight_layout()
    plt.savefig("/home/jupyter-humerben/axuv_paper/plot_outputs/current_ramp/dIsh_dli1_col_sustain.png")
    plt.close()
    
    return


def li1_over_inv_Ish_squared(shots, gun_fluxes, use_crash_as_max=True):
    sbc = Sbc()
    sbc.project.set("default")
    sbc.experiment.set('pi3b')
    sbc.asset.set('reconstruction')
    
    df = sbc.get(shots=shots, columns=['li1', 'Ishaft', "Ipl"], criteria='statistic = mean')
    sbc.asset.set('shot_log')
    df_sustain = sbc.get(shots=shots, columns=['sustain'])
    
    crash_df = load_crash_data(shots)
    
    dli1_plot = []
    dt_plot = []
    inv_dIsh_sqr_plot = []
    gun_flux_plot = []
    sustain_plot = []
    li1_plot = []
    for s in range(len(shots)):
        shot = shots[s]
        gun_flux = gun_fluxes[s]
        sustain = df_sustain[df_sustain.shot == shot].sustain.iloc[0]/1000

        df_ss = df[df.shot == shot]
        if len(df_ss) == 0:
            continue
        
        max_Ipl_index = np.argmax(np.array(df_ss.Ipl))
        max_Ipl_time = np.array(df_ss.t)[max_Ipl_index]
        if use_crash_as_max:
            crash_times_ss = crash_df[crash_df.shot == shot].pre_crash_t
            if len(crash_times_ss)>0:
                max_Ipl_time = min(crash_times_ss)
            else:
                continue
        df_ss = df_ss[df_ss.t <= max_Ipl_time]
        Ish, Ipl, li1, t = np.array(df_ss.Ishaft), np.array(df_ss.Ipl), np.array(df_ss.li1), np.array(df_ss.t)
        
        dli1 = li1[-1]-li1[0]
        R0=0.6
        U_initial = 1/4 * (const.mu_0 * (li1[0]) * R0) * ((Ipl[0]))**2
        U = 1/4 * (const.mu_0 * (li1[-1]+li1[0])/2 * R0) * ((Ipl[-1]+Ipl[0])/2)**2
        U=U_initial
        inv_dIsh_sqr = ((1/Ipl[-1])**2 - (1/Ipl[0])**2)
        inv_dIsh_sqr = 4*U*inv_dIsh_sqr / (const.mu_0 * R0)
        if (not dli1 > 0.4) and (not inv_dIsh_sqr < -0.4) and (not inv_dIsh_sqr > 0.4):
            dli1_plot.append(dli1)
            dt_plot.append(t[-1]-t[0])
            inv_dIsh_sqr_plot.append(inv_dIsh_sqr)
            sustain_plot.append(sustain)
            gun_flux_plot.append(gun_flux)
            li1_plot.append(li1[0])
    
    # fit functions
        # linear
    coeffs = np.polyfit(inv_dIsh_sqr_plot, dli1_plot, 1)
    print("linear coeffs:", coeffs)
    poly_func = np.poly1d(coeffs)
    r2 = calc_R2(inv_dIsh_sqr_plot, dli1_plot, poly_func)
        # exp*linear
    popt, pcov = curve_fit(linear_with_exp_adjust, (inv_dIsh_sqr_plot, dt_plot), dli1_plot, p0=(0.5, 0, 0.01))
    print(f"lin*exp coeffs: a = {popt[0]:.4f}, b = {popt[1]:.4f}, c = {popt[2]:.4f}")
    dli1_plot_adjusted = [dli1_plot[i]*np.exp(-1*dt_plot[i]/popt[2]) for i in range(len(dli1_plot))]
    inv_dIsh_sqr_plot_adjusted = [inv_dIsh_sqr_plot[i]*np.exp(-1*dt_plot[i]/popt[2]) for i in range(len(dt_plot))]
    poly_func_adjusted = np.poly1d([popt[0], popt[1]])
    r2_adjusted = calc_R2(inv_dIsh_sqr_plot_adjusted, dli1_plot, poly_func_adjusted)
    print(f"fit quality: non-adjusted={r2}, adjusted={r2_adjusted}")

    
    plt.figure(figsize=(10,6), dpi=300)
    plt.grid(True, alpha=0.5, zorder=0)
    cbar_array=sustain_plot
    plt.scatter(inv_dIsh_sqr_plot, dli1_plot, c=cbar_array, marker='o', cmap='viridis',
                vmin=min(cbar_array), vmax=max(cbar_array), edgecolors='black', linewidths=0.25, zorder=3, alpha=0.2)
    
    plt.scatter(inv_dIsh_sqr_plot_adjusted, dli1_plot, c=cbar_array, marker='d', cmap='viridis',
                vmin=min(cbar_array), vmax=max(cbar_array), edgecolors='black', linewidths=0.25, zorder=4, alpha=0.7)
    
    plt.plot(inv_dIsh_sqr_plot, poly_func(inv_dIsh_sqr_plot), color='r', label=r'Linear Fit: $R^2=$'+f"{round(r2, 3)}", alpha=0.4, zorder=3, linewidth=2)
    plt.plot(inv_dIsh_sqr_plot_adjusted, poly_func_adjusted(inv_dIsh_sqr_plot_adjusted), color='r', label=r'Fit to $y=ax\cdot e^{-t/\tau}+b$: $R^2=$'+f"{round(r2_adjusted, 3)}",
             zorder=5, linewidth=2)
    
    
    cbar = plt.colorbar()
    cbar.set_label(r"$V_\mathrm{sust}\,\mathrm{[kV]}$", fontsize=18)
    plt.xlabel(r"$\frac{4\,U_{\mathrm{L}}}{\mu_0R_0}\Delta (I_\mathrm{\phi}^{-2})$", fontsize=18)
    plt.ylabel(r"$\Delta \ell_{i1}$", fontsize=18)
    plt.title(r"$\Delta \ell_{i1}$ over $\frac{4\,U_{\mathrm{L}}}{\mu_0R_0}\Delta (I_\mathrm{\phi}^{-2})$", fontsize=24)
    
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("/home/jupyter-humerben/axuv_paper/plot_outputs/current_ramp/dli1_dinv_Ipl_squared.png")
    plt.close()
    
    
    
    return


def linear_with_exp_adjust(X, a, b, c):
    x, t = X
    return (a * x* np.exp(t/c*-1) + b) 


def calc_R2(x,y,fit_fn):

    # Predicted y values
    y_pred = fit_fn(x)

    # R² calculation
    ss_res = np.sum((y - y_pred) ** 2)       # Residual sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)   # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    
    return r_squared


if __name__ == '__main__':
    non_sustain_high_gun_flux_and_formation = [22729,22731,22732,22733,22734,22735,22736,22737,22743,22744,22745,22754,22755]
    
    non_sustain = pd.read_csv("/home/jupyter-humerben/axuv_paper/datasets/non_sustain_dataset.csv")
    sustain = pd.read_csv("/home/jupyter-humerben/axuv_paper/datasets/sustain_dataset.csv")
    non_sustain_shots = list(np.array(non_sustain['shot']))
    non_sustain_gun_flux = list(np.array(non_sustain['gun_flux_mWb']))
    sustain_shots = list(np.array(sustain['shot']))
    sustain_gun_flux = list(np.zeros(len(sustain_shots)))
    
    shots = list(np.append(non_sustain_shots, sustain_shots))
    gun_flux = list(np.append(non_sustain_gun_flux, sustain_gun_flux))
    

    # dIsh_dli1_ov_li1(non_sustain_high_gun_flux_and_formation)
    # dIsh_over_dli1(non_sustain_high_gun_flux_and_formation)
    
    # dIsh_over_dli1_matplotlib(shots, gun_flux)
    
    # li1_over_inv_Ish_squared(non_sustain_shots, non_sustain_gun_flux)
    li1_over_inv_Ish_squared(shots, gun_flux)