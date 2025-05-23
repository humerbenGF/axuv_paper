# import standard libraries
###################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import sandbox client
###################################################################################################
from gf_sandbox_client.sbc import Sbc


def calculate_plasma_current_slopes(shot_list, gun_flux, sbc=Sbc()):
    df = sbc.get(shot=shot_list, experiment='pi3b', columns=["Ipl", 'li1'], asset='reconstruction', user='humerben', criteria='statistic = mean')
    
    shots_for_df = []
    slopes_for_df = []
    dIpl_for_df = []
    dt_for_df = []
    li1_for_df = []
    gun_flux_for_df = []
    
    for i in range(len(shot_list)):
        shot = shot_list[i]
        if shot not in np.unique(df.shot):
            continue
        # load in data
        df_ss = df[df.shot == shot]
        times = np.array(df_ss.t)
        Ipl = np.array(df_ss.Ipl)
        li1 = np.array(df_ss.li1)
        
        # sort data
        times_sorting_inds = np.argsort(times)
        times = np.array([times[i] for i in times_sorting_inds])
        Ipl = np.array([Ipl[i] for i in times_sorting_inds])
        li1 = np.array([li1[i] for i in times_sorting_inds])
        
        # find min and max of Ipl
        Ipl_max_ind = np.argmax(Ipl)
        if Ipl_max_ind > 0:
            Ipl_min_ind = np.argmin(Ipl[:Ipl_max_ind])
            
            # find dIpl/dt
            dIpl = (Ipl[Ipl_max_ind] - Ipl[Ipl_min_ind])
            dt = (times[Ipl_max_ind] - times[Ipl_min_ind])
            slope = dIpl / dt / 10**6

            # save data for dataframe
            shots_for_df.append(shot)
            slopes_for_df.append(slope)
            dIpl_for_df.append(dIpl)
            dt_for_df.append(dt)
            li1_for_df.append(li1[Ipl_max_ind])
            gun_flux_for_df.append(gun_flux[i])
        
    dict_for_df = {'shot':shots_for_df, 'slope':slopes_for_df, 'li1':li1_for_df, 'gun_flux':gun_flux_for_df, "dIpl":dIpl_for_df, 'dt':dt_for_df}
    
    df_slopes = pd.DataFrame(dict_for_df)
    
    
    return df_slopes




if __name__ == '__main__':
    sbc = Sbc()
    sbc.project.set("high_time_res_recon_crashes")
    df = calculate_plasma_current_slopes([22289, 22744], sbc=sbc)
    
    print(df)