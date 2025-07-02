# import standard gf libraries
#################################################################
import numpy as np
import pandas as pd

# import gf_sandbox
#################################################################
from gf_sandbox_client.sbc import Sbc

# import personal files
#################################################################
from AXUV_crash_categorization.detect_features.detect_rational_q_surfaces import detect_rational_q_surfaces_singleshot as det_rat_q_ss
from data_loaders.load_crash_data import load_crash_data as load_crash

def get_percentage_double_surface_crossings_leading_to_crash(shots, rational_surfaces_list=[1, 3/2, 2, 5/2, 3], sbc=Sbc()):
    sbc.user.set('humerben')
    sbc.experiment.set('pi3b')
    multishot_crash = load_crash(shots)
    
    data = {"total":[]}
    shots_with_dtm = {"shots_with_dtm":[], "shots_without_dtm":[]}
    
    for surface in rational_surfaces_list:
        data[surface] = []
    
    for shot in shots:
        if shot not in np.array(multishot_crash.shot):
            continue
        
        rat_q_df = det_rat_q_ss(shot, rational_surfaces_list, sbc)
        crash_df = multishot_crash[multishot_crash.shot == shot]        
        # check which surfaces are crossed at t_min
        t_min = min(rat_q_df.t)
        singletime_df = rat_q_df[(rat_q_df.t == t_min)]
        singletime_df.loc[:, 'psibar'] = singletime_df['psibar'].fillna(0)
        singletime_df = singletime_df[singletime_df.psibar > 0]
                        
        surfaces_not_crossing_at_tmin = []
        for surface in rational_surfaces_list:
            if surface in np.nan_to_num(np.array(singletime_df.surface)):
                continue
            else:
                surfaces_not_crossing_at_tmin.append(surface)
    
        # get recon times before and after the crash
        pre_crash_recon_time = max(rat_q_df[rat_q_df.t < min(crash_df.pre_crash_t)].t)
        post_crash_recon_time = min(rat_q_df[rat_q_df.t > min(crash_df.post_crash_t)].t)
        
        # get dfs for pre and post
        pre_crash_df = rat_q_df[(rat_q_df.t == pre_crash_recon_time) & (rat_q_df.psibar > 0)]
        post_crash_df = rat_q_df[(rat_q_df.t == post_crash_recon_time) & (rat_q_df.psibar < 0)]
        
        # check pre for any crossing of listed surfaces
        surfaces_that_cross = []
        for surface in surfaces_not_crossing_at_tmin:
            early_time_rat_q_df_singlesurface = rat_q_df[(rat_q_df.t <= pre_crash_recon_time) & (rat_q_df.surface == surface)]
            if np.nanmax(np.nan_to_num(np.array(early_time_rat_q_df_singlesurface.psibar), nan=0)) > 0:
                surfaces_that_cross.append(surface)
        
        surfaces_with_crash = []
        for surface in surfaces_not_crossing_at_tmin:
            if (surface in np.array(pre_crash_df[pre_crash_df.gt_qmin == True].surface)) and (surface in np.array(pre_crash_df[pre_crash_df.gt_qmin == False].surface)):
                if (surface not in np.array(post_crash_df[post_crash_df.gt_qmin == True].surface)) and (surface not in np.array(post_crash_df[post_crash_df.gt_qmin == False].surface)):
                    surfaces_with_crash.append(surface)

        print("shot:", shot, "surfaces_that_cross:", surfaces_that_cross, "surfaces_with_crash:", surfaces_with_crash)

        for surface in rational_surfaces_list:
            if surface in surfaces_that_cross:
                if surface in surfaces_with_crash:
                    data[surface].append(1)
                    data['total'].append(1)
                    if shot not in shots_with_dtm['shots_with_dtm']:
                        shots_with_dtm['shots_with_dtm'].append(shot)
                elif surface < custom_max(surfaces_with_crash):
                    print("test")
                    continue
                else:
                    data[surface].append(0)
                    data['total'].append(0)
                    
        if shot not in shots_with_dtm['shots_with_dtm']:
            shots_with_dtm['shots_without_dtm'].append(shot)
    
    all_shots_dtm = []
    all_shots_dtm_tf = []
    for k in shots_with_dtm.keys():
        all_shots_dtm += shots_with_dtm[k]
        for i in range(len(shots_with_dtm[k])):
            if k == 'shots_with_dtm':
                all_shots_dtm_tf.append(True)
            else:
                all_shots_dtm_tf.append(False)
    
    pd.DataFrame({"shot":all_shots_dtm, "double_crossing_at_crash":all_shots_dtm_tf}).to_csv("/home/jupyter-humerben/axuv_paper/processing_outputs/shots_with_and_without_double_crossing_crash.csv", index=False)
                    
    printout_crash_surface_data(data)
    
    return

def get_percent_double_surface_crossing_while_assigning_every_crash(shots, rational_surfaces_list=[1, 3/2, 2, 5/2, 3], sbc=Sbc()):
    sbc.user.set('humerben')
    sbc.experiment.set('pi3b')
    multishot_crash = load_crash(shots)
    
    data = {"total":0}
    first_crashes_per_surface = {}
    
    for surface in rational_surfaces_list:
        data[surface] = 0
    
    for shot in shots:
        if shot not in np.array(multishot_crash.shot):
            continue
        
        # prep data for this specific shot
        rat_q_df = det_rat_q_ss(shot, rational_surfaces_list, sbc)
        rat_q_df.loc[:, 'psibar'] = rat_q_df['psibar'].fillna(0)
        crash_df = multishot_crash[multishot_crash.shot == shot]
        
        # get timing information
        first_crash_time = np.min(crash_df.t)
        recon_times = np.sort(np.unique(rat_q_df[rat_q_df.t < first_crash_time].t))[::-1]
        crossing = False
        for recon_time in recon_times:
            for surface in rational_surfaces_list:
                rat_q_df_st_ss = rat_q_df[(rat_q_df.t == recon_time) & (rat_q_df.surface == surface)]
                inner_crossing = True if rat_q_df_st_ss[rat_q_df_st_ss.gt_qmin == False].psibar.iloc[0] > 0 else False
                outer_crossing = True if rat_q_df_st_ss[rat_q_df_st_ss.gt_qmin == True].psibar.iloc[0] > 0 else False
                if inner_crossing and outer_crossing:
                    data[surface] += 1
                    data['total'] += 1
                    crossing = True
                    break
                
            if crossing:
                break
    
    for k in data.keys():
        print(f"q={k} surface: {data[k]}/{data['total']}, {data[k]/data['total']*100}%")
    
    return


def custom_max(surfaces_with_crash):
    if len(surfaces_with_crash) == 0:
        return 0
    return max(surfaces_with_crash)


def printout_crash_surface_data(data_dict):
    for k in data_dict.keys():
        if k == 'total':
            print(f"total dataset: {str(np.mean(data_dict[k])*100)[:5]}% of surface crossings lead to the first crash ({len(data_dict[k])} surface crossings)")
        elif len(data_dict[k]) > 0:
            print(f"\tq={k} surface: {str(np.mean(data_dict[k])*100)[:5]}% of surface crossings lead to the first crash ({len(data_dict[k])} surface crossings)")



if __name__ == '__main__':
    # get_percentage_double_surface_crossings_leading_to_crash([22289, 22744])
    get_percent_double_surface_crossing_while_assigning_every_crash([22289, 22744])