# import standard libraries
###################################################################################################
import json
import pandas as pd
import pickle as pkl


# import files
###################################################################################################
from .detect_IRE_singleshot import detect_IRE_singleshot


def detect_IRE_multishot(shots, coils_list = ['z211_t*_r100', 'z291_t*_r060', 'z161_t*_r087', 'z261_t*_r087']):
    # load in crash data
    with open('/home/jupyter-humerben/axuv_paper/IRE_exploration/IRE_Bprobe_detection/data/crash_info_with_hardware_error.json', 'r') as f:
        crash_info_dict = json.load(f)
    
    multishot_data = {'shot':[], 't':[], 'z':[], 'r':[], 'omega':[], 'omega_R2':[], 'num_points':[]}
    for shot in shots:
        singleshot_data = detect_IRE_singleshot(shot, coils_list, crash_info_dict)
        shots_singleshot, crash_t_singleshot, probe_z_singleshot, probe_r_singleshot, omega_singleshot, omega_R2_singleshot, num_points_singleshot = singleshot_data
        multishot_data['shot'] += shots_singleshot
        multishot_data['t'] += crash_t_singleshot
        multishot_data['z'] += probe_z_singleshot
        multishot_data['r'] += probe_r_singleshot
        multishot_data['omega'] += omega_singleshot
        multishot_data['omega_R2'] += omega_R2_singleshot
        multishot_data['num_points'] += num_points_singleshot
        
    df = pd.DataFrame(multishot_data)
    with open('/home/jupyter-humerben/axuv_paper/IRE_exploration/IRE_Bprobe_detection/data/output.pkl', 'wb') as f:
        pkl.dump(df, f)
    
    return


