import pickle as pkl

def load():
    with open('/home/jupyter-humerben/axuv_paper/IRE_exploration/IRE_Bprobe_detection/data/output.pkl', 'rb') as f:
        df = pkl.load(f)
        
    print(df)
    
    return df