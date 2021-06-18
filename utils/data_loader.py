import numpy as np

def load_data(filename, print_params=False, npz=False):
    """load data from .npz files into dictionary and return"""
    if not npz:
        # .npy file
        data = np.load(filename, allow_pickle=True).item()
        
    if npz:
        # .npz file
        data = {}
        
        df = np.load(filename)   
        print(df)
        for k in df.files:
            data[k] = df[k]

    if print_params:
        print(data.keys())

    return data


