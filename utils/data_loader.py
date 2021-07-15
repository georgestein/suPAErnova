import tensorflow as tf
import numpy as np
import pandas as pd

def load_data(filename, remove_negatives=True, print_params=False, npz=False, to_tensor=False):
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

    if to_tensor:
        # convert dtype from numpy.float32 or np.float64 to tensor
        # AE model trains slower when using tf.tensor()...
        for k, v in data.items():
            print(k, v.dtype)
            v_dtype = v.dtype
            if v_dtype==np.float32 or v_dtype==np.float64:
                data[k] = tf.convert_to_tensor(v, dtype=tf.float32)
        
    if print_params:
        print(data.keys())

    if remove_negatives:
        # remove negative flux values in observed spectrum
        dm = data['mask'][..., 0] == 1
        data['spectra'][dm] = np.clip(data['spectra'][dm], 0, np.inf)

    return data

def get_train_mask(data, params):

    dm_redshift = (data['redshifts'] > params['min_train_redshift']) & \
        (data['redshifts'] < params['max_train_redshift'])

    dm_maxlight = (data['times_orig'] > params['max_light_cut'][0]) & \
\
        (data['times_orig'] < params['max_light_cut'][1])
    dm_maxlight = np.any(dm_maxlight, axis=(1,2))

    dm = dm_redshift & dm_maxlight

    if params['twins_cut']:
        in_twins, dm_twins = get_twins_mask(data)
        dm = dm & dm_twins

    return dm

def get_twins_mask(data):
    """Return mask arrays denoting whether SN exists in the Twins dataset (in_twins),
    and whether it was used in the Twins paper (mask_twins)"""
    df_twins = pd.read_csv('data/boone_data.dat', delimiter = ",")
    # get twins data for sn
    set_snf = set(list(data['names']))
    set_boone = set(list(df_twins['name']))

    intersection = list(set_snf & set_boone)

    in_twins = np.full(data['redshifts'].shape[0], False)
    mask_twins = np.full(data['redshifts'].shape[0], False)
    dm_twins = np.zeros(data['redshifts'].shape[0])
    dm_salt = np.zeros(data['redshifts'].shape[0])

    for i, name in enumerate(intersection):
        ind_snf = np.argwhere(data['names'] == name)[0]
        dfi = df_twins[df_twins['name'] == name]
        
        mask_twins[ind_snf] = dfi['mask_twins']
        in_twins[ind_snf]   = True

        #dm_twins[ind_snf] = dfi['dm_residuals_twins']
        #dm_salt[ind_snf]  = dfi['dm_residuals_salt']

    return in_twins,  mask_twins
