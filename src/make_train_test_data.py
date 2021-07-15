#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
from astropy.cosmology import FlatLambdaCDM

"""
Reshape data from individual spectra to (N_sn, n_timesample, n_wavelengths)
"""

# Load Data
class time_normalization():
    def __init__(self, times, normed=False):
#        super(time_normalization, self).__init__()

        self.tmin = -10
        self.tmax = 40
#         self.tmin = np.min(times)
#         self.tmax = np.max(times)
        
        self.minmax = self.tmax - self.tmin
        
        # only normalize existing times, not null value (-100)
        self.dm = (times > -100)
        
        self.normed = normed

    def scale(self, times):
        
        if self.normed:
#             times = times * self.minmax
            times[self.dm] = times[self.dm] * self.minmax + self.tmin
            self.normed = False

        else:
#             times = times/self.minmax
            times[self.dm] = (times[self.dm] - self.tmin)/self.minmax

            times[~self.dm] = -1.
        
            self.normed = True
            
        return times

if __name__ == '__main__':
    
    data   = np.load('data/snf_data_wSALT.npz')
    print(data.files)
    figdir = '../figures/'

    # only_snf = True
    only_snf = False

    train_frac = 0.75
    test_frac = 1-train_frac
    nkfold = int(1./test_frac)
    print('nkfold = ', nkfold)
    min_redshift = 0.02
    
    train_file_head = 'data/train_data'
    test_file_head = 'data/test_data'

    verbose = True

    data_out = {}
    data_out['salt_params_label'] = data['salt_params_label']
    data_out['wavelengths'] = data['wavelengths']     

    data_out['names'] = data['names']
    # reshape SN by ids, and 0 pad to (n_sn, n_timemax, n_wavelength)
    # where n_timemax is the maximum number of observations of any SN in dataset
    
    n_timestep     = np.max(np.bincount(data['spectra_IDs']))
    n_timestep_min = np.min(np.bincount(data['spectra_IDs']))
    
    n_sn         = len(np.unique(data['spectra_IDs']))
    n_wavelength = len(data['wavelengths'])

    data_out['spectra']      = np.zeros( (n_sn, n_timestep, n_wavelength), dtype=np.float32) - 1 # set missing data to -1
    data_out['spectra_salt'] = np.zeros( (n_sn, n_timestep, n_wavelength), dtype=np.float32) - 1 # set missing data to -1
    data_out['sigma']        = np.ones(  (n_sn, n_timestep, n_wavelength), dtype=np.float32) - 1
    data_out['mask']         = np.zeros( (n_sn, n_timestep, 1), dtype=np.float32)
    
    data_out['times']        = np.zeros( (n_sn, n_timestep, 1),            dtype=np.float32) - 100 # set missing data to -100
    
    data_out['salt_params']  = np.zeros( (n_sn, data['salt_params'].shape[1]), dtype=np.float32)
    data_out['redshifts']    = np.zeros( (n_sn), dtype=np.float32)
    
    n_spec_each = np.zeros(n_sn)
    for i, idi in enumerate(np.unique(data['spectra_IDs'])):   
        ids   = np.where(data['spectra_IDs'] == idi)[0]
        n_sni = min(n_timestep, len(ids))
        n_spec_each[i] = n_sni

        for k in ['spectra', 'spectra_salt']:
            data_out[k][i,:n_sni] = data[k][ids[:n_timestep]]
        data_out['sigma'][i,:n_sni] = data['sigma'][ids[:n_timestep]]
        data_out['mask'][i,:n_sni] = 1. 
        
        data_out['times'][i,:n_sni] = data['cond_params'][ids[:n_timestep], 0, None]
        data_out['redshifts'][i]    = data['cond_params'][ids[0], 1]

        data_out['salt_params'][i,:] = data['salt_params'][ids[0]]

    data_out['times_orig'] = data_out['times'].copy()
    time_normalizer = time_normalization(data_out['times_orig'])
    data_out['times'] = time_normalizer.scale(data_out['times'])

    # Keep only SN from super novae factory
    if verbose: print("number of spectra each: ", n_spec_each)

    #if only_snf:
    #    n_spec_each = n_spec_each[sn_ind_SNF]
    #    print(n_spec_each.min(), n_spec_each.max())

    #    sn_spectra      = sn_spectra[sn_ind_SNF]
    #    sn_spectra_salt = sn_spectra_salt[sn_ind_SNF]
    #    sn_sigma        = sn_sigma[sn_ind_SNF]
    #    mask            = mask[sn_ind_SNF]
        
    #    times           = times[sn_ind_SNF]
    #    times_orig      = times_orig[sn_ind_SNF]
        
    #    salts           = salts[sn_ind_SNF]
    #    redshifts       = redshifts[sn_ind_SNF]
        
    #    sn_labels       = sn_labels[sn_ind_SNF]
        
    #    print(sn_spectra_2d.shape)
    #    n_sn = sn_spectra_2d.shape[0]


    from astropy.cosmology import WMAP9 as cosmo
#    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    data_out['luminosity_distance'] = cosmo.luminosity_distance(data_out['redshifts']).value

    # Save data to train/test files

    # Train test split
    ind_split  = int(n_sn * train_frac)
    print("Number in training set = ", ind_split)
    
    # Select train_frac for training, the rest for testing
    np.random.seed(13579)
    inds = np.arange(n_sn)
    np.random.shuffle(inds)
    
    # Split into k cross validation sets
    for kfold in range(nkfold):
        inds_k = np.roll(inds, kfold*inds.shape[0]//nkfold)
        if verbose: print("inds of kfold:", inds_k)
        
        inds_train = inds_k[:ind_split]
        inds_test  = inds_k[ind_split:]

        train_data = {}
        test_data  = {}
        for k_, v_ in data_out.items():
            if (v_.shape[0] != n_sn):
                train_data[k_] = v_
                test_data[k_] = v_
            else:
                train_data[k_] = v_[inds_train]
                test_data[k_]  = v_[inds_test]
            
        np.save('{:s}_kfold{:d}.npy'.format(train_file_head, kfold), train_data)
        np.save('{:s}_kfold{:d}.npy'.format(test_file_head, kfold), test_data)


