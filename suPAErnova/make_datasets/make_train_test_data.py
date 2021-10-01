#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
from astropy.cosmology import FlatLambdaCDM

"""
Reshape data from arrays of individual spectra to (N_sn, n_timesample, n_wavelengths)

Then save a number of kfolds to individual files for training/testing models
"""

class time_normalization():
    def __init__(self, times, normed=False):
#        super(time_normalization, self).__init__()

        self.tmin = -10
        self.tmax = 40
        
        self.minmax = self.tmax - self.tmin
        
        # only normalize existing times, not null value (-100)
        self.dm = (times > -100)
        
        self.normed = normed

    def scale(self, times):
        
        if self.normed:
            times[self.dm] = times[self.dm] * self.minmax + self.tmin
            self.normed = False

        else:
            times[self.dm] = (times[self.dm] - self.tmin)/self.minmax

            times[~self.dm] = -1.
        
            self.normed = True
            
        return times

if __name__ == '__main__':
    
    data   = np.load('data/snf_data_wSALT.npy', allow_pickle=True).item()
    figdir = 'figures/'

    train_frac = 0.75
    test_frac = 1-train_frac
    nkfold = int(1./test_frac)
    print('nkfold = ', nkfold)
    min_redshift = 0.02
    
    train_file_head = 'data/train_data'
    test_file_head = 'data/test_data'

    verbose = True

    data_out = {}
    data_out['wavelengths'] = data['wavelengths']     
    data_out['names'] = data['names']
    # reshape SN by ids, and 0 pad to (n_sn, n_timemax, n_wavelength)
    # where n_timemax is the maximum number of observations of any SN in dataset
    
    n_timestep     = np.max(np.bincount(data['ID']))
    n_timestep_min = np.min(np.bincount(data['ID']))
    
    n_sn         = len(np.unique(data['ID']))
    n_wavelength = len(data['wavelengths'])

    data_out['spectra']      = np.zeros( (n_sn, n_timestep, n_wavelength), dtype=np.float32) - 1 # set missing data to -1
    data_out['spectra_salt'] = np.zeros( (n_sn, n_timestep, n_wavelength), dtype=np.float32) - 1 # set missing data to -1
    data_out['sigma']        = np.ones(  (n_sn, n_timestep, n_wavelength), dtype=np.float32) - 1
    data_out['mask']         = np.zeros( (n_sn, n_timestep, 1), dtype=np.float32)
    
    data_out['times']        = np.zeros( (n_sn, n_timestep, 1),            dtype=np.float32) - 100 # set missing data to -100
    
    n_spec_each = np.zeros(n_sn)
    for i, idi in enumerate(np.unique(data['ID'])):   
        ids   = np.where(data['ID'] == idi)[0]
        n_sni = min(n_timestep, len(ids))
        n_spec_each[i] = n_sni

        for k in ['spectra', 'spectra_salt']:
            data_out[k][i,:n_sni] = data[k][ids[:n_timestep]]
        data_out['sigma'][i,:n_sni] = data['sigma'][ids[:n_timestep]]
        data_out['mask'][i,:n_sni] = 1.         
        data_out['times'][i,:n_sni] = data['phase'][ids[:n_timestep], None]

    data.pop('wavelengths')
    data.pop('spectra')
    data.pop('spectra_salt')
    data.pop('sigma')
    data.pop('phase')

    #remove duplicate params for multiple spectra from same SN
    unique_IDs, inds = np.unique(data['ID'], return_index=True)
    for k, v in data.items():
        data_out[k] = v[inds]

    data_out['times_orig'] = data_out['times'].copy()
    time_normalizer = time_normalization(data_out['times_orig'])
    data_out['times'] = time_normalizer.scale(data_out['times'])

    print(data_out.keys())

    # Keep only SN from super novae factory
    if verbose: print("number of spectra each: ", n_spec_each)

    from astropy.cosmology import WMAP7 as cosmo
    data_out['luminosity_distance'] = cosmo.luminosity_distance(data_out['redshift']).value

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



