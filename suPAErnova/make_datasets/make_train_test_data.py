#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
from astropy.cosmology import FlatLambdaCDM
from scipy.signal import savgol_filter

"""
Reshape data from arrays of individual spectra to (N_sn, n_timesample, n_wavelengths).

Construct mask for each spectra from data/mask_info_wmin_wmax.txt

Then save a number of kfolds to individual files for training/testing models
"""
def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


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
    #print(mask_sn_name, mask_spectra_name, mask_wavelength_min, mask_wavelength_max)
    
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
    data_out['spectra_ids'] = data['spectra_ids']
    # reshape SN by ids, and 0 pad to (n_sn, n_timemax, n_wavelength)
    # where n_timemax is the maximum number of observations of any SN in dataset
    
    n_timestep     = np.max(np.bincount(data['ID']))
    n_timestep_min = np.min(np.bincount(data['ID']))
    
    n_sn         = len(np.unique(data['ID']))
    n_wavelength = len(data['wavelengths'])

    data_out['spectra']      = np.zeros( (n_sn, n_timestep, n_wavelength), dtype=np.float32) - 1 # set missing data to -1
    data_out['spectra_salt'] = np.zeros( (n_sn, n_timestep, n_wavelength), dtype=np.float32) - 1 # set missing data to -1
    data_out['sigma']        = np.ones(  (n_sn, n_timestep, n_wavelength), dtype=np.float32) - 1
    data_out['mask']         = np.full( (n_sn, n_timestep, n_wavelength), False)
    data_out['times']        = np.zeros( (n_sn, n_timestep, 1),            dtype=np.float32) - 100 # set missing data to -100
    
    n_spec_each = np.zeros(n_sn)
    for i, idi in enumerate(np.unique(data['ID'])):   
        ids   = np.where(data['ID'] == idi)[0]
        n_speci = min(n_timestep, len(ids))
        n_spec_each[i] = n_speci

        for k in ['spectra', 'spectra_salt']:
            data_out[k][i,:n_speci] = data[k][ids[:n_timestep]]
        data_out['sigma'][i,:n_speci] = data['sigma'][ids[:n_timestep]]
        #data_out['mask'][i,:n_speci] = True
        # Mask bad parts of spectrum

        for ispec in range(n_speci):

            keep_min, keep_max = data['wavelength_mask'][ids[ispec]]
            ind_keep_min = 0
            ind_keep_max = data['wavelengths'].shape[0]
            if keep_min != -1.:
                # entire spectrum was not rejected, so see if part of it was
                if keep_min > data['wavelengths'][0]:
                    ind_keep_min = find_nearest_idx(data['wavelengths'], keep_min)

                if keep_max < data['wavelengths'][0]:
                    ind_keep_max = find_nearest_idx(data['wavelengths'], keep_max)
                    
                data_out['mask'][i, ispec, ind_keep_min:ind_keep_max] = True # mark as valid spectrum 

            # Mask any huge laser lines, Na D (5674 - 5692A)
            # these are large jumps in flux, localized over a few wavelength bins
            wavelength_bin_start = find_nearest_idx(data['wavelengths'], 5000.)
            wavelength_bin_end   = find_nearest_idx(data['wavelengths'], 8000.)
            laser_width = 2 # in units of wavelength bins
            laser_height = 0.4 # fractional increase in amplitude over neighbours to be considered laser

            speci = data_out['spectra'][i, ispec, wavelength_bin_start:wavelength_bin_end]
            speci_smooth = (data_out['spectra'][i, ispec, wavelength_bin_start-laser_width:wavelength_bin_end-laser_width]
                            + data_out['spectra'][i, ispec, wavelength_bin_start+laser_width:wavelength_bin_end+laser_width])/2
            #speci_smooth = savgol_filter(speci, laser_width, 1)

            laser_mask = ((speci - speci_smooth) > laser_height)
            laser_mask = np.array([np.any(laser_mask[i-laser_width:i+laser_width]) for i in range(laser_mask.shape[0])]) # mask bins nearby laser spike as well
            
            data_out['mask'][i, ispec, wavelength_bin_start:wavelength_bin_end] = ~laser_mask
            if np.sum(laser_mask) > 0:
                print("Laser line found in")
                print(data_out['names'][ids[ispec]], data_out['spectra_ids'][ids[ispec]])
                #print(speci[laser_mask], speci_smooth[laser_mask])

        data_out['times'][i,:n_speci] = data['phase'][ids[:n_timestep], None]


    data_out['mask'] = data_out['mask'].astype(np.float32)
    
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



