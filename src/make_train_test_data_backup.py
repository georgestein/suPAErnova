#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
from astropy.cosmology import FlatLambdaCDM

# Load Data
class time_normalization():
    def __init__(self, times, normed=False):
        super(time_normalization, self).__init__()

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
    
    snfile = np.load('data/sn_spectra_arrays_wsalt.npz')
    figdir = '../figures/'

    # only_snf = True
    only_snf = False

    train_frac = 0.8

    nkfold = 1

    train_file_head = 'data/trial_train_data'
    test_file_head = 'data/trial_test_data'

    verbose = True
    print(snfile.files)
    
    wavelengths       = snfile['wavelengths']
    spectra           = snfile['spectra'] 
    spectra_salt      = snfile['spectra_salt']
    spectra_sigma     = snfile['spectra_sigma']
    spectra_IDs       = snfile['spectra_IDs']
    cond_params       = snfile['cond_params']
    cond_params_label = snfile['cond_params_label']
    salt_params       = snfile['salt_params']
    salt_params_label = snfile['salt_params_label']  
    
    
    sn_labels = np.array(['CSS110918_01', 'CSS110918_02', 'CSS120424_01', 'CSS130502_01',
                          'LSQ12cyz', 'LSQ12dbr', 'LSQ12ekl', 'LSQ12fhe', 'LSQ12fmx',
                          'LSQ12fxd', 'LSQ12gdj', 'LSQ12gxj', 'LSQ12hjm', 'LSQ13abo',
                          'LSQ13aiz', 'LSQ13avx', 'LSQ13vy', 'LSQ14cnm', 'PTF09dlc',
                          'PTF09dnl', 'PTF09dnp', 'PTF09fox', 'PTF09foz', 'PTF10hmv',
                          'PTF10icb', 'PTF10mwb', 'PTF10ndc', 'PTF10nlg', 'PTF10ops',
                          'PTF10qjl', 'PTF10qjq', 'PTF10qyz', 'PTF10tce', 'PTF10ufj',
                          'PTF10wnm', 'PTF10wof', 'PTF10xyt', 'PTF10ygu', 'PTF10yux',
                          'PTF10zdk', 'PTF11bgv', 'PTF11bju', 'PTF11bnx', 'PTF11cao',
                          'PTF11drz', 'PTF11kly', 'PTF11mkx', 'PTF11pbp', 'PTF11qmo',
                          'PTF11qzq', 'PTF12dxm', 'PTF12eer', 'PTF12ena', 'PTF12evo',
                          'PTF12fuu', 'PTF12ghy', 'PTF12grk', 'PTF12hwb', 'PTF12iiq',
                          'PTF12ikt', 'PTF12izc', 'PTF12jqh', 'PTF13ajv', 'PTF13anh',
                          'PTF13asv', 'PTF13ayw', 'PTF13azs', 'SN2004ef', 'SN2004gc',
                          'SN2004gs', 'SN2005bc', 'SN2005bg', 'SN2005cf', 'SN2005cg',
                          'SN2005el', 'SN2005hc', 'SN2005hj', 'SN2005ir', 'SN2006X',
                          'SN2006cj', 'SN2006dm', 'SN2006do', 'SN2006ob', 'SN2007bc',
                          'SN2007bd', 'SN2007cq', 'SN2007kk', 'SN2007le', 'SN2007nq',
                          'SN2008ec', 'SN2009hi', 'SN2010dt', 'SN2010ex', 'SN2010kg',
                          'SN2012cu', 'SN2012fr', 'SNBOSS38', 'SNF20050624-000',
                          'SNF20050728-000', 'SNF20050728-006', 'SNF20050729-002',
                          'SNF20050821-007', 'SNF20050927-005', 'SNF20051003-004',
                          'SNF20051113-000', 'SNF20060511-014', 'SNF20060512-001',
                          'SNF20060512-002', 'SNF20060514-003', 'SNF20060521-008',
                          'SNF20060526-003', 'SNF20060530-003', 'SNF20060609-002',
                          'SNF20060618-014', 'SNF20060618-023', 'SNF20060621-012',
                          'SNF20060621-015', 'SNF20060624-019', 'SNF20060908-004',
                          'SNF20060911-014', 'SNF20060912-000', 'SNF20060912-004',
                          'SNF20060915-006', 'SNF20060916-002', 'SNF20061011-005',
                          'SNF20061020-000', 'SNF20061021-003', 'SNF20061022-005',
                          'SNF20061022-014', 'SNF20061024-000', 'SNF20061030-010',
                          'SNF20061107-027', 'SNF20061108-004', 'SNF20061111-002',
                          'SNF20070330-024', 'SNF20070331-014', 'SNF20070403-000',
                          'SNF20070403-001', 'SNF20070420-001', 'SNF20070424-003',
                          'SNF20070427-001', 'SNF20070429-000', 'SNF20070506-006',
                          'SNF20070630-006', 'SNF20070701-005', 'SNF20070712-000',
                          'SNF20070712-003', 'SNF20070714-007', 'SNF20070717-003',
                          'SNF20070725-001', 'SNF20070727-016', 'SNF20070802-000',
                          'SNF20070803-005', 'SNF20070806-026', 'SNF20070810-004',
                          'SNF20070817-003', 'SNF20070818-001', 'SNF20070820-000',
                          'SNF20070831-015', 'SNF20070902-018', 'SNF20070902-021',
                          'SNF20070903-001', 'SNF20071003-004', 'SNF20071003-016',
                          'SNF20071015-000', 'SNF20071021-000', 'SNF20071108-021',
                          'SNF20080323-009', 'SNF20080507-000', 'SNF20080510-001',
                          'SNF20080510-005', 'SNF20080512-008', 'SNF20080512-010',
                          'SNF20080514-002', 'SNF20080516-000', 'SNF20080516-022',
                          'SNF20080522-000', 'SNF20080522-011', 'SNF20080531-000',
                          'SNF20080610-000', 'SNF20080612-003', 'SNF20080614-010',
                          'SNF20080620-000', 'SNF20080623-001', 'SNF20080626-002',
                          'SNF20080707-012', 'SNF20080714-008', 'SNF20080717-000',
                          'SNF20080720-001', 'SNF20080725-004', 'SNF20080731-000',
                          'SNF20080802-006', 'SNF20080803-000', 'SNF20080806-002',
                          'SNF20080810-001', 'SNF20080815-017', 'SNF20080821-000',
                          'SNF20080822-005', 'SNF20080825-010', 'SNF20080908-000',
                          'SNF20080909-030', 'SNF20080913-031', 'SNF20080914-001',
                          'SNF20080918-000', 'SNF20080918-004', 'SNF20080919-000',
                          'SNF20080919-001', 'SNF20080919-002', 'SNF20080920-000',
                          'SNF20080926-009', 'SNIC3573', 'SNNGC0927', 'SNNGC2370',
                          'SNNGC2691', 'SNNGC4076', 'SNNGC4424', 'SNNGC6343', 'SNPGC027923',
                          'SNPGC51271', 'SNhunt46'])

    
    sn_ind_SNF = [i for i in range(sn_labels.shape[0]) if sn_labels[i].startswith('SNF')]

    # reshape SN by ids, and 0 pad to (n_sn, n_timemax, n_wavelength)
    # where n_timemax is the maximum number of observations of any SN in dataset

    n_timestep     = np.max(np.bincount(spectra_IDs))
    n_timestep_min = np.min(np.bincount(spectra_IDs))
    
    n_sn         = len(np.unique(spectra_IDs))
    n_wavelength = len(wavelengths)
    
    sn_spectra      = np.zeros( (n_sn, n_timestep, n_wavelength), dtype=np.float32) - 1 # set missing data to -1
    sn_spectra_salt = np.zeros( (n_sn, n_timestep, n_wavelength), dtype=np.float32) - 1 # set missing data to -1
    sn_sigma        = np.ones(  (n_sn, n_timestep, n_wavelength), dtype=np.float32) - 1
    mask            = np.zeros( (n_sn, n_timestep, 1), dtype=np.float32)
    
    times           = np.zeros( (n_sn, n_timestep, 1),            dtype=np.float32) - 100 # set missing data to -100
    
    salts           = np.zeros( (n_sn, salt_params.shape[1]), dtype=np.float32)
    redshifts       = np.zeros( (n_sn), dtype=np.float32)
    
    n_spec_each = np.zeros(n_sn)
    for i, idi in enumerate(np.unique(spectra_IDs)):   
        ids   = np.where(spectra_IDs == idi)[0]
        n_sni = min(n_timestep, len(ids))
        n_spec_each[i] = n_sni
        
        sn_spectra[i,:n_sni]      = spectra[ids[:n_timestep]]
        sn_spectra_salt[i,:n_sni] = spectra_salt[ids[:n_timestep]]
        sn_sigma[i,:n_sni]        = spectra_sigma[ids[:n_timestep]]
        mask[i,:n_sni]            = 1. 
        
        times[i,:n_sni]    = cond_params[ids[:n_timestep], 0, None]
        redshifts[i]       = cond_params[ids[0], 1]
        salts[i,:]         = salt_params[ids[0]]

    times_orig = times.copy()

    time_normalizer = time_normalization(times_orig)
    times = time_normalizer.scale(times)

    # Keep only SN from super novae factory
    if verbose: print("number of spectra each: ", n_spec_each)

    if only_snf:
        n_spec_each = n_spec_each[sn_ind_SNF]
        print(n_spec_each.min(), n_spec_each.max())

        sn_spectra      = sn_spectra[sn_ind_SNF]
        sn_spectra_salt = sn_spectra_salt[sn_ind_SNF]
        sn_sigma        = sn_sigma[sn_ind_SNF]
        mask            = mask[sn_ind_SNF]
        
        times           = times[sn_ind_SNF]
        times_orig      = times_orig[sn_ind_SNF]
        
        salts           = salts[sn_ind_SNF]
        redshifts       = redshifts[sn_ind_SNF]
        
        sn_labels       = sn_labels[sn_ind_SNF]
        
        print(sn_spectra_2d.shape)
        n_sn = sn_spectra_2d.shape[0]


    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    luminosity_distance = cosmo.luminosity_distance(redshifts).value

    # Save data to train/test files

    # Train test split
    ind_split  = int(n_sn * train_frac)

    # Select train_frac for training, the rest for testing
    np.random.seed(13579)
    inds = np.arange(n_sn)
    np.random.shuffle(inds)
    
    # Split into k cross validation sets
    for k in range(nkfold):
        inds_k = np.roll(inds, k*inds.shape[0]//nkfold)
        if verbose: print("inds of kfold:", inds_k)
        
        inds_train = inds_k[:ind_split]
        inds_test  = inds_k[ind_split:]
        
        
        np.savez('{:s}_kfold{:d}.npz'.format(train_file_head, k),       
                 wavelengths       = wavelengths,
                 salt_params_label = salt_params_label,
                 luminosity_distance = luminosity_distance[inds_train].astype(np.float32),
                 
                 spectra      = sn_spectra[inds_train].astype(np.float32),
                 spectra_salt = sn_spectra_salt[inds_train].astype(np.float32),
                 sigma        = sn_sigma[inds_train].astype(np.float32),
                 mask         = mask[inds_train].astype(np.float32),
                 times        = times[inds_train].astype(np.float32),
                 times_orig   = times_orig[inds_train].astype(np.float32),
                 salt_params  = salts[inds_train].astype(np.float32),
                 redshifts    = redshifts[inds_train].astype(np.float32),
                 labels       = sn_labels[inds_train])
        
        np.savez('{:s}_kfold{:d}.npz'.format(test_file_head, k), 
                 wavelengths       = wavelengths,
                 salt_params_label = salt_params_label,
                 luminosity_distance = luminosity_distance[inds_test].astype(np.float32),
                 
                 spectra      = sn_spectra[inds_test].astype(np.float32),
                 spectra_salt = sn_spectra_salt[inds_test].astype(np.float32),
                 sigma        = sn_sigma[inds_test].astype(np.float32),
                 mask         = mask[inds_test].astype(np.float32),
                 times        = times[inds_test].astype(np.float32),
                 times_orig   = times_orig[inds_test].astype(np.float32),
                 salt_params  = salts[inds_test].astype(np.float32),
                 redshifts    = redshifts[inds_test].astype(np.float32),
                 labels       = sn_labels[inds_test])


