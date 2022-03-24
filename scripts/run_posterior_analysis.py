#!/usr/bin/env python
# coding: utf-8
"""
This code performs posterior analysis,
based on the parameters specified in the configuration file, config/posterior_analysis.yaml.

To find the maximum of the posterior (MAP) we begin LBFGS optimization from the best fit encoded value of the data, as well as additional randomly initialized points in the parameter space. We denote the MAP latent variables as the best fit parameters that maximize the posterior from these minima. From the MAP value we then run Hamiltonian Monte Carlo (HMC) to marginalize over the parameters to obtain the final best fit model parameters and their uncertainty.

The Autoencoder architecture is specified in models/autoencoder.py,
The flow architecture is specified in models/flow.py,
"""

import tensorflow as tf
print('tensorflow version: ', tf.__version__)
print('devices: ', tf.config.list_physical_devices('GPU') )

tfk  = tf.keras
tfkl = tf.keras.layers
print("TFK Version", tfk.__version__)

import tensorflow_probability as tfp

tfb  = tfp.bijectors
tfd  = tfp.distributions
print("TFP Version", tfp.__version__)

import numpy as np
import os
import time
import sys

import tensorboard.plugins.hparams as HParams
import argparse

from suPAErnova.utils.YParams import YParams
from suPAErnova.utils import data_loader
from suPAErnova.utils import calculations

from suPAErnova.models import losses
from suPAErnova.models import loader as model_loader

from suPAErnova.models import posterior
from suPAErnova.models import flows
from suPAErnova.models import posterior_analysis

#def find_MAP(model, params, verbose=False):

#def run_HMC(model, params, verbose=False):
    
#def train(PAE, params, train_data, test_data, tstrs=['train', 'test']):

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='../config/posterior_analysis.yaml', type=str)
    parser.add_argument("--config", default='posterior', type=str)
    parser.add_argument("--print_params", default=True, action='store_true')
    
    args = parser.parse_args()
    
    params = YParams(os.path.abspath(args.yaml_config), args.config, print_params=True)

    for il, latent_dim in enumerate(params['latent_dims']):
        print('Training model with {:d} latent dimensions'.format(latent_dim))
        params['latent_dim'] = latent_dim

        # Get PAE model
        PAE = model_loader.PAE(params)
    
        train_data = data_loader.load_data(os.path.join(params['PROJECT_DIR'],
                                                    params['train_data_file']),
                                           print_params=params['print_params'],
                                           set_data_min_val=params['set_data_min_val'])
        test_data  = data_loader.load_data(os.path.join(params['PROJECT_DIR'],
                                                    params['test_data_file']),
                                           set_data_min_val=params['set_data_min_val'])
        

        # Mask certain supernovae         
        train_data['mask_sn'] = data_loader.get_train_mask(train_data, params)
        test_data['mask_sn'] = data_loader.get_train_mask(test_data, params)

	# Mask certain spectra
        train_data['mask_spectra'] = data_loader.get_train_mask_spectra(train_data, params)
        test_data['mask_spectra'] = data_loader.get_train_mask_spectra(test_data, params)

        train_data['mask'] *= train_data['mask_spectra']
        test_data['mask']  *= test_data['mask_spectra']

        # Get latent representations from encoder and flow
        train_data['z_latent'] = PAE.encoder((train_data['spectra'], train_data['times'], train_data['mask'])).numpy()
        test_data['z_latent']  = PAE.encoder((test_data['spectra'], test_data['times'], test_data['mask'])).numpy()

        istart = 0
        if params['physical_latent']:
            istart = 2
        train_data['u_latent'] = PAE.flow.bijector.inverse(train_data['z_latent'][:, istart:]).numpy()
        test_data['u_latent']  = PAE.flow.bijector.inverse(test_data['z_latent'][:, istart:]).numpy()
    
        # Get reconstructions
        train_data['spectra_ae'] = PAE.decoder((train_data['z_latent'], train_data['times'], train_data['mask'])).numpy()
        test_data['spectra_ae']  = PAE.decoder((test_data['z_latent'], test_data['times'], test_data['mask'])).numpy()

        # Measure AE reconstruction uncertainty as a function of time
        dm = train_data['mask_sn']
        print(train_data['spectra'].shape, dm.shape,
              train_data['mask_sn'].shape, train_data['mask_spectra'].shape)
        train_data['sigma_ae_time'], ae_noise_t_bin_edge, train_data['sigma_ae_time_tbin_cent'] = calculations.compute_sigma_ae_time(train_data['spectra'][dm],
                                                                                                                                    train_data['spectra_ae'][dm], 
                                                                                                                                    train_data['sigma'][dm], 
                                                                                                                                    train_data['times'][dm],
                                                                                                                                     train_data['mask'][dm])
        
        dm = test_data['mask_sn']
        test_data['sigma_ae_time'], ae_noise_t_bin_edge, test_data['sigma_ae_time_tbin_cent'] = calculations.compute_sigma_ae_time(test_data['spectra'][dm],
                                                                                                     test_data['spectra_ae'][dm],
                                                                                                     test_data['sigma'][dm],
                                                                                                                                   test_data['times'][dm],
                                                                                                                                   test_data['mask'][dm])


        tstrs = ['train', 'test']
        #tstrs = ['train']
        #tstrs = ['test']
        posterior_analysis.train(PAE, params, train_data, test_data, tstrs=tstrs)

if __name__ == '__main__':

    main()
