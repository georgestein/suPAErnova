#!/usr/bin/env python
# coding: utf-8
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

from utils.YParams import YParams
from utils.data_loader import *
from utils.calculations import *

from models.losses import *
from models.posterior import *

import models.loader 
import models.flow

def find_MAP(model, params, verbose=False):

    for ichain in range(params['nchains']):
        # Run chains from different starting points, and keep the one with lowest negative log likelihood
        print('\n\nRunning chain: {:d}\n\n'.format(ichain))

        if ichain==0:
            initial_position = model.MAP_ini.numpy()
            if params['train_amplitude']:
                initial_position = np.c_[initial_position, model.amplitude_ini.numpy()]
            if params['train_dtime']:
                initial_position = np.c_[initial_position, model.dtime_ini.numpy()]

        else:
            initial_position = model.get_latent_prior().sample(model.nsamples).numpy()
            if params['train_amplitude']:
                initial_position = np.c_[initial_position, model.get_amplitude_prior().sample(model.nsamples).numpy()]
            if params['train_dtime']:
                initial_position = np.c_[initial_position, model.get_dtime_prior().sample(model.nsamples).numpy()]

            #print(initial_position)

        def func_bfgs(x):
            return tfp.math.value_and_gradient(
                lambda x: -1./100*model(x),
                x)
    
        results =  tfp.optimizer.bfgs_minimize(func_bfgs,
                                               initial_position=initial_position,
                                               tolerance=1e-4,
                                               max_iterations=params['max_iterations'],
                                               max_line_search_iterations=params['max_line_search_iterations'])

        if verbose: tf.print(results.converged)
        if verbose: tf.print("Function evaluations: {0}".format(results.num_objective_evaluations))
        if verbose: tf.print("Function minimum: {0}".format(results.objective_value))

        if ichain == 0:
            # initialize amplitude and dtime
            amplitude = model.amplitude_ini.numpy()
            amplitude_ini = model.amplitude_ini.numpy()

            dtime = model.dtime_ini.numpy()
            dtime_ini = model.dtime_ini.numpy()

            chain_min = np.zeros(model.nsamples)
            # Check convergence properties
            converged = np.array(results.converged)
            # Check that the argmin is close to the actual value.
            num_evaluations = [results.num_objective_evaluations]

            negative_log_likelihood = np.array(results.objective_value)

            inv_hessian = np.array(results.inverse_hessian_estimate)
        
            if params['train_amplitude']:
                amplitude = np.array(results.position)[:, params['latent_dim']]
            if params['train_dtime']:
                dtime = np.array(results.position)[:, -1]
                
            MAP = np.array(results.position)[:, :params['latent_dim']]

            MAP_ini = initial_position[:, :params['latent_dim']]
            if params['train_amplitude']:
                amplitude_ini = initial_position[:, params['latent_dim']]
            if params['train_dtime']:
                dtime_ini = initial_position[:, -1]
                
        else:
            dm = results.objective_value < negative_log_likelihood

            chain_min[dm] = ichain
            # Check convergence properties
            converged[dm] = np.array(results.converged)[dm]
            # Check that the argmin is close to the actual value.
            num_evaluations += results.num_objective_evaluations

            negative_log_likelihood[dm] = np.array(results.objective_value)[dm]

            inv_hessian[dm] = np.array(results.inverse_hessian_estimate)[dm]
        
            if params['train_amplitude']:
                amplitude[dm] = np.array(results.position)[dm, params['latent_dim']]
                amplitude_ini[dm] = initial_position[dm, params['latent_dim']]
            if params['train_dtime']:
                dtime[dm] = np.array(results.position)[dm, -1]
                dtime_ini[dm] = initial_position[dm, -1]

            MAP[dm] = np.array(results.position)[dm, :params['latent_dim']]
            MAP_ini[dm] = initial_position[dm, :params['latent_dim']]

    print('Min found on chain {0}'.format(chain_min))
    model.chain_min = chain_min
    # Check convergence properties
    model.converged = converged

    # Check that the argmin is close to the actual value.
    model.num_evaluations = num_evaluations
    
    model.negative_log_likelihood = negative_log_likelihood
    
    model.inv_hessian = inv_hessian
    
    model.amplitude = amplitude
    model.dtime = dtime
    model.MAP = MAP
    model.amplitude_ini = amplitude_ini
    model.dtime_ini = dtime_ini
    model.MAP_ini = MAP_ini

    return model
    
def train(PAE, params, train_data, test_data, tstrs=['train', 'test']):

    for tstr in tstrs:

        if tstr == 'train':
            data_use = train_data

        if tstr == 'test':
            data_use = test_data
        
        nsn =  data_use['spectra'].shape[0]

        batch_size = params['batch_size']
        if nsn < batch_size:
            batch_size = nsn
        file_base = os.path.splitext(params['{:s}_data_file'.format(tstr)])[0]
        fout = '{:s}_posterior_{:02d}Dlatent_layers{:s}{:s}'.format(file_base,
                                                                    params['latent_dim'],
                                                                    '-'.join(str(e) for e in params['encode_dims']),
                                                                    params['out_file_tail'])
    
        training_hist = {}
        map_fits = {}
        for batch_start in np.arange(0, nsn, batch_size):
            batch_end = batch_start+batch_size
        
            print('Finding MAP of batch ', batch_start//batch_size)
            
            # Construct new data for batch
            data = {}
            data['spectra']   = data_use['spectra'][batch_start:batch_end]
            data['sigma']     = data_use['sigma'][batch_start:batch_end]
            data['times']     = data_use['times'][batch_start:batch_end]
            data['redshifts'] = data_use['redshifts'][batch_start:batch_end] #* 0. + redshift
            data['mask']      = data_use['mask'][batch_start:batch_end].copy()
            data['wavelengths'] = data_use['wavelengths']
            # convert from flux to luminosity
            # data['spectra']         = L_to_F(data['spectra'], data['redshifts'][..., None, None])

            
            # Get model
            log_posterior = LogPosterior(PAE, params, data, ae_noise_t_bin_cent, test_data['sigma_ae_time'])

            # Find MAP
            log_posterior = find_MAP(log_posterior, params, verbose=True)

            # Parameters to save
            data_map_batch = {}            

            data_map_batch['u_latent_ini'] = log_posterior.MAP_ini
            data_map_batch['amplitude_ini'] = log_posterior.amplitude_ini
            data_map_batch['dtime_ini'] = log_posterior.dtime_ini
            
            data_map_batch['chain_min'] = log_posterior.chain_min
            data_map_batch['converged'] = log_posterior.converged
            data_map_batch['num_evaluations'] = log_posterior.num_evaluations
            data_map_batch['negative_log_likelihood'] = log_posterior.negative_log_likelihood

            data_map_batch['covariance'] = log_posterior.inv_hessian


            zi = log_posterior.get_z()

            data_map_batch['spectra_map'] = log_posterior.fwd_pass().numpy()
            data_map_batch['u_latent_map'] = log_posterior.MAP
            data_map_batch['z_latent_map'] = zi

            if params['use_amplitude']:
                # overall amplitude factor learned in Autoencoder
                data_map_batch['amplitude_map']   = log_posterior.amplitude
            else:
                # overall amplitude factor added in Posterior analysis
                data_map_batch['amplitude_map']   = log_posterior.amplitude

            if params['train_dtime']:
                data_map_batch['dtime_map']   = log_posterior.dtime

            data_map_batch['logp_z_latent'] = log_posterior.flow.log_prob(zi)
            data_map_batch['logp_u_latent'] = -1./2 * np.sum(log_posterior.MAP**2, axis=1)
            data_map_batch['logJ_u_latent'] = log_posterior.flow.bijector.forward_log_det_jacobian(log_posterior.MAP, event_ndims=1).numpy()

            tf.print('evaluation stop={0}:\namplitude: {1}\ndtime {2}'.format(log_posterior.num_evaluations,
                     log_posterior.amplitude,
                     log_posterior.dtime*50))

            if batch_start == 0:
                data_map = data_map_batch.copy()
            else:
                for k in data_map_batch.keys():
                    print(k, data_map_batch[k].shape)
                    data_map[k] = np.concatenate((data_map[k], data_map_batch[k]))


        # save to disk
        dicts = [data_use, data_map]
        dict_save = {}
        for d in dicts:
            for k, v in d.items():
                dict_save[k] = v

        np.save('{:s}'.format(fout), dict_save)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./config/posterior_analysis.yaml', type=str)
    parser.add_argument("--config", default='posterior', type=str)
    parser.add_argument("--print_params", default=True, action='store_true')
    
    args = parser.parse_args()
    
    params = YParams(os.path.abspath(args.yaml_config), args.config, print_params=True)

    if params['use_amplitude']:
        params['train_amplitude'] = False

    for il, latent_dim in enumerate(params['latent_dims']):
        print('Training model with {:d} latent dimensions'.format(latent_dim))
        params['latent_dim'] = latent_dim


        # Get PAE model
        PAE = models.loader.PAE(params)
    
        train_data = load_data(params['train_data_file'], print_params=params['print_params'])
        test_data  = load_data(params['test_data_file'])
        
        # get latent representations from encoder and flow
        train_data['z_latent'] = PAE.encoder((train_data['spectra'], train_data['times'], train_data['mask'])).numpy()
        test_data['z_latent']  = PAE.encoder((test_data['spectra'], test_data['times'], test_data['mask'])).numpy()
    
        train_data['u_latent'] = PAE.flow.bijector.inverse(train_data['z_latent']).numpy()
        test_data['u_latent']  = PAE.flow.bijector.inverse(test_data['z_latent']).numpy()
    
        # get reconstructions
        train_data['spectra_ae'] = PAE.decoder((train_data['z_latent'], train_data['times'])).numpy()
        test_data['spectra_ae']  = PAE.decoder((test_data['z_latent'], test_data['times'])).numpy()


        # Measure ae reconstruction uncertainty as a function of time
        train_data['sigma_ae_time'], ae_noise_t_bin_edge, ae_noise_t_bin_cent = compute_sigma_ae_time(train_data['spectra'], 
                                                                                                      train_data['spectra_ae'],
                                                                                                      train_data['sigma'],
                                                                                                      train_data['times'])

        
        test_data['sigma_ae_time'], ae_noise_t_bin_edge, ae_noise_t_bin_cent = compute_sigma_ae_time(test_data['spectra'], 
                                                                                                     test_data['spectra_ae'],
                                                                                                     test_data['sigma'],
                                                                                                     test_data['times'])



        train(PAE, params, train_data, test_data, tstrs=['train', 'test'])
