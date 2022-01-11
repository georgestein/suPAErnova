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

from utils.YParams import YParams
import utils.data_loader as data_loader
import utils.calculations as calculations

import models.losses as losses
import models.loader as model_loader

import models.posterior
import models.loader 
import models.flow

def find_MAP(model, params, verbose=False):

    ind_amplitude = 0
    ind_dtime = 0
    if params['train_dtime']:
        ind_amplitude = 1

    for ichain in range(params['nchains']):
        # Run optimization from different starting points, and keep the one with lowest negative log likelihood
        print('\n\nRunning chain: {:d}\n\n'.format(ichain))

        if ichain==0:
            initial_position = model.MAPu_ini.numpy()
            if params['train_amplitude']:
                # add amplitude as first parameter
                initial_position = np.c_[model.amplitude_ini.numpy(), initial_position]
            if params['train_dtime']:
                # add delta time as last parameter
                initial_position = np.c_[model.dtime_ini.numpy(), initial_position]

        if ichain > 1 and ichain < 10:
            initial_position = model.get_latent_prior().sample(model.nsamples).numpy() 
            if params['train_amplitude']:

                # add amplitude as first parameter
                initial_position = np.c_[model.get_amplitude_prior().sample(model.nsamples).numpy(),
                                         initial_position]
            if params['train_dtime']:
                # add delta time as last parameter
                initial_position = np.c_[model.get_dtime_prior().sample(model.nsamples).numpy(),
                                         initial_position]
            
        if ichain >= 10 and ichain < 20:
            #initial_position = model.get_latent_prior().sample(model.nsamples).numpy()
            initial_position = model.get_latent_prior().sample(model.nsamples).numpy() * 0.
            if params['train_amplitude']:
                # replace amplitude paramater with larger variance
                #initial_position[:, 0] = model.get_amplitude_prior().sample(model.nsamples).numpy()
                Amax = 1.5
                Amin = -1.5
                dA = (Amax-Amin)/(10-1)
                A = np.zeros(initial_position.shape[0], dtype=np.float32) + Amin + (ichain-10)*dA

                # add amplitude as first parameter
                initial_position = np.c_[A, initial_position]

            if params['train_dtime']:
                initial_position = np.c_[model.get_dtime_prior().sample(model.nsamples).numpy(),
                                         initial_position]

        if ichain >= 20:
            # vary Av
            # get mean spectra in u
            initial_position = model.get_latent_prior().sample(model.nsamples).numpy() * 0.
            # transform to z
            initial_position = model.flow.bijector.forward(initial_position).numpy()

            # replace Av paramater with larger variance
            Avmax = 0.5
            Avmin = -0.5
            dA = (Avmax-Avmin)/(params['nchains']-20)
            Av = np.zeros(initial_position.shape[0], dtype=np.float32) + Avmin + (ichain-20)*dA

            initial_position[:, 0] = Av

            # transform back to u
            initial_position = model.flow.bijector.inverse(initial_position)

            # add amplitude as first parameter
            A = np.zeros(initial_position.shape[0], dtype=np.float32) 
            initial_position = np.c_[A, initial_position]

            if params['train_dtime']:
                initial_position = np.c_[model.get_dtime_prior().sample(model.nsamples).numpy(),
                                         initial_position]

        if params['train_dtime']:
            initial_position[:, ind_dtime] *= params['dtime_norm']

        def func_bfgs(x):
            return tfp.math.value_and_gradient(
                lambda x: -1./100*model(x),
                x)

        results =  tfp.optimizer.lbfgs_minimize(func_bfgs,
                                                initial_position=initial_position,
                                                tolerance=params['tolerance'],
                                                x_tolerance=params['tolerance'],
                                                max_iterations=params['max_iterations'],
                                                num_correction_pairs=1)#,
                                               #max_line_search_iterations=params['max_line_search_iterations'])

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

            if params['train_amplitude']:
                amplitude = np.array(results.position)[:, ind_amplitude]
                amplitude_ini = initial_position[:, ind_amplitude]
            if params['train_dtime']:
                dtime = np.array(results.position)[:, ind_dtime]/params['dtime_norm']
                
            MAPu = np.array(results.position)[:, model.istart_map:]
            MAPu_ini = initial_position[:, model.istart_map:]

            if params['train_dtime']:
                dtime_ini = initial_position[:, ind_dtime]
                
        else:
            dm = results.objective_value < negative_log_likelihood

            chain_min[dm] = ichain
            # Check convergence properties
            converged[dm] = np.array(results.converged)[dm]
            # Check that the argmin is close to the actual value.
            num_evaluations += results.num_objective_evaluations

            negative_log_likelihood[dm] = np.array(results.objective_value)[dm]

#            inv_hessian[dm] = np.array(results.inverse_hessian_estimate)[dm]
        
            if params['train_amplitude']:
                amplitude[dm] = np.array(results.position)[dm, ind_amplitude]
                amplitude_ini[dm] = initial_position[dm,  ind_amplitude]
            if params['train_dtime']:
                dtime[dm] = np.array(results.position)[dm, ind_dtime]/params['dtime_norm']
                dtime_ini[dm] = initial_position[dm, ind_dtime]

            MAPu[dm] = np.array(results.position)[dm, model.istart_map:]
            MAPu_ini[dm] = initial_position[dm, model.istart_map:]

    print('Min found on chain {0}'.format(chain_min))
    model.chain_min = chain_min
    # Check convergence properties
    model.converged = converged

    # Check that the argmin is close to the actual value.
    model.num_evaluations = num_evaluations
    
    model.negative_log_likelihood = negative_log_likelihood
    
#    model.inv_hessian = inv_hessian
    
    model.amplitude = amplitude
    model.dtime = dtime
    model.MAPu = MAPu
    model.amplitude_ini = amplitude_ini
    model.dtime_ini = dtime_ini
    model.MAPu_ini = MAPu_ini

    return model

def run_HMC(model, params, verbose=False):
    # Initialize the HMC transition kernel.
    # @tf.function(autograph=False)

    num_warmup_steps   = int(params['num_burnin_steps'] * 0.8)

    initial_position = tf.convert_to_tensor(model.MAPu).numpy()
    if params['train_amplitude'] or params['use_amplitude']:
        # add amplitude as first parameter
        initial_position = np.c_[tf.convert_to_tensor(model.amplitude).numpy(), initial_position]
    if params['train_dtime']:
        initial_position = np.c_[tf.convert_to_tensor(model.dtime).numpy()*params['dtime_norm'], initial_position]

    '''
    else:
        initial_position_ichain = model.get_latent_prior().sample(model.nsamples).numpy()
            if params['train_amplitude']:
                initial_position_ichain = np.c_[initial_position_ichain, model.get_amplitude_prior().sample(model.nsamples).numpy()]
            if params['train_dtime']:
                initial_position_ichain = np.c_[initial_position_ichain, model.get_dtime_prior().sample(model.nsamples).numpy()]

            initial_position = tf.concat([initial_position, initial_position_ichain], axis=0)
                
            print(initial_position.shape)
    '''    
#    step_sizes = tf.fill([initial_position.shape[0], initial_position.shape[1]], params['step_size'])
    step_sizes = tf.zeros([initial_position.shape[0], initial_position.shape[1]]) + model.z_latent_std
    print('Initial step sizes', step_sizes)

    unnormalized_posterior_log_prob = lambda *args: model(*args)

    @tf.function()
    def sample_chain(ihmc=True):
        # from https://www.tensorflow.org/probability/examples/TensorFlow_Probability_Case_Study_Covariance_Estimation
        
        if ihmc:
            # run hmc
            hmc = tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=unnormalized_posterior_log_prob,
                    num_leapfrog_steps=params['num_leapfrog_steps'], #to improve convergence
                    step_size=step_sizes)
            #         state_gradients_are_stopped=True)    
            

            kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                inner_kernel=hmc,
                num_adaptation_steps=num_warmup_steps,
                target_accept_prob=params['target_accept_rate'])
            
            # Run the chain (with burn-in).
            samples, [step_sizes_final, is_accepted] = tfp.mcmc.sample_chain(
                num_results      = params['num_samples'],
                num_burnin_steps = params['num_burnin_steps'],
                current_state    = initial_position,
                kernel           = kernel,
                #parallel_iterations = params['nchains'],
                trace_fn=lambda _, pkr: [pkr.inner_results.accepted_results.step_size,
                                         pkr.inner_results.is_accepted])
        
            return samples, step_sizes_final, is_accepted

        else:
            # just do random walk
            kernel = tfp.mcmc.RandomWalkMetropolis(target_log_prob_fn=unnormalized_posterior_log_prob)
            # Run the chain (with burn-in).
            samples, is_accepted = tfp.mcmc.sample_chain(
                num_results      = params['num_samples'],
                num_burnin_steps = params['num_burnin_steps'],
                current_state    = initial_position,
                kernel           = kernel)
                #parallel_iterations = params['nchains'])

            return samples, np.full((samples[0].shape[0],samples[0].shape[1]), True, dtype=bool)


    start = time.time()
    samples, step_sizes_final, is_accepted = sample_chain(params['ihmc'])
    samples, step_sizes_final, is_accepted = samples.numpy(), step_sizes_final.numpy(), is_accepted.numpy()
    end = time.time()

    print('{:.2f} s elapsed for {:d} samples'.format(end-start, params['num_samples']+params['num_burnin_steps']))
    print('Fraction of accepted = ', np.mean(is_accepted), np.mean(is_accepted, axis=0))

    return samples, step_sizes_final, is_accepted
    
def train(PAE, params, train_data, test_data, tstrs=['train', 'test']):

    dm = data_loader.get_train_mask(train_data, params)
    z_latent_std = np.std(train_data['z_latent'][dm], axis=0)
    u_latent_std = np.std(train_data['u_latent'][dm], axis=0)
    z_latent_std[2:] = u_latent_std
    
    for tstr in tstrs:

        if tstr == 'train':
            data_use = train_data

        if tstr == 'test':
            data_use = test_data
        
        nsn =  data_use['spectra'].shape[0]

        batch_size = params['batch_size']
        if nsn < batch_size:
            batch_size = nsn
        file_base = os.path.basename(os.path.splitext(params['{:s}_data_file'.format(tstr)])[0])
        fout = '{:s}_posterior_{:02d}Dlatent_layers{:s}{:s}'.format(file_base,
                                                                    params['latent_dim'],
                                                                    '-'.join(str(e) for e in params['encode_dims']),
                                                                    params['posterior_file_tail'])
        fout = os.path.join(params['output_dir'], fout)
    
        training_hist = {}
        tstart = time.time()
        for batch_start in np.arange(0, nsn, batch_size):
            batch_end = batch_start+batch_size
        
            print('Posterior analysis of batch ', batch_start//batch_size)
            
            # Construct new data for batch
            data = {}
            data['spectra']   = data_use['spectra'][batch_start:batch_end]#*10**(-0.4*0.25)
            data['sigma']     = data_use['sigma'][batch_start:batch_end]
            data['times']     = data_use['times'][batch_start:batch_end]
            data['redshift']  = data_use['redshift'][batch_start:batch_end] #* 0. + redshift
            data['mask']      = data_use['mask'][batch_start:batch_end]
            data['wavelengths'] = data_use['wavelengths']

            #if params['run_HMC']:
                # stack data multiple times. Each becomes a seperate chain in HMC


            # convert from flux to luminosity
            # data['spectra']         = L_to_F(data['spectra'], data['redshifts'][..., None, None])

            
            # Get model
            log_posterior = models.posterior.LogPosterior(PAE, params, data,
                                                          test_data['sigma_ae_time_tbin_cent'],
                                                          test_data['sigma_ae_time'])

            log_posterior.z_latent_std = z_latent_std

            # Parameters to save
            data_map_batch = {}            
                
            data_map_batch['u_latent_ini'] = log_posterior.MAPu_ini
            data_map_batch['amplitude_ini'] = log_posterior.amplitude_ini
            data_map_batch['dtime_ini'] = log_posterior.dtime_ini
                
            if params['find_MAP']:
                # Find MAP
                log_posterior = find_MAP(log_posterior, params, verbose=True)

                data_map_batch['chain_min'] = log_posterior.chain_min
                data_map_batch['converged'] = log_posterior.converged
                data_map_batch['num_evaluations'] = log_posterior.num_evaluations
                data_map_batch['negative_log_likelihood'] = log_posterior.negative_log_likelihood

                #            data_map_batch['covariance'] = log_posterior.inv_hessian
                zi = log_posterior.get_z()
                
                data_map_batch['spectra_map'] = log_posterior.fwd_pass().numpy()
                data_map_batch['u_latent_map'] = log_posterior.MAPu
                data_map_batch['z_latent_map'] = zi.numpy()
                
                data_map_batch['amplitude_map']   = log_posterior.amplitude

                if params['train_dtime']:
                    data_map_batch['dtime_map']   = log_posterior.dtime/params['dtime_norm']

                data_map_batch['logp_z_latent_map'] = log_posterior.flow.log_prob(log_posterior.get_z()[:, log_posterior.istart_map:])
                data_map_batch['logp_u_latent_map'] = np.log(1./np.sqrt(2*np.pi) * np.exp(-1./2 * np.sum(log_posterior.MAPu**2, axis=1)))
                data_map_batch['logJ_u_latent_map'] = log_posterior.flow.bijector.forward_log_det_jacobian(log_posterior.MAPu, event_ndims=1).numpy()
                
                tf.print('evaluation stop={0}:\namplitude: {1}\ndtime {2}'.format(log_posterior.num_evaluations,
                                                                                  log_posterior.amplitude,
                                                                                  log_posterior.dtime/params['dtime_norm']*50))


                
            if params['run_HMC']:
                samples, step_sizes_final, is_accepted = run_HMC(log_posterior, params, verbose=True)            

                z_samples = log_posterior.flow.bijector.forward(samples[:, :, log_posterior.istart_map:].reshape(-1, log_posterior.latent_dim_u)).numpy().reshape(samples.shape[0], samples.shape[1], log_posterior.latent_dim_u)
                ind_amplitude = 0
                ind_dtime = 0
                if params['train_dtime']:
                    ind_amplitude = 1
                    
                data_map_batch['u_samples'] = samples[:, :, log_posterior.istart_map:]
                if params['train_dtime']:
                    data_map_batch['dtime_samples'] = samples[:, :, ind_dtime]/params['dtime_norm']
                                
                if params['train_amplitude']:
                    data_map_batch['amplitude_samples'] = samples[:, :, ind_amplitude]
                    z_samples = np.concatenate((samples[:, :, 0:ind_amplitude+1], z_samples), axis=-1)

                data_map_batch['z_samples'] = z_samples
                data_map_batch['is_accepted'] = is_accepted
                data_map_batch['step_sizes_final'] = step_sizes_final
                print('final step sizes = ', step_sizes_final.shape, step_sizes_final[-1])
                parameters_mean = np.mean(samples, axis=0)
                parameters_std  = np.std(samples, axis=0)
                z_parameters_mean = np.mean(z_samples, axis=0)
                z_parameters_std  = np.std(z_samples, axis=0)

                log_posterior.amplitude = parameters_mean[:, ind_amplitude]
                data_map_batch['amplitude_mcmc'] = z_parameters_mean[:, ind_amplitude]
                data_map_batch['amplitude_mcmc_err'] = z_parameters_std[:, ind_amplitude]

                if params['train_dtime']:
                    data_map_batch['dtime_mcmc'] = parameters_mean[:, ind_dtime]/params['dtime_norm']
                    data_map_batch['dtime_mcmc_err'] = parameters_std[:, ind_dtime]/params['dtime_norm']
                    log_posterior.dtime = parameters_mean[:, ind_dtime]/params['dtime_norm']

                log_posterior.MAPu = parameters_mean[:, log_posterior.istart_map:]
                log_posterior.MAPz = z_parameters_mean[:, log_posterior.istart_map:]

                data_map_batch['u_latent_mcmc'] = parameters_mean[:, log_posterior.istart_map:]
                data_map_batch['u_latent_mcmc_err'] = parameters_std[:, log_posterior.istart_map:]
                data_map_batch['z_latent_mcmc'] = z_parameters_mean
                data_map_batch['z_latent_mcmc_err'] = z_parameters_std

                data_map_batch['spectra_mcmc'] = log_posterior.fwd_pass().numpy()

                print(log_posterior.MAPu)
                data_map_batch['logp_z_latent_mcmc'] = log_posterior.flow.log_prob(log_posterior.MAPz[:, log_posterior.istart_map:])
                data_map_batch['logp_u_latent_mcmc'] = np.log(1./np.sqrt(2*np.pi) * np.exp(-1./2 * np.sum(log_posterior.MAPu**2, axis=1)))
                data_map_batch['logJ_u_latent_mcmc'] = log_posterior.flow.bijector.forward_log_det_jacobian(log_posterior.MAPu, event_ndims=1).numpy()

            if batch_start == 0:
                data_map = data_map_batch.copy()
            else:
                for k in data_map_batch.keys():
                    data_map[k] = np.concatenate((data_map[k], data_map_batch[k]))

        '''
        # Get Hessian and covariance matrix at MAP values:
        log_posterior.MAP = tf.convert_to_tensor(log_posterior.MAP)
        log_posterior.amplitude = tf.convert_to_tensor(log_posterior.amplitude)
        log_posterior.dtime = tf.convert_to_tensor(log_posterior.dtime)

        trainable_params, trainable_params_label = setup_trainable_parameters(log_posterior, params)

        map_params = log_posterior.MAP
        if params['train_amplitude']:
            map_params = np.c_[map_params, log_posterior.amplitude]
        if params['train_dtime']:
            map_params = np.c_[map_params, log_posterior.dtime]

        map_params = tf.Variable(tf.convert_to_tensor(map_params))
        hess = get_hessian(log_posterior, map_params, trainable_params)
        '''
        tend = time.time()
        print('\nTraining took {:.2f} s\n'.format(tend-tstart))
        # save to disk
        dicts = [data_use, data_map]
        dict_save = {}
        for d in dicts:
            for k, v in d.items():
                dict_save[k] = v

        np.save('{:s}'.format(fout), dict_save)


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./config/posterior_analysis.yaml', type=str)
    parser.add_argument("--config", default='posterior', type=str)
    parser.add_argument("--print_params", default=True, action='store_true')
    
    args = parser.parse_args()
    
    params = YParams(os.path.abspath(args.yaml_config), args.config, print_params=True)

    for il, latent_dim in enumerate(params['latent_dims']):
        print('Training model with {:d} latent dimensions'.format(latent_dim))
        params['latent_dim'] = latent_dim

        # Get PAE model
        PAE = model_loader.PAE(params)
    
        train_data = data_loader.load_data(params['train_data_file'],
                                           print_params=params['print_params'],
                                           set_data_min_val=params['set_data_min_val'])
        test_data  = data_loader.load_data(params['test_data_file'],
                                           set_data_min_val=params['set_data_min_val'])
        

        # Mask certain supernovae         
        train_data['mask_sn'] = data_loader.get_train_mask(train_data, params)
        test_data['mask_sn'] = data_loader.get_train_mask(test_data, params)

	# Mask certain spectra
        train_data['mask_spectra'] = data_loader.get_train_mask_spectra(train_data, params)
        test_data['mask_spectra'] = data_loader.get_train_mask_spectra(test_data, params)

        train_data['mask'] *= train_data['mask_spectra']
        test_data['mask']  *= test_data['mask_spectra']

        # get latent representations from encoder and flow
        train_data['z_latent'] = PAE.encoder((train_data['spectra'], train_data['times'], train_data['mask'])).numpy()
        test_data['z_latent']  = PAE.encoder((test_data['spectra'], test_data['times'], test_data['mask'])).numpy()

        istart = 0
        if params['physical_latent']:
            istart = 2
        train_data['u_latent'] = PAE.flow.bijector.inverse(train_data['z_latent'][:, istart:]).numpy()
        test_data['u_latent']  = PAE.flow.bijector.inverse(test_data['z_latent'][:, istart:]).numpy()
    
        # get reconstructions
        train_data['spectra_ae'] = PAE.decoder((train_data['z_latent'], train_data['times'], train_data['mask'])).numpy()
        test_data['spectra_ae']  = PAE.decoder((test_data['z_latent'], test_data['times'], test_data['mask'])).numpy()

        # Measure ae reconstruction uncertainty as a function of time
        dm = data_loader.get_train_mask(train_data, params)
        train_data['sigma_ae_time'], ae_noise_t_bin_edge, train_data['sigma_ae_time_tbin_cent'] = calculations.compute_sigma_ae_time(train_data['spectra'][dm],
                                                                                                                                    train_data['spectra_ae'][dm], 
                                                                                                                                    train_data['sigma'][dm], 
                                                                                                                                    train_data['times'][dm],
                                                                                                                                     train_data['mask'][dm])
            
        dm = data_loader.get_train_mask(test_data, params)
        test_data['sigma_ae_time'], ae_noise_t_bin_edge, test_data['sigma_ae_time_tbin_cent'] = calculations.compute_sigma_ae_time(test_data['spectra'][dm], 
                                                                                                     test_data['spectra_ae'][dm],
                                                                                                     test_data['sigma'][dm],
                                                                                                                                   test_data['times'][dm],
                                                                                                                                   test_data['mask'][dm])


        #tstrs = ['test']
        #tstrs = ['train']
        tstrs = ['train', 'test']
        train(PAE, params, train_data, test_data, tstrs=tstrs)

if __name__ == '__main__':

    main()
