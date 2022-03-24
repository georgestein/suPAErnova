# !/usr/bin/env python
# coding: utf-8
"""
This code constructs and trains the Autoencoder model,
based on the parameters specified in the configuration file, config/train.yaml.

The Autoencoder architecture is specified in models/autoencoder.py,
and the loss terms are specified in models/losses.py.
"""

import tensorflow as tf
print('tensorflow version: ', tf.__version__)
print('devices: ', tf.config.list_physical_devices('GPU') )
import tensorflow_addons as tfa

import numpy as np

import tensorboard.plugins.hparams as HParams
import argparse

import os
import time
import sys

from . import autoencoder
from . import losses
from . import loader as model_loader

#@tf.function
def train_step(model, optimizer, compute_apply_gradients_ae, epoch, nbatches, train_data):
    """Run one training step"""
    training_loss, training_loss_terms = 0., [0., 0., 0.]

    # shuffle indices each epoch for batches
    # batch feeding can be improved, but the various types of specialized masks/dropout
    # are easy to implement in this non-optimized fashion, and the dataset is small
    np.random.seed(epoch)
    inds = np.arange(train_data['spectra'].shape[0])
    np.random.shuffle(inds)
    inds = inds.reshape(-1, model.params['batch_size'])

    # Add noise during training drawn from observational uncertainty
    if model.params['train_noise']:
#        noise_vary = params['noise_scale']*tf.math.abs(tf.random.normal(train_data['mask'].shape, mean=train_data['spectra'], stddev=train_data['sigma']))
        noise_vary = model.params['noise_scale']*np.abs(np.random.normal(0., train_data['sigma']).astype(np.float32))
    else:
        noise_vary = np.zeros(train_data['mask'].shape, dtype=np.float32)

    # Mask certain spectra
    mask_vary = np.ones(train_data['mask'].shape, dtype=np.float32)
    if model.params['vary_mask']:
        # masks individual spectra of SN. Slow, but works
        ni = np.sum(train_data['mask'], axis=1).astype(np.int32)
        num_mask = (model.params['mask_vary_frac'] * ni).astype(np.int32)
        for i in range(mask_vary.shape[0]):

            nii    = ni[i][0]
            nmaski = num_mask[i][0]

            mask_vary[i,:nmaski]  = 0. 
            mask_vary[i:i+1,:nii] = mask_vary[i:i+1, np.random.rand(nii).argsort()]
            # np.take(mask_vary[i:i+1,:nii], np.random.rand(nii).argsort(), axis=1, out=mask_vary[i:i+1,:nii])

    mask_vary[~train_data['mask_sn']] = 0.
    mask_vary[~train_data['mask_spectra'][..., 0]] = 0.

    # Vary phase by observational uncertainty
    dtime = np.zeros(train_data['times'].shape[0], dtype=np.float32)
    if model.params['train_time_uncertainty']:
        # Equal sigma for all SN
        # dtime = np.random.normal(0, model.params['time_scale']/50, size=(train_data['times'].shape[0], 1, 1))

        # Sigma from SALT2 fits
        dtime = np.random.normal(0, train_data['dphase']/50)[:, None, None].astype(np.float32)
        
    if epoch == 0 :
        print("Number of training SN to use for training: ", np.sum(train_data['mask_sn']))
    
    # loop over batches
    for batch in range(nbatches):
        inds_batch = sorted(inds[batch])
        # tensorflow tensors do not support gathering tensors of indices as numpy would. Need to instead call tf.gather(tensor, indices). Unfortunately tf.tensor() makes training slower
        '''
        training_loss_b, training_loss_terms_b = compute_apply_gradients_ae(model, 
                                            tf.gather(train_data['spectra'], inds_batch) + tf.gather(noise_vary, inds_batch), 
                                            tf.gather(train_data['times'], inds_batch),
                                            tf.gather(train_data['sigma'], inds_batch),
                                            tf.gather(train_data['mask'], inds_batch) * tf.gather(mask_vary, inds_batch),
                                            tf.gather(train_data['luminosity_distance'], inds_batch),
                                            optimizer)
        '''
        training_loss_b, training_loss_terms_b = compute_apply_gradients_ae(
            model, 
            train_data['spectra'][inds_batch] + noise_vary[inds_batch], 
            train_data['times'][inds_batch] + dtime[inds_batch],
            train_data['sigma'][inds_batch],
            train_data['mask'][inds_batch] * mask_vary[inds_batch],
            optimizer,
        )

        training_loss += training_loss_b.numpy()
        for ii in range(len(training_loss_terms_b)):
            training_loss_terms[ii] += training_loss_terms_b[ii].numpy()

    training_loss_terms = [training_loss_terms[ii]/nbatches for ii in range(len(training_loss_terms_b))]
    
    return training_loss, training_loss_terms

#@tf.function
def test_step(model, data):  
    """Calculate test loss"""
    mask_vary = np.ones(data['mask'].shape, dtype=np.float32)
    mask_vary[~data['mask_sn']] = 0.
    mask_vary[~data['mask_spectra'][..., 0]] = 0.

    test_loss, test_loss_terms = losses.compute_loss_ae(
        model,
        data['spectra'],
        data['times'],
        data['sigma'],
        data['mask']*mask_vary,
    )
    
    return test_loss, test_loss_terms

def calculate_mean_parameters_batches(model, data, nbatches):
    """Calculate the mean physical latent parameters over the whole dataset"""

    inds = np.arange(data['spectra'].shape[0])
    inds = inds.reshape(-1, model.params['batch_size'])

    mask_vary = np.ones(data['mask'].shape, dtype=np.float32)
    mask_vary[~data['mask_sn']] = 0.
    mask_vary[~data['mask_spectra'][..., 0]] = 0.

    dtime_mean = 0
    amp_mean = 0
    color_mean = 0
    total_dm = 0
    for batch in range(nbatches):
        inds_batch = sorted(inds[batch])

        z = model.encode(
            data['spectra'][inds_batch],
            data['times'][inds_batch],
            data['mask'][inds_batch] * mask_vary[inds_batch],
        ).numpy()
        
        dm = mask_vary[inds_batch, 0, 0] == 1.
        dtime_mean += np.sum(z[dm, 0])
        amp_mean   += np.sum(z[dm, 1])
        color_mean += np.sum(z[dm, 2])

    dtime_mean /= data['mask_sn'].sum()
    dtime_mean = np.float32(dtime_mean)

    amp_mean /= data['mask_sn'].sum()
    amp_mean = np.float32(amp_mean)

    color_mean /= data['mask_sn'].sum()
    color_mean = np.float32(color_mean)

    return dtime_mean, amp_mean, color_mean

def train_model(train_data, val_data, test_data, model):
    """
    Train model.
    """

    compute_apply_gradients_ae = losses.get_apply_grad_fn()

    lr = model.params['lr']
    if model.params['train_stage'] == model.params['latent_dim']+2:
        lr = model.params['lr_deltat']
    lr_ini = lr
    
    if model.params['scheduler'].upper()=='EXPONENTIAL':
        # Set up learning rate scheduler
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=model.params['lr_decay_steps'],
            decay_rate=model.params['lr_decay_rate'],
        )
        
    # Set up optimizer
    if model.params['optimizer'].upper() == 'ADAM':
        optimizer  = tf.keras.optimizers.Adam(learning_rate=lr)
    elif model.params['optimizer'].upper() == 'ADAMW':

        wd_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=model.params['weight_decay_rate'],
            decay_steps=model.params['lr_decay_steps'],
            decay_rate=model.params['lr_decay_rate'],
	)

        optimizer  = tfa.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=lambda: None,
        )
        optimizer.weight_decay = lambda: wd_schedule(optimizer.iterations)
        
    elif model.params['optimizer'].upper() == 'SGD':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr,
            momentum=0.9,
        )
    else:
        print("Optimizer {:s} does not exist".format(params['optimizer']))

    
    nbatches = train_data['spectra'].shape[0]//model.params['batch_size']
    
    ncolumn_loss = 4
    train_loss_hist = np.zeros((model.params['epochs'], ncolumn_loss))
    val_loss_hist   = np.zeros((model.params['epochs']//model.params['val_every'], ncolumn_loss))
    test_loss_hist  = np.zeros((model.params['epochs']//model.params['val_every'], ncolumn_loss))

    val_loss_min = 1.e9
    val_iteration = 0
    val_iteration_best = 0 
    for epoch in range(model.params['epochs']):

        is_best = False
        start_time = time.time()

        training_loss, training_loss_terms = train_step(
            model,
            optimizer,
            compute_apply_gradients_ae,
            epoch,
            nbatches,
            train_data,
        )
        
        # get average loss over batches
        train_loss_hist[epoch, 0] = epoch 
        train_loss_hist[epoch, 1] = training_loss_terms[0]
        train_loss_hist[epoch, 2] = training_loss_terms[1]
        train_loss_hist[epoch, 3] = training_loss_terms[2]

        # test on test spectra
        end_time = time.time()

        if epoch % model.params['val_every'] == 0:
            t_epoch  = end_time-start_time
            
            # Calculate test loss
            val_loss, val_loss_terms = test_step(model, val_data)
            val_loss_hist[val_iteration, 0] = epoch 
            val_loss_hist[val_iteration, 1] = val_loss_terms[0].numpy()
            val_loss_hist[val_iteration, 2] = val_loss_terms[1].numpy()
            val_loss_hist[val_iteration, 3] = val_loss_terms[2].numpy()

            # Calculate test loss
            test_loss, test_loss_terms = test_step(model, test_data)
            test_loss_hist[val_iteration, 0] = epoch 
            test_loss_hist[val_iteration, 1] = test_loss_terms[0].numpy()
            test_loss_hist[val_iteration, 2] = test_loss_terms[1].numpy()
            test_loss_hist[val_iteration, 3] = test_loss_terms[2].numpy()

            print('\nepoch={:d}, time={:.3f}s\n (total_loss, loss_recon, loss_cov)\ntrain loss: {:.2E} {:.2E} {:.2E}\nval loss: {:.2E} {:.2E} {:.2E}\ntest loss: {:.2E} {:.2E} {:.2E}'.format(
                epoch,
                end_time-start_time,
                training_loss_terms[0],
                training_loss_terms[1],
                training_loss_terms[2],
                val_loss_terms[0],
                val_loss_terms[1],
                val_loss_terms[2],
                test_loss_terms[0],
                test_loss_terms[1],
                test_loss_terms[2],
            ))
            
            if model.params['scheduler'].upper() == 'EXPONENTIAL':
                print("Learning rate is currently: {:.6f}".format(optimizer._decayed_lr('float32').numpy()))

            if model.params['optimizer'].upper() == 'ADAMW':
                print("Weight decay is currently: {:.6f}".format(optimizer.weight_decay))
                
            previous_val_decrease = val_iteration - val_iteration_best # number of validation iterations since last loss decrease

            if val_loss.numpy() < val_loss_min:
                print('Best validation epoch so far. Saving model.')
                is_best = True
                val_iteration_best = val_iteration
                val_loss_min = min(val_loss_min, val_loss.numpy())
                save_model(model, model.params, train_data, nbatches, is_best=is_best)

            val_iteration += 1
            
        # Save model at last epoch
        if epoch == model.params['epochs']-1:
            save_model(model, model.params, train_data, nbatches)
            
    return train_loss_hist, val_loss_hist, test_loss_hist


def save_model(model, params, data, nbatches, is_best=False):

    if params['savemodel']:

        fname = 'AE_kfold{:d}_{:02d}Dlatent_layers{:s}_{:s}'.format(params['kfold'],
                                                                   params['latent_dim'], 
                                                                   '-'.join(str(e) for e in params['encode_dims']),
                                                                   params['out_file_tail'])

        if is_best:
            fname += '_best'
            
        # Save model
        encoder_file = os.path.join(params['PROJECT_DIR'], params['MODEL_DIR'], f"encoder_{fname}")
        decoder_file = os.path.join(params['PROJECT_DIR'], params['MODEL_DIR'], f"decoder_{fname}")

        save_dict = {}
        save_dict['MODEL_DIR'] = params['MODEL_DIR']
        save_dict['encoder'] = encoder_file
        save_dict['decoder'] = decoder_file
        save_dict['parameters'] = params
        np.save(os.path.join(params['PROJECT_DIR'],
                             params['PARAM_DIR'],
                             f"{fname}"),
                save_dict)

        # Create new model with training=False (dropout deactivated). Copy weights to new model, and save.
        # Required as we are not using model.fit() and model.predict() due to architecture/training uniqueness.
        if model.params['physical_latent']:
            # create new model without normalization on amplitude
            # use this model to calculate mean amplitude
            model_save = autoencoder.AutoEncoder(params, training=False)
            model_save.encoder.set_weights(model.encoder.get_weights())
            mean_dtime, mean_amplitude, mean_color = calculate_mean_parameters_batches(model_save, data, nbatches)
            
            # create new model with mean normalization on amplitude
            model_save = autoencoder.AutoEncoder(params, training=False,
                                                        bn_moving_means=[mean_dtime,
                                                                         mean_amplitude,
                                                                         mean_color])
            
            model_save.encoder.set_weights(model.encoder.get_weights())
            mean_dtime, mean_amplitude, mean_color = calculate_mean_parameters_batches(model_save, data, nbatches)
            
        else:
            model_save = autoencoder.AutoEncoder(params, training=False)
            model_save.encoder.set_weights(model.encoder.get_weights())    

        model_save.decoder.set_weights(model.decoder.get_weights())

        model_save.encoder.save(encoder_file)
        model_save.decoder.save(decoder_file)

