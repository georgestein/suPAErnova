#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
print('tensorflow version: ', tf.__version__)
print('devices: ', tf.config.list_physical_devices('GPU') )

import numpy as np
import os
import time

import tensorboard.plugins.hparams as HParams
import argparse

from utils.YParams import YParams
from utils.data_loader import load_data

import models.autoencoder
from models.losses import *


def train_step(model, optimizer, compute_apply_gradients_ae, epoch, nbatches, train_data):
    
    training_loss, training_loss_terms = 0, [0, 0]
    if not model.params['train_noise']: 
        noise_scale=0.

    # shuffle indices each epoch for batches
    # batch feeding can be improved, but the various types of specialized masks/dropout
    # are easy to implement in this non-optimized fashion
    np.random.seed(epoch)
    inds = np.arange(train_data['spectra'].shape[0])
    np.random.shuffle(inds)
    inds = inds.reshape(-1, params['batch_size'])

    if model.params['train_noise']:
        # add noise during training drawn from observational unvertainty
        means = np.random.normal(0., 0.005, size=(train_data['sigma'].shape[0])).astype(dtype=np.float32) #
        noise = noise_scale*np.random.normal(0., train_data['sigma']).astype(dtype=np.float32) #+ means[:, None, None]

    mask_vary = np.ones(train_data['mask'].shape, dtype=np.float32)
    if model.params['vary_mask']:
        # masks individual spectra of SN. Slow, but works
        ni = np.sum(train_data['mask'], axis=1).astype(np.int32)
        num_mask = (params['mask_vary_frac'] * ni).astype(np.int32)
        for i in range(mask_vary.shape[0]):

            nii    = ni[i][0]
            nmaski = num_mask[i][0]

            mask_vary[i,:nmaski]  = 0. 
            mask_vary[i:i+1,:nii] = mask_vary[i:i+1, np.random.rand(nii).argsort()]
            # np.take(mask_vary[i:i+1,:nii], np.random.rand(nii).argsort(), axis=1, out=mask_vary[i:i+1,:nii])

    # Mask training samples outside of (min_train_redshift < z < max_train_redshift) range
    dm_redshift = (train_data['redshifts'] > model.params['min_train_redshift']) & \
                  (train_data['redshifts'] < model.params['max_train_redshift'])
    mask_vary[~dm_redshift] = 0.

    if epoch == 0 :
        print("Number of training SN in redshift range: ", np.sum(dm_redshift))

    
    # loop over batches
    for batch in range(nbatches):
        dm_batch = inds[batch]
        training_loss_b, training_loss_terms_b = compute_apply_gradients_ae(model, 
                                            train_data['spectra'][dm_batch], # + noise[dm_batch], 
                                            train_data['times'][dm_batch],
                                            train_data['sigma'][dm_batch],
                                            train_data['mask'][dm_batch] * mask_vary[dm_batch],
                                            train_data['luminosity_distance'][dm_batch],
                                            optimizer)

        training_loss += training_loss_b.numpy()
        training_loss_terms += training_loss_terms_b

    return training_loss, training_loss_terms


def test_step(model, test_data):  
    """Calculate test loss"""
    mask_vary = np.ones(test_data['mask'].shape, dtype=np.float32)

    dm_redshift = (test_data['redshifts'] > model.params['min_train_redshift']) & \
                  (test_data['redshifts'] < model.params['max_train_redshift'])
    mask_vary[~dm_redshift] = 0.

    test_loss, test_loss_terms = compute_loss_ae(model, test_data['spectra'], test_data['times'], test_data['sigma'], test_data['mask']*mask_vary, test_data['luminosity_distance'])

    return test_loss, test_loss_terms


def train_model(train_data, test_data,
                model, optimizer=tf.keras.optimizers.Adam(1e-3)):

    compute_apply_gradients_ae = get_apply_grad_fn()

    # 176 training, so batch size can be 16, 22, or 44
    nbatches = train_data['spectra'].shape[0]//model.params['batch_size']
    
    test_every = model.params['test_every']
    
    ncolumn_loss = 3
    training_loss_hist = np.zeros((model.params['epochs'], ncolumn_loss))
    test_loss_hist     = np.zeros((model.params['epochs']//test_every, ncolumn_loss))

    for epoch in range(model.params['epochs']):
        start_time = time.time()

        training_loss, training_loss_terms = train_step(model, optimizer, compute_apply_gradients_ae, epoch, nbatches, train_data)
        
        # get average loss over batches
        training_loss_hist[epoch, 0]     = epoch 
        training_loss_hist[epoch, 1]     = training_loss/nbatches
        training_loss_hist[epoch, 2]     = training_loss_terms[0]/nbatches

        # test on test spectra
        end_time = time.time()

        if epoch % test_every == 0:
            t_epoch  = end_time-start_time

            # Calculate test loss
            test_loss, test_loss_terms = test_step(model, test_data)
            test_loss_hist[epoch//test_every, 0] = epoch 
            test_loss_hist[epoch//test_every, 1] = test_loss.numpy()
            test_loss_hist[epoch//test_every, 2] = test_loss_terms[0].numpy()

            print('\nepoch={:d}, time={:.3f}s\ntrain loss: {:.2E}\ntest loss:  {:.2E}'.format(epoch,
                                                                                     end_time-start_time,
                                                                                     training_loss/nbatches,
                                                                                     test_loss.numpy()))
#            print('test loss terms ', test_loss_hist[epoch//test_every])

    return training_loss_hist, test_loss_hist


def save_model(model, params):

    if params['savemodel']:
        fname = 'AE_kfold{:d}_{:02d}Dlatent_layers{:s}{:s}'.format(params['kfold'],
                                                                   params['latent_dim'], 
                                                                   '-'.join(str(e) for e in params['encode_dims']),
                                                                   params['out_file_tail'])

        # Save model
        encoder_file = '{:s}/{:s}{:s}'.format(params['model_dir'], 'encoder_', fname)
        decoder_file = '{:s}/{:s}{:s}'.format(params['model_dir'], 'decoder_', fname)

        save_dict = {}
        save_dict['model_dir'] = params['model_dir']
        save_dict['encoder'] = encoder_file
        save_dict['decoder'] = decoder_file
        save_dict['parameters'] = params
        np.save('{:s}{:s}.npy'.format(params['param_dir'], fname), save_dict)

        model.encoder.save(encoder_file)
        model.decoder.save(decoder_file)

        
if __name__ == '__main__':

    # Set model Architecture and training params and train
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./config/train.yaml', type=str)
    parser.add_argument("--config", default='autoencoder', type=str)
    
    args = parser.parse_args()
    
    params = YParams(os.path.abspath(args.yaml_config), args.config, print_params=True)
    
    train_data = load_data(params['train_data_file']) 
    test_data  = load_data(params['test_data_file']) 
    
    for il, latent_dim in enumerate(params['latent_dims']):

        optimizer  = tf.keras.optimizers.Adam(params['lr'])

        params['latent_dim'] = latent_dim
        tf.random.set_seed(params['seed'])

        # Create model
        AEmodel = models.autoencoder.AutoEncoder(params)
        
        # Model Summary
        if params['model_summary'] and (il == 0):
            print("Encoder Summary")
            AEmodel.encoder.summary()
            
            print("Decoder Summary")
            AEmodel.decoder.summary()

        print('Training model with {:d} latent dimensions'.format(latent_dim))

        # Train
        training_loss, test_loss = train_model(train_data, 
                                               test_data,
                                               AEmodel, 
                                               optimizer)

        # Save
        save_model(AEmodel, params)

