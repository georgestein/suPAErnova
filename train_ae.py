#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
print('tensorflow version: ', tf.__version__)
print('devices: ', tf.config.list_physical_devices('GPU') )

import numpy as np

import tensorboard.plugins.hparams as HParams
import argparse

from utils.YParams import YParams
from utils.data_loader import *

import models.autoencoder
from models.losses import *

import os
import time
import sys

def train_step(model, optimizer, compute_apply_gradients_ae, epoch, nbatches, train_data):
    
    training_loss, training_loss_terms = 0, [0, 0]

    # shuffle indices each epoch for batches
    # batch feeding can be improved, but the various types of specialized masks/dropout
    # are easy to implement in this non-optimized fashion
    #tf.random.set_seed(epoch)
    #inds = tf.range(train_data['spectra'].shape[0])
    #tf.random.shuffle(inds)
    #inds = tf.reshape(inds, [-1, params['batch_size']])
    np.random.seed(epoch)
    inds = np.arange(train_data['spectra'].shape[0])
    np.random.shuffle(inds)
    inds = inds.reshape(-1, params['batch_size'])

    if model.params['train_noise']:
        # add noise during training drawn from observational unvertainty
#        noise_vary = params['noise_scale']*tf.math.abs(tf.random.normal(train_data['mask'].shape, mean=train_data['spectra'], stddev=train_data['sigma']))
        noise_vary = params['noise_scale']*np.abs(np.random.normal(0., train_data['sigma']).astype(np.float32))
    else:
        noise_vary = np.zeros(train_data['mask'].shape, dtype=np.float32)

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

    mask_vary[~train_data['dm']] = 0.
    #    mask_vary = tf.convert_to_tensor(mask_vary, dtype=tf.float32)
    if epoch == 0 :
        print("Number of training SN to use for training: ", np.sum(train_data['dm']))
    
    # loop over batches
    for batch in range(nbatches):
        inds_batch = sorted(inds[batch])
#        print(len(inds_batch), inds_batch, train_data['spectra'].shape, noise_vary.shape)
#        print(inds_batch, train_data['spectra'][[1,2,3]], noise_vary[[1,2,3]])
#        print(len(inds_batch), inds_batch, train_data['spectra'][inds_batch].shape, noise_vary[inds_batch].shape)

#        print('DEBUG', inds.dtype, inds_batch, inds_batch.shape, tf.gather(train_data['spectra'], inds_batch), noise_vary.shape)
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
        training_loss_b, training_loss_terms_b = compute_apply_gradients_ae(model, 
                                                                            train_data['spectra'][inds_batch] + noise_vary[inds_batch], 
                                                                            train_data['times'][inds_batch],
                                                                            train_data['sigma'][inds_batch],
                                                                            train_data['mask'][inds_batch] * mask_vary[inds_batch],
                                                                            train_data['luminosity_distance'][inds_batch],
                                                                            optimizer)

        training_loss += training_loss_b.numpy()
        training_loss_terms += training_loss_terms_b

    return training_loss, training_loss_terms


def test_step(model, data):  
    """Calculate test loss"""
    mask_vary = np.ones(data['mask'].shape, dtype=np.float32)
    mask_vary[~data['dm']] = 0.

    test_loss, test_loss_terms = compute_loss_ae(model, data['spectra'], data['times'], data['sigma'], data['mask']*mask_vary, data['luminosity_distance'])

    return test_loss, test_loss_terms

def train_model(train_data, test_data,
                model, optimizer=tf.keras.optimizers.Adam(1e-3)):

    compute_apply_gradients_ae = get_apply_grad_fn()

    # 176 training, so batch size can be 16, 22, or 44
    nbatches = train_data['spectra'].shape[0]//model.params['batch_size']
    
    test_every = model.params['test_every']

    train_data['dm'] = get_train_mask(train_data, model.params)
    test_data['dm'] = get_train_mask(test_data, model.params)

    '''
    train_dm = (train_data['redshifts'] > model.params['min_train_redshift']) & \
        (train_data['redshifts'] < model.params['max_train_redshift'])

    train_dm_maxlight = (train_data['times_orig'] > model.params['max_light_cut'][0]) & \
        (train_data['times_orig'] < model.params['max_light_cut'][1])

    train_dm_maxlight = np.any(train_dm_maxlight, axis=(1,2))
    train_dm = train_dm & train_dm_maxlight

    if model.params['twins_cut']:
        train_in_twins, train_dm_twins = get_twins_mask(train_data)
        train_dm = train_dm & train_dm_twins
    train_data['dm'] = train_dm 
        
    test_dm = (test_data['redshifts'] > model.params['min_train_redshift']) & \
                  (test_data['redshifts'] < model.params['max_train_redshift'])
    test_dm_maxlight = (test_data['times_orig'] > model.params['max_light_cut'][0]) & \
\
        (test_data['times_orig'] < model.params['max_light_cut'][1])

    test_dm_maxlight = np.any(test_dm_maxlight, axis=(1,2))
    test_dm = test_dm & test_dm_maxlight

    if model.params['twins_cut']:
        test_in_twins, test_dm_twins = get_twins_mask(test_data)
        test_dm = test_dm & test_dm_twins
    '''
    
    ncolumn_loss = 3
    training_loss_hist = np.zeros((model.params['epochs'], ncolumn_loss))
    test_loss_hist     = np.zeros((model.params['epochs']//test_every, ncolumn_loss))

    test_loss_min = 1.e9
    for epoch in range(model.params['epochs']):
        is_best = False
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
            if test_loss.numpy() < test_loss_min:
                print('Best test epoch so far. Saving model.')
                is_best=True
                test_loss_min=min(test_loss_min, test_loss.numpy())
                save_model(model, model.params)
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

        if params['dropout']:
            # Create new model with training=False (dropout deactivated). Copy weights to new model, and save.
            # Required as we are not using model.fit() and model.predict() due to architecture/training uniqueness.
            model_save = models.autoencoder.AutoEncoder(params, training=False)

            model_save.encoder.set_weights(model.encoder.get_weights())
            model_save.decoder.set_weights(model.decoder.get_weights())

            model_save.encoder.save(encoder_file)
            model_save.decoder.save(decoder_file)

        else:
            model.encoder.save(encoder_file)
            model.decoder.save(decoder_file)

if __name__ == '__main__':

    # Set model Architecture and training params and train
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./config/train.yaml', type=str)
    parser.add_argument("--config", default='autoencoder', type=str)
    
    args = parser.parse_args()
    
    params = YParams(os.path.abspath(args.yaml_config), args.config, print_params=True)
    
    train_data = load_data(params['train_data_file'])#, to_tensor=True) 
    test_data  = load_data(params['test_data_file'])#, to_tensor=True) 
    
    for il, latent_dim in enumerate(params['latent_dims']):

        optimizer  = tf.keras.optimizers.Adam(params['lr'])

        params['latent_dim'] = latent_dim
        tf.random.set_seed(params['seed'])

        # Create model
        AEmodel = models.autoencoder.AutoEncoder(params, training=True)
        
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
        # save_model(AEmodel, params)

