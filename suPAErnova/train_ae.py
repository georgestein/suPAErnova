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

from utils.YParams import YParams
import utils.data_loader as data_loader

import models.autoencoder
import models.losses as losses
import models.loader as model_loader

import os
import time
import sys

def train_step(model, optimizer, compute_apply_gradients_ae, epoch, nbatches, train_data):
    """Run one training step"""
    training_loss, training_loss_terms = 0., [0., 0., 0.]

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
    inds = inds.reshape(-1, model.params['batch_size'])

    if model.params['train_noise']:
        # add noise during training drawn from observational uncertainty
#        noise_vary = params['noise_scale']*tf.math.abs(tf.random.normal(train_data['mask'].shape, mean=train_data['spectra'], stddev=train_data['sigma']))
        noise_vary = model.params['noise_scale']*np.abs(np.random.normal(0., train_data['sigma']).astype(np.float32))
    else:
        noise_vary = np.zeros(train_data['mask'].shape, dtype=np.float32)

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
    mask_vary[~train_data['mask_spectra']] = 0.

    dtime = np.zeros(train_data['times'].shape[0], dtype=np.float32)
    if model.params['train_time_uncertainty']:
        # equal sigma for all SN
        # dtime = np.random.normal(0, model.params['time_scale']/50, size=(train_data['times'].shape[0], 1, 1))
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
        training_loss_b, training_loss_terms_b = compute_apply_gradients_ae(model, 
                                                                            train_data['spectra'][inds_batch] + noise_vary[inds_batch], 
                                                                            train_data['times'][inds_batch] + dtime[inds_batch],
                                                                            train_data['sigma'][inds_batch],
                                                                            train_data['mask'][inds_batch] * mask_vary[inds_batch],
                                                                            optimizer)

        training_loss += training_loss_b.numpy()
        for ii in range(len(training_loss_terms_b)):
            training_loss_terms[ii] += training_loss_terms_b[ii].numpy()

    training_loss_terms = [training_loss_terms[ii]/nbatches for ii in range(len(training_loss_terms_b))]
    
    return training_loss, training_loss_terms

def test_step(model, data):  
    """Calculate test loss"""
    mask_vary = np.ones(data['mask'].shape, dtype=np.float32)
    mask_vary[~data['mask_sn']] = 0.
    mask_vary[~data['mask_spectra']] = 0.

    test_loss, test_loss_terms = losses.compute_loss_ae(model, data['spectra'], data['times'], data['sigma'], data['mask']*mask_vary)

    return test_loss, test_loss_terms

def calculate_mean_parameters_batches(model, data, nbatches):
    """Calculate the mean physical latent parameters over the whole dataset"""

    inds = np.arange(data['spectra'].shape[0])
    inds = inds.reshape(-1, model.params['batch_size'])

    mask_vary = np.ones(data['mask'].shape, dtype=np.float32)
    mask_vary[~data['mask_sn']] = 0.
    mask_vary[~data['mask_spectra']] = 0.

    dtime_mean = 0
    amp_mean = 0
    color_mean = 0
    total_dm = 0
    for batch in range(nbatches):
        inds_batch = sorted(inds[batch])

        z = model.encode(data['spectra'][inds_batch],
                          data['times'][inds_batch],
                          data['mask'][inds_batch] * mask_vary[inds_batch]).numpy()

        
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

def train_model(train_data, test_data, model):
    """
    Train model.
    """

    compute_apply_gradients_ae = losses.get_apply_grad_fn()

    lr = model.params['lr']
    if model.params['train_stage'] == model.params['latent_dim']+2:
        lr = model.params['lr_deltat']
        
    if model.params['optimizer'].upper() == 'ADAM':
        optimizer  = tf.keras.optimizers.Adam(learning_rate=lr)
    elif model.params['optimizer'].upper() == 'ADAMW':
        optimizer  = tfa.optimizers.AdamW(learning_rate=lr)
    elif model.params['optimizer'].upper() == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    else:
        print("Optimizer {:s} does not exist".format(params['optimizer']))
            
    nbatches = train_data['spectra'].shape[0]//model.params['batch_size']
    
    test_every = model.params['test_every']

    # Mask certain supernovae
    train_data['mask_sn'] = data_loader.get_train_mask(train_data, model.params)
    test_data['mask_sn'] = data_loader.get_train_mask(test_data, model.params)

    # Mask certain spectra
    train_data['mask_spectra'] = data_loader.get_train_mask_spectra(train_data, model.params)
    test_data['mask_spectra'] = data_loader.get_train_mask_spectra(test_data, model.params)

    ncolumn_loss = 4
    training_loss_hist = np.zeros((model.params['epochs'], ncolumn_loss))
    test_loss_hist     = np.zeros((model.params['epochs']//test_every, ncolumn_loss))

    test_loss_min = 1.e9
    test_iteration = 0
    test_iteration_best = 0 
    for epoch in range(model.params['epochs']):
        is_best = False
        start_time = time.time()

        training_loss, training_loss_terms = train_step(model, optimizer, compute_apply_gradients_ae, epoch, nbatches, train_data)
        
        # get average loss over batches
        training_loss_hist[epoch, 0] = epoch 
        training_loss_hist[epoch, 1] = training_loss_terms[0]
        training_loss_hist[epoch, 2] = training_loss_terms[1]
        training_loss_hist[epoch, 3] = training_loss_terms[2]

        # test on test spectra
        end_time = time.time()

        if epoch % test_every == 0:
            t_epoch  = end_time-start_time

            # Calculate test loss
            test_loss, test_loss_terms = test_step(model, test_data)
            test_loss_hist[test_iteration, 0] = epoch 
            test_loss_hist[test_iteration, 1] = test_loss_terms[0].numpy()
            test_loss_hist[test_iteration, 2] = test_loss_terms[1].numpy()
            test_loss_hist[test_iteration, 3] = test_loss_terms[2].numpy()

            print('\nepoch={:d}, time={:.3f}s\ntrain loss: {:.2E} {:.2E} {:.2E}\ntest loss: {:.2E} {:.2E} {:.2E}'.format(epoch,
                                                                                             end_time-start_time,
                                                                                             training_loss_terms[0],
                                                                                             training_loss_terms[1],
                                                                                             training_loss_terms[2],
                                                                                             test_loss_terms[0],
                                                                                             test_loss_terms[1],
                                                                                             test_loss_terms[2]))

            previous_test_decrease = test_iteration - test_iteration_best # number of test iterations since last loss decrease

            if test_loss.numpy() < test_loss_min:
                print('Best test epoch so far. Saving model.')
                is_best = True
                test_iteration_best = test_iteration
                test_loss_min = min(test_loss_min, test_loss.numpy())
                save_model(model, model.params, train_data, nbatches, is_best=is_best)

            test_iteration += 1
            
        # Save model at last epoch
        if epoch == model.params['epochs']-1:
            save_model(model, model.params, train_data, nbatches)
            
    return training_loss_hist, test_loss_hist


def save_model(model, params, data, nbatches, is_best=False):

    if params['savemodel']:

        fname = 'AE_kfold{:d}_{:02d}Dlatent_layers{:s}{:s}'.format(params['kfold'],
                                                                   params['latent_dim'], 
                                                                   '-'.join(str(e) for e in params['encode_dims']),
                                                                   params['out_file_tail'])

        if is_best:
            fname += '_best'
            
        # Save model
        encoder_file = '{:s}/{:s}{:s}'.format(params['model_dir'], 'encoder_', fname)
        decoder_file = '{:s}/{:s}{:s}'.format(params['model_dir'], 'decoder_', fname)

        save_dict = {}
        save_dict['model_dir'] = params['model_dir']
        save_dict['encoder'] = encoder_file
        save_dict['decoder'] = decoder_file
        save_dict['parameters'] = params
        np.save('{:s}{:s}.npy'.format(params['param_dir'], fname), save_dict)

        # Create new model with training=False (dropout deactivated). Copy weights to new model, and save.
        # Required as we are not using model.fit() and model.predict() due to architecture/training uniqueness.

        if model.params['physical_latent']:
            # create new model without normalization on amplitude
            # use this model to calculate mean amplitude
            model_save = models.autoencoder.AutoEncoder(params, training=False)
            model_save.encoder.set_weights(model.encoder.get_weights())
            mean_dtime, mean_amplitude, mean_color = calculate_mean_parameters_batches(model_save, data, nbatches)
            
            # create new model with mean normalization on amplitude
            model_save = models.autoencoder.AutoEncoder(params, training=False,
                                                        bn_moving_means=[mean_dtime,
                                                                         mean_amplitude,
                                                                         mean_color])
            
            model_save.encoder.set_weights(model.encoder.get_weights())
            mean_dtime, mean_amplitude, mean_color = calculate_mean_parameters_batches(model_save, data, nbatches)
            # print('DEBUG ', mean_dtime, mean_amplitude, mean_color)
            
        else:
            model_save = models.autoencoder.AutoEncoder(params, training=False)
            model_save.encoder.set_weights(model.encoder.get_weights())    

        model_save.decoder.set_weights(model.decoder.get_weights())

        model_save.encoder.save(encoder_file)
        model_save.decoder.save(decoder_file)

def main():
    
    # Set model Architecture and training params and train
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./config/train.yaml', type=str)
    parser.add_argument("--config", default='pae', type=str)
    
    args = parser.parse_args()
    
    params = YParams(os.path.abspath(args.yaml_config), args.config, print_params=True)
    epochs_initial = params['epochs']

    train_data = data_loader.load_data(params['train_data_file'],
                                       set_data_min_val=params['set_data_min_val'])
    test_data  = data_loader.load_data(params['test_data_file'],
                                       set_data_min_val=params['set_data_min_val'])

    for il, latent_dim in enumerate(params['latent_dims']):

        params['latent_dim'] = latent_dim
        params['num_training_stages'] = latent_dim + 3
        params['train_stage'] = 0
        
        if latent_dim == 0:
            # Model parameters are (\Delta t, \Delta m, \Delta A_v)
            # train \Delta m and \Delta A_v first. Then \Delta t
            params['train_stage'] = 1
            params['num_training_stages'] = 2
            
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
                                               AEmodel)

        params['train_stage'] += 1
        if not params['train_latent_individual']:
            params['train_stage'] += params['latent_dim'] - 1

        while params['train_stage'] < params['num_training_stages']:
            
            print('Running training stage ', params['train_stage'])

            epochs_initial = params['epochs']

            AEmodel_second =  models.autoencoder.AutoEncoder(params, training=True)
            if params['train_stage'] <  params['num_training_stages'] - 2: 
                AEmodel_second.params['epochs'] = params['epochs_latent']
            if params['train_stage'] >= params['num_training_stages'] - 2: # add in delta mag                     
                AEmodel_second.params['epochs'] = params['epochs_final']
            #AEmodel_second.params['epochs'] = int(epochs_initial * (params['train_stage']+1))

            # Load best checkpoint from step 0 training
            encoder, decoder, AE_params = model_loader.load_ae_models(params)
            #for il, layer in enumerate(decoder.layers):
            #    print(il, layer, layer.get_weights())
            #    print("weights:", len(layer.weights))
            #    print("trainable_weights:", len(layer.trainable_weights))
            #    print("non_trainable_weights:", len(layer.non_trainable_weights))
                
            final_dense_layer = len(params['encode_dims']) + 3

            final_layer_weights = encoder.layers[final_dense_layer].get_weights()[0]
            final_layer_weights_init =  AEmodel_second.encoder.layers[final_dense_layer].get_weights()[0]

            if params['train_stage'] <= params['latent_dim']: # add in z_1, ..., z_n
                idim = 2 + params['train_stage']
                final_layer_weights[:, idim] = final_layer_weights_init[:, idim]/100

            if params['train_stage'] == params['num_training_stages'] - 2: # add in delta mag
                final_layer_weights[:, 1] = final_layer_weights_init[:, 1]/100
                if not params['train_latent_individual']:
                    final_layer_weights[:, 3:] = final_layer_weights_init[:, 3:]/100
                    
            if params['train_stage'] == params['num_training_stages'] - 1: # add in delta t
                final_layer_weights[:, 0] = final_layer_weights_init[:, 0]/1000
            #final_layer_weights[:, :iend] *= 0.
            #iend = 3 - params['train_stage']
            #final_layer_weights[:, :iend] = final_layer_weights_init[:, :iend]/100

            encoder.layers[final_dense_layer].set_weights([final_layer_weights])

            #print('encoder before ', params['train_stage'], encoder.layers[final_dense_layer].get_weights()[0])
            #print('encoder reinit before ', params['train_stage'], final_layer_weights)

            #print('decoder before ', params['train_stage'], decoder.layers[8].get_weights()[0])

            AEmodel_second.encoder.set_weights(encoder.get_weights())
            AEmodel_second.decoder.set_weights(decoder.get_weights())

            training_loss, test_loss = train_model(train_data, 
                                                   test_data,
                                                   AEmodel_second)
            params['train_stage'] += 1
            #print('encoder after ', params['train_stage'], AEmodel_second.encoder.layers[final_dense_layer].get_weights()[0])
            #print('decoder after ', params['train_stage'], AEmodel_second.decoder.layers[8].get_weights()[0])
        # Save
        # save_model(AEmodel, params)

if __name__ == '__main__':
    
    main()
