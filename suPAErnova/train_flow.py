#!/usr/bin/env python
# coding: utf-8
"""
This code constructs and trains the normalizing flow model,
based on the parameters specified in the configuration file, config/train.yaml.

The Autoencoder architecture is specified in models/autoencoder.py,
and the loss terms are specified in models/losses.py. 
"""

import tensorflow as tf
print('tensorflow version: ', tf.__version__)
print('devices: ', tf.config.list_physical_devices('GPU') )
import tensorflow_addons as tfa

tfk  = tf.keras
tfkl = tf.keras.layers
print("TFK Version", tfk.__version__)

# %pip install tensorflow-probability==0.9.0
import tensorflow_probability as tfp

tfb  = tfp.bijectors
tfd  = tfp.distributions
print("TFP Version", tfp.__version__)

import numpy as np
import os
import time

import tensorboard.plugins.hparams as HParams
import argparse

from utils.YParams import YParams
import utils.data_loader as data_loader

import models.flow 
import models.losses as losses
import models.loader as model_loader


def train_flow(data, params, verbose=False):
    """Train a simple MAF model for density estimation. 
    Can definitely be improved/should be later,
    as the flow does not always train well in high dimensions
    """

    if params['optimizer'].upper() == 'ADAM':
        optimizer  = tf.keras.optimizers.Adam(params['lr_flow'])
    if params['optimizer'].upper() == 'ADAMW':
        optimizer  = tfa.optimizers.AdamW(params['lr_flow'])
    else:
        print("Optimizer {:s} does not exist".format(params['optimizer']))

    # Mask training samples outside of (min_train_redshift < z < max_train_redshift) range
    dm = data_loader.get_train_mask(data, params)

    # Don't use time shift or amplitude in normalizing flow
    # Amplitude represents uncorrelated shift from peculiar velocity and/or gray instrumental effects
    # And this is the parameter we want to fit to get "cosmological distances", thus we don't want a prior on it
    istart = 2

    z_latent = tf.convert_to_tensor(data['z_latent'][dm, istart:], dtype=tf.float32)

    print('Size of training data = ', z_latent.shape)
    checkpoint_filepath = '{:s}flow_kfold{:d}_{:02d}Dlatent_nlayers{:02d}{:s}'.format(params['model_dir'],
                                                                                      params['kfold'],
                                                                                      params['latent_dim'],
                                                                                      params['nlayers'],
                                                                                      params['out_file_tail'])

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         save_freq=min(100, params['epochs_flow']))

    NFmodel, flow = models.flow.normalizing_flow(params, optimizer=optimizer)

    NFmodel.fit(x=z_latent,
              y=tf.zeros((z_latent.shape[0], 0), dtype=tf.float32),
              batch_size=params['batch_size'],
              epochs=params['epochs_flow'],
              steps_per_epoch=z_latent.shape[0]//params['batch_size'], 
              shuffle=True,
              verbose=verbose,
              callbacks=[cp_callback])

    NFmodel.trainable=False
    print('Done training flow!')

    return NFmodel, flow

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./config/train.yaml', type=str)
    parser.add_argument("--config", default='flow', type=str)
    parser.add_argument("--print_params", default=True, action='store_true')
    
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config, print_params=True)

    for il, latent_dim in enumerate(params['latent_dims']):
        print('Training model with {:d} latent dimensions'.format(latent_dim))
        params['latent_dim'] = latent_dim

        encoder, decoder, AE_params = model_loader.load_ae_models(params)

        train_data = data_loader.load_data(params['train_data_file'], print_params=params['print_params'])#, to_tensor=True)
        test_data  = data_loader.load_data(params['test_data_file'])#, to_tensor=True)

        # get latent representations from encoder
        train_data['z_latent'] = encoder((train_data['spectra'],train_data['times'], train_data['mask'])).numpy()
        test_data['z_latent']  = encoder((test_data['spectra'], test_data['times'], test_data['mask'])).numpy()

        # saved on checkpoint, so no need to save again
        NFmodel, flow = train_flow(train_data, params, verbose=True)

if __name__ == '__main__':
    main()
    
