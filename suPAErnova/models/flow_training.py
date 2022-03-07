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

from . import flows 
from . import losses
from . import loader as model_loader


def train_flow(train_data, test_data, params):
    """Train a simple MAF model for density estimation. 
    Can definitely be improved/should be later,
    as the flow does not always train well in high dimensions
    """

    optimizer  = tf.keras.optimizers.Adam(params['lr_flow'])

    # Don't use time shift or amplitude in normalizing flow
    # Amplitude represents uncorrelated shift from peculiar velocity and/or gray instrumental effects
    # And this is the parameter we want to fit to get "cosmological distances", thus we don't want a prior on it
    if params['use_extrinsic_params']:
        istart = 2
    else:
        istart = 3
        
    z_latent = tf.convert_to_tensor(train_data['z_latent'][:, istart:], dtype=tf.float32)

    print('Size of training data = ', z_latent.shape)
    layers_str = '-'.join(str(e) for e in params['encode_dims'])
    checkpoint_filepath = (f"{params['MODEL_DIR']}flow_kfold{params['kfold']}_{params['latent_dim']:02d}Dlatent_"
                           + f"layers{layers_str}_nlayers{params['nlayers']:02d}_{params['out_file_tail']}")

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            params['PROJECT_DIR'],
            checkpoint_filepath,
        ),
        save_weights_only=True,
        verbose=params['verbose'],
        save_freq=min(params['checkpoint_flow_every'], params['epochs_flow']),
    )

    earlystopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=params['patience'],
    )

    NFmodel, flow = flows.normalizing_flow(params, optimizer=optimizer)

    NFmodel.fit(
        x=z_latent,
        y=tf.zeros((z_latent.shape[0], 0), dtype=tf.float32),
        validation_split=params['val_frac_flow'],
        batch_size=params['batch_size'],
        epochs=params['epochs_flow'],
        steps_per_epoch=z_latent.shape[0]//params['batch_size'], 
        shuffle=True,
        verbose=params['verbose'],
        callbacks=[cp_callback, earlystopping_callback],
    )

    NFmodel.trainable=False
    print('Done training flow!')

    return NFmodel, flow

