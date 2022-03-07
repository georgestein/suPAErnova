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

from suPAErnova.utils.YParams import YParams
from suPAErnova.utils import data_loader

from suPAErnova.models import flows
from suPAErnova.models import flow_training
from suPAErnova.models import losses
from suPAErnova.models import loader as model_loader

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--yaml_config", default='../config/train.yaml', type=str)
    parser.add_argument("--config", default='pae', type=str)
    parser.add_argument("--print_params", default=True, action='store_true')
    
    args = parser.parse_args()

    return args
    
def main():
    
    args = parse_arguments()
    
    params = YParams(os.path.abspath(args.yaml_config), args.config, print_params=True)

    for il, latent_dim in enumerate(params['latent_dims']):
        print('Training model with {:d} latent dimensions'.format(latent_dim))
        params['latent_dim'] = latent_dim

        encoder, decoder, AE_params = model_loader.load_ae_models(params)

        train_data = data_loader.load_data(
            os.path.join(
                params['PROJECT_DIR'],
                params['train_data_file'],
            ),
            print_params=params['print_params'],
            set_data_min_val=params['set_data_min_val'],
        )

        test_data  = data_loader.load_data(
            os.path.join(
                params['PROJECT_DIR'],
                params['test_data_file'],
            ),
            set_data_min_val=params['set_data_min_val'],
        )

        # Mask certain supernovae       
        train_data['mask_sn'] = data_loader.get_train_mask(train_data, AE_params.params)
        test_data['mask_sn'] = data_loader.get_train_mask(test_data, AE_params.params)

        # Mask certain spectra
        train_data['mask_spectra'] = data_loader.get_train_mask_spectra(train_data, AE_params.params)
        test_data['mask_spectra'] = data_loader.get_train_mask_spectra(test_data, AE_params.params)

        # Get latent representations from encoder
        train_data['z_latent'] = encoder((
            train_data['spectra'],
            train_data['times'],
            train_data['mask']*train_data['mask_spectra'],
        )).numpy()

        test_data['z_latent']  = encoder((
            test_data['spectra'],
            test_data['times'],
            test_data['mask']*test_data['mask_spectra'],
        )).numpy()

        train_data['z_latent'] = train_data['z_latent'][train_data['mask_sn']]
        test_data['z_latent'] = test_data['z_latent'][test_data['mask_sn']]

        # Split off validation set from training set
        # train_data, val_data = data_loader.split_train_and_val(train_data, params)

        # Saved on checkpoint, so no need to save again
        NFmodel, flow = flow_training.train_flow(
            train_data,
            test_data,
            params,
            )

if __name__ == '__main__':
    main()
    
