import tensorflow as tf
tfk  = tf.keras

import numpy as np
import os

from . import flows

def load_ae_models(params):
    """load encoder and decoder models"""                                                                                 
    ae_model_params_fname = 'AE_kfold{:d}_{:02d}Dlatent_layers{:s}_{:s}'.format(params['kfold'],
                                                                          params['latent_dim'],
                                                                           '-'.join(str(e) for e in params['encode_dims']),
                                                                                params['out_file_tail'],
    )


    model_params_out_path = os.path.join(params['PROJECT_DIR'], f"{params['PARAM_DIR']}{ae_model_params_fname}" )
    if not params['overfit']:
        model_params_out_path += "_best"

    model_params_out_path += ".npy"
    
    if params['verbose']: 
        print("loading AE model from: ", model_params_out_path)

    AE_model_params = np.load(model_params_out_path,
                              allow_pickle='TRUE').item()

    if params['verbose']:
        print("AE model params: ", AE_model_params)

    print("\n\n\n\n\n", AE_model_params['encoder'])
    encoder = tfk.models.load_model(AE_model_params['encoder'], compile=False)
    decoder = tfk.models.load_model(AE_model_params['decoder'], compile=False)
    AE_params = AE_model_params['parameters']

    if params['model_summary'] and params['verbose']:
        print("Encoder Summary")
        encoder.summary()

        print("Decoder Summary")
        decoder.summary()

    return encoder, decoder, AE_params


def load_flow(params):
    checkpoint_filepath = '{:s}flow_kfold{:d}_{:02d}Dlatent_layers{:s}_nlayers{:02d}_{:s}'.format(params['MODEL_DIR'],
                                                                                      params['kfold'],
                                                                                      params['latent_dim'],
                                                                                      '-'.join(str(e) for e in params['encode_dims']),
                                                                                      params['nlayers'],
                                                                                      params['out_file_tail'])

    checkpoint_filepath = os.path.join(params['PROJECT_DIR'], checkpoint_filepath)

    if params['verbose']:
        print("loading flow from ", checkpoint_filepath)

    NFmodel, flow = flows.normalizing_flow(params)
    NFmodel.load_weights(checkpoint_filepath)

    return NFmodel, flow

class PAE:
    """Probabilistic AutoEncoder

    contains models for the three necessary components:
    encoder: x -> z
    decoder: z -> x'
    flow: z <-> u
    """
    def __init__(self, params):
        self.params=params

        self.encoder, self.decoder, self.AE_params = load_ae_models(params)

        NF, self.flow = load_flow(params)

        
    def generate_sample(self, n_samp=1, times=None, redshift=0.05, rand=True, seed=13579):
        """Generates random SN from gaussian latent space for a given set of observation times. 
    
        Parameters
        ----------
        n_samp: int
           number of samples to generate
        times: array
           observation time of each spectra, scaled to (0,1)
                       -1 padded to the maximum number of SN in a training/test sample

        Returns
        -------
        spectra: array of (N_sn, n_timesamples, data_dim)
        times: time of observations (N_sn, n_timesamples,)
        N_sn: int=1 for now. May update later
        n_timesamples: number of spectra observed from the SN, 
                       -1 padded to the maximum number of SN in a training/test sample
        data_dim: number of wavelength bins observed (288)
        """
        
        if type(times) is not np.ndarray:
            times = np.zeros((n_samp, self.params['n_timestep'])) + np.linspace(0, 1, self.params['n_timestep'])

        if not rand:
            tf.random.set_seed(seed)
            
        # randomly sample latent space of normalizing flow (u), and transform to latent space of autoencoder (z)'''
        z_ = self.flow.sample(n_samp)

        # decode spectra at given observation times
        spec_ = self.decoder((z_, times))
        
        # redshift
        # spec_ = L_to_F(spec_, redshift).numpy()

        return spec_, times

