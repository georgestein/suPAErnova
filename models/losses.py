import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow.keras as tfk
import tensorflow_probability as tfp

import numpy as np
import random as rn
from pathlib import Path
import os

def get_apply_grad_fn():
    """
     Wrap @tf.function's to prevent autograph causing the following error when called more than once: 
     "ValueError: tf.function-decorated function tried to create variables on non-first call.""
     see https://github.com/tensorflow/tensorflow/issues/27120#issuecomment-540071844
    """
    @tf.function
    def compute_apply_gradients(model, x, cond, sigma, mask, dl, optimizer):
        with tf.GradientTape() as tape:
            loss, loss_terms = compute_loss_ae(model, x, cond, sigma, mask, dl)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss, loss_terms

    return compute_apply_gradients

@tf.function
def compute_loss_ae(model, x, cond, sigma, mask, dl):

    # get latent paramaters
    z      = model.encode(x, cond, mask)

    # from latent paramaters and observation times reconstruct data
    x_pred = model.decode(z, cond)

    # RECONSTRUCTION LOSS TERM
    # Take mean of non masked latent variables. 
    # SN with more observations should be given greater weights, so take sum instead of mean

    # sigma can be -1 before masking, or 0 after. Can't take log of this so add small number
    if model.params['loss_fn'].upper() == 'MAE':
        loss = tf.reduce_sum( tf.reduce_sum( tf.abs((x - x_pred) * mask), axis=(-2,-1)))

    if model.params['loss_fn'].upper() == 'wMAE':
        loss = tf.reduce_sum( tf.reduce_sum( tf.abs((x - x_pred)/(sigma + 1e-9)) * mask, axis=(-2,-1)))

    if model.params['loss_fn'].upper() == 'MSE':
        loss = tf.reduce_mean( tf.reduce_sum( (x - x_pred)**2 * mask, axis=(-2,-1)))

    if model.params['loss_fn'].upper() == 'wMSE':
        loss = tf.reduce_mean( tf.reduce_sum( (x - x_pred)**2/(sigma + 1e-9)**2 * mask, axis=(-2,-1)))

    if model.params['loss_fn'].upper() == 'RMSE':
        loss = tf.reduce_mean( tf.reduce_sum( (x - x_pred)**2 * mask, axis=(-2,-1))/tf.reduce_sum(mask, axis=-2))

    if model.params['loss_fn'].upper() == 'wRMSE':
        loss = tf.reduce_mean( tf.sqrt( tf.reduce_sum( (x - x_pred)**2/(sigma + 1e-9)**2 * mask, axis=(-2,-1)) / tf.reduce_sum(mask, axis=-2)))

    if model.params['loss_fn'].upper() == 'NGLL':
        loss = tf.reduce_mean( tf.reduce_sum( (tf.math.log(sigma**2 * mask + 1e-9)/2 + (x - x_pred)**2/(2*sigma**2+1e-9)) * mask, axis=(-2,-1)))#/tf.reduce_sum(mask, axis=-2))        

    if model.params['loss_fn'].upper() == 'HUBER':
        clip_delta = 10
        error = (x - x_pred)/(sigma + 1e-9) * mask
        cond  = tf.keras.backend.abs(error) < model.params['clip_delta']

        squared_loss = 0.5 * tf.keras.backend.square(error)
        linear_loss  = model.params['clip_delta'] * (tf.keras.backend.abs(error) - 0.5 * model.params['clip_delta'])

        loss = tf.reduce_sum( tf.where(cond, squared_loss, linear_loss), axis=(-2, -1))

    loss_recon = tf.reduce_mean(loss)
    loss = loss_recon

    # Punish overall amplitude offset of spectra from SN
    if model.params['iloss_amplitude_offset']:
        #    loss = loss + tf.abs(tf.reduce_sum( (x - x_pred)/(sigma + 1e-9) * mask, axis=(-2, -1)))
        loss_offset = tf.reduce_mean(tf.abs(tf.reduce_sum( (x - x_pred) * mask, axis=(-2, -1)))) 

        loss += loss_offset * model.params['lambda_amplitude_offset'] 

    if model.params['use_amplitude'] and model.params['iloss_amplitude_parameter']:
#        mask_z = tf.reduce_max(mask, axis=(-2, -1))
#        sum_mask_z = tf.reduce_sum(mask_z)
#        mean_z = tf.reduce_sum(z[:,0]*mask_z)/sum_mask_z
#        sig = tf.sqrt(tf.reduce_sum( (z[:,0]-mean_z)**2 * mask_z)/sum_mask_z)

        #loss_amplitude = tf.reduce_sum( (z[:, 0] - 1)**2 * mask_z)/sig   

        z_median = tfp.stats.percentile(z[:,0], 50.0, interpolation='midpoint')
        loss_amplitude = (1-z_median)**2 

        loss += loss_amplitude*model.params['lambda_amplitude_parameter']
#         if model.physical_latent and not model.colorlaw_preset:
#             # if fitting for colorlaw instead of just using premade,
#             # make colowlaw amplitude small, such that colorlaw itself gets 
#             loss_colorlaw = tf.reduce_sum(z[:, :-1]**2)
#             loss += model.lambda_colorlaw * loss_colorlaw

    # COVARIANCE LOSS TERM
    # want to add loss to uncorrelate latent features
    # found similar in A PCA-LIKE AUTOENCODER - https://arxiv.org/abs/1904.01277
    # PCAAE: Principal Component Analysis Autoencoder for organising the latent space of generative networks - https://arxiv.org/abs/2006.07827
    if model.params['iloss_covariance']:
        # apply covariace loss to physical latent parameters
        # amplitude is from peculiar velocity and color is from line-of-sight dust, so these should be uncorrelated with the other latent parameters, and with each other
        z_cov = z

        mean_z = tf.reduce_mean(z_cov, axis=0, keepdims=True)
        mz = tf.matmul(tf.transpose(mean_z), mean_z)
        vz = tf.matmul(tf.transpose(z_cov), z_cov)/tf.cast(tf.shape(z_cov)[0], tf.float32)
        cov_z = (vz - mz)/mz # normalize covariance, else latent variables are just pushed to small numbers

        # punishes all off-diagonal covariance elements
        #loss_cov = tf.reduce_sum(tf.square(cov_z - tf.math.multiply(cov_z, tf.eye(cov_z.shape[1]))))

        # only punish covariance of latent parameters with amplitude (first latent parameter)
        istart = 0
        cov_mask = np.ones( (model.params['latent_dim'], model.params['latent_dim']), dtype=np.float32)
        if model.params['use_amplitude']:
            istart = 1
            ## amplitude and colorlaw can be correlated
            #cov_mask[0, -1] = 0. 
            #cov_mask[-1, 0] = 0. 
            cov_mask[0, 0]  = 0.  # corners

        cov_mask[istart:, istart:] = 0. # central region
        
        cov_mask = tf.convert_to_tensor(cov_mask)
        loss_cov = tf.reduce_sum(tf.square(tf.math.multiply(cov_z, cov_mask))) 

        # tf.print('DEBUG COVARIANCE', loss_cov, loss_cov*model.params['lambda_covariance'])
        # loss_cov = tf.reduce_sum(tf.square(cov_z))
        loss += loss_cov * model.params['lambda_covariance'] 

#    tf.print(loss_recon, loss_offset, loss_amplitude, loss_cov)
#    tf.print(loss_recon, loss_offset*model.params['lambda_amplitude_offset'], loss_amplitude*model.params['lambda_amplitude_parameter'], loss_cov * model.params['lambda_covariance'])
#         if model.params['iloss_amplitude']:
#             # AMPLITUDE PREDICTION LOSS
#             lambda_amp = 100
#             if params['physical_latent']: # first latent paramater is intrinsic luminosity
#                 loss_amp = tf.reduce_sum(tf.square((A_pred - z[:, 0:1])))
#             else: 
#                 loss_amp = tf.reduce_sum(tf.square((A_pred - dl)))

# #             tf.print('z, A = ', z[:, 0:1].shape, A_pred.shape, z[:, 0:1], A_pred)
#             loss += model.params['lambda_amplitude']*loss_amp


    # KERNEL REGULARIZER LOSS
    if model.kernel_regularizer:
        loss += tf.math.reduce_sum(model.losses)

    return loss, [loss]#model.params['lambda_covariance']*loss_cov]#, model.params['lambda_amplitude']*loss_amp]


@tf.function
def compute_loss_posterior(model, x):
    loss = model(x)
    return -loss

def get_compute_MAP():
    
    @tf.function
    def compute_apply_gradients_MAP(model, x, trainable_params, optimizer):
        with tf.GradientTape() as tape:
            nlPost_MAP = compute_loss_posterior(model, x)
        gradients = tape.gradient(nlPost_MAP, trainable_params)
        optimizer.apply_gradients(zip(gradients, trainable_params))

        return nlPost_MAP

    return compute_apply_gradients_MAP

