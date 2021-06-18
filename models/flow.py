import numpy as np

import tensorflow as tf
tfk  = tf.keras
tfkl = tf.keras.layers

import tensorflow_probability as tfp
tfb  = tfp.bijectors
tfd  = tfp.distributions


def normalizing_flow(params, optimizer=tf.optimizers.Adam(1e-3)):
    '''event_dim: dimensions of input data'''
    train_phase = True

    indices = np.roll(np.arange(params['latent_dim']), 1)
    permutations = [indices for ii in range(params['nlayers'])]

    bijectors = []
    if params['batchnorm']: 
        bijectors.append(tfb.BatchNormalization(training=train_phase, name='batch_normalization'))

    for i in range(params['nlayers']):
        bijectors.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=tfb.AutoregressiveNetwork(params=2, hidden_units=[params['nunit'], params['nunit']], activation='relu', use_bias=True)))
        if params['batchnorm']:
            bijectors.append(tfb.BatchNormalization(training=train_phase, name='batch_normalization'))
        bijectors.append(tfb.Permute(permutation=permutations[i]))


    flow = tfd.TransformedDistribution(
        distribution=tfd.MultivariateNormalDiag(loc=tf.zeros(params['latent_dim']),
                                                scale_diag=tf.ones(params['latent_dim'])),
        bijector=tfb.Chain(list(reversed(bijectors[:-1]))))#,

    # Construct and fit model.
    z_ = tfkl.Input(shape=(params['latent_dim'],), dtype=tf.float32)
    log_prob_ = flow.log_prob(z_)

    model = tfk.Model(inputs=z_, outputs=log_prob_)

    model.compile(optimizer=optimizer,
                  loss=lambda _, log_prob: -log_prob)

    return model, flow



