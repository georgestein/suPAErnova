import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow.keras as tfk
import numpy as np
import random as rn
from pathlib import Path
import os

class AutoEncoder(tf.keras.Model):
    '''Autoencoder with option for fixed amplitude parameter and colorlaw of decoder 
        Modified to take in conditional vatiables (time of observation)'''
    def __init__(self, params, training=True,
                 num_physical_latent_dims=3,
                 bn_moving_means=[0., 0., 0.]): 
        super(AutoEncoder, self).__init__()
        # network dimensions and layers
        self.num_physical_latent_dims = num_physical_latent_dims #[delta t, delta m, Av]
        self.params = params

        # activation functions
        if params['activation'].upper()=='RELU':
            self.activation = tf.nn.relu
        elif params['activation'].upper()=='GELU':
            self.activation = tfk.activations.gelu
        elif params['activation'].upper()=='ELU':
            self.activation = tf.nn.elu
        elif params['activation'].upper()=='SWISH':
            self.activation = tf.nn.swish
        else:
            out_str = "Activation {:s} not included".format(params['activation'])
            print(out_str)
            sys.exit(out_str)
        # training 
        self.training = training

        if Path(params['colorlaw_file']).is_file():
            # load preset colorlaw
            w, CL, CL_deriv = np.loadtxt(params['colorlaw_file'], unpack=True)
            self.colorlaw = CL
            self.colorlaw_init = tf.constant_initializer(CL)

            self.colorlaw_deriv = CL_deriv
            self.colorlaw_deriv_init = tf.constant_initializer(CL_deriv)


        else:
            print('colorlaw file {:s} does not exist'.format(params['colorlaw_file']))
            self.colorlaw_preset = False

        if self.params['kernel_regularizer']:
            self.kernel_regularizer = tfk.regularizers.l2(params['kernel_regularizer_val'])
        else:
            self.kernel_regularizer = None

        self.bn_moving_means = bn_moving_means

        # set random seeds
        os.environ['PYTHONHASHSEED']=str(params['seed'])
        tf.random.set_seed(params['seed'])
        np.random.seed(params['seed'])
        rn.seed(params['seed'])
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

        # build models
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
#         self.amplitude_predictor = self.build_amplitude_predictor()

    def build_encoder(self): 
        '''Encoder architecture'''
    
        encode_inputs_data = tfkl.Input(shape=(self.params['n_timestep'], self.params['data_dim']))
        encode_inputs_cond = tfkl.Input(shape=(self.params['n_timestep'], self.params['cond_dim']))
        encode_inputs_mask = tfkl.Input(shape=(self.params['n_timestep'], 1))

        # add either convolutional or fully connected block
        encode_x = encode_inputs_data
        if self.params['layer_type'].upper() == 'DENSE':
            # Append time as additional parameter.
            # If convolutional layers, instead append after convolutions
            encode_x = tfkl.concatenate([encode_inputs_data, encode_inputs_cond])

        for iunit, nunit in enumerate(self.params['encode_dims'][:-1]):
#            print(iunit, encode_x.shape)
            # fully connected layer
            if self.params['layer_type'].upper()=='DENSE':
                encode_x = tf.keras.layers.Dense(nunit,
                                                 activation=self.activation,
                                                 kernel_regularizer=self.kernel_regularizer)(encode_x)

            elif self.params['layer_type'].upper()=='CONVOLUTION':
                # reshape (batch_shape, timesteps, spectra) event
                # from (batch_shape,H,W) to (batch_shape,H,W,C=1) in order to use Conv2D
                # keep first dimension ov kernel_size and stride as 1 in order to operate along spectra only
                if iunit==0:
                    # convolutional layer
                    encode_x = tf.expand_dims(encode_x, axis=-1)

                    encode_x = tfkl.Conv2D(nunit,
                                           kernel_size=(1, self.params['kernel_size']),
                                           strides=(1, self.params['stride']),
                                           activation=self.activation,
                                           kernel_regularizer=self.kernel_regularizer,
                                           padding='same',
                                           input_shape=(self.params['n_timestep'], self.params['data_dim']))(encode_x)
                else:
                    # convolutional layer
                    encode_x = tfkl.Conv2D(nunit,
                                           kernel_size=(1, self.params['kernel_size']),
                                           strides=(1, self.params['stride']),
                                           activation=self.activation,
                                           padding='same',
                                           kernel_regularizer=self.kernel_regularizer)(encode_x)#,

            else:
                sys.exit('Layer type {:s} does not exist'.format(params['layer_type']))
                
            if self.params['dropout']:
                # dropout along features dimension, keeping dropout along time dimension consistent
                encode_x = tfkl.Dropout(self.params['dropout_rate'],
                                        noise_shape=[None, 1, None])(encode_x, training=self.training)

            if self.params['batchnorm']:
                encode_x = tfkl.BatchNormalization()(encode_x)

        if self.params['layer_type'].upper()=='CONVOLUTION':
            # reshape to pass to Dense layers
            encode_x_shape = encode_x.shape
            self.encode_x_shape = encode_x_shape
            encode_x = tfkl.Reshape((encode_x_shape[-3], encode_x_shape[-2]*encode_x_shape[-1]))(encode_x)
            
        # add conditional time parameter as new feature
        if self.params['layer_type'].upper()=='CONVOLUTION':
            encode_x = tfkl.concatenate([encode_x, encode_inputs_cond])

        # dense layers
        encode_x = tfkl.Dense(self.params['encode_dims'][-1],
                              activation=self.activation,
                              kernel_regularizer=self.kernel_regularizer)(encode_x)

        encode_x = tfkl.Dense(self.params['latent_dim']+self.num_physical_latent_dims, 
                              kernel_regularizer=self.kernel_regularizer,
                              use_bias=False)(encode_x)

        
        # need to mask time samples that do not exist = take mean of non masked latent variables
        encode_outputs = tf.reduce_sum(encode_x*encode_inputs_mask, axis=-2)/tf.math.maximum(tf.reduce_sum(encode_inputs_mask, axis=-2), 1) 

        if self.params['physical_latent']:
            # encode_color   = tf.nn.relu(encode_outputs[..., -1:])
            encode_dtime     = encode_outputs[..., 0:1] # delta time
            encode_amplitude = encode_outputs[..., 1:2] # delta m
            encode_color     = encode_outputs[..., 2:3] # color Av
            encode_latent    = encode_outputs[..., 3:]  # intrinsic latent variables

            if self.params['train_stage'] == 0:
                # set these parameters to 0 at this stage in training
                encode_amplitude = encode_amplitude*0.
                encode_dtime     = encode_dtime*0.

            if self.params['train_stage'] == 1:
                # set these parameters to 0 at this stage in training
                encode_dtime     = encode_dtime*0. 
                    
            # Make dtime, damplitude, and Av of nonmasked SN have mean 0
            # This roundabout way is required due to the dreaded TensorFlow error:
            # TypeError: 'Tensor' object does not support item assignment
                
            if self.training:
                is_kept = tf.reduce_max(encode_inputs_mask[:, :, 0], axis=-1)
                
                batch_mean_dtime = tf.reduce_sum(encode_dtime * is_kept, axis=0)/tf.reduce_sum(is_kept)
                encode_dtime = tfkl.subtract([encode_dtime, batch_mean_dtime])
                
                batch_mean_amplitude = tf.reduce_sum(encode_amplitude * is_kept, axis=0)/tf.reduce_sum(is_kept)
                encode_amplitude = tfkl.subtract([encode_amplitude, batch_mean_amplitude])
                
                batch_mean_color = tf.reduce_sum(encode_color * is_kept, axis=0)/tf.reduce_sum(is_kept)
                encode_color = tfkl.subtract([encode_color, batch_mean_color])
                
            else:
                encode_dtime     = tfkl.subtract([encode_dtime, tf.Variable([self.bn_moving_means[0]])])
                encode_amplitude = tfkl.subtract([encode_amplitude, tf.Variable([self.bn_moving_means[1]])])
                encode_color     = tfkl.subtract([encode_color, tf.Variable([self.bn_moving_means[2]])])
                
            encode_outputs = tfkl.concatenate([encode_dtime, encode_amplitude, encode_color, encode_latent])

            
        return tfk.Model(inputs=[encode_inputs_data, encode_inputs_cond, encode_inputs_mask], outputs=encode_outputs)


    def build_decoder(self):
        '''Decoder architecture''' 

        decode_inputs_latent = tfkl.Input(shape=(self.params['latent_dim']+self.num_physical_latent_dims,), name='latent_params')
        decode_inputs_cond   = tfkl.Input(shape=(self.params['n_timestep'], self.params['cond_dim']), name='conditional_params')
        decode_inputs_mask = tfkl.Input(shape=(self.params['n_timestep'], 1))

        # repeat latent vector to match number of data timesteps
        decode_latent = tfkl.RepeatVector(self.params['n_timestep'])(decode_inputs_latent)

        if self.params['physical_latent']:
            #decode_color       = tf.exp(decode_latent[..., -1:]) # ensure colorlaw latent parameter>=0
            decode_dtime     = decode_latent[..., 0:1]
            decode_amplitude = decode_latent[..., 1:2]
            decode_color     = decode_latent[..., 2:3] # ensure colorlaw latent parameter>=0

            decode_latent = decode_latent[..., self.num_physical_latent_dims:]
            # add in latent parameter after first layer
            decode_x = tfkl.concatenate([decode_latent,
                                         decode_inputs_cond + decode_dtime])
        else:
            decode_x = tfkl.concatenate([decode_latent, decode_inputs_cond])

        decode_x = tf.keras.layers.Dense(self.params['decode_dims'][0],
                                         activation=self.activation,
                                         kernel_regularizer=self.kernel_regularizer)(decode_x)

        #decode_x = tfkl.concatenate([decode_x, decode_inputs_cond])
        # add either convolutional or fully connected block
        for iunit, nunit in enumerate(self.params['decode_dims'][1:]):

            if self.params['layer_type'].upper()=='DENSE':
                # fully connected network
                decode_x = tf.keras.layers.Dense(nunit,
                                                 activation=self.activation,
                                                 kernel_regularizer=self.kernel_regularizer)(decode_x)


            elif self.params['layer_type'].upper()=='CONVOLUTION':
                # reshape (batch_shape, timesteps, spectra) event
                # from (batch_shape,H,W) to (batch_shape,H,W,C=1) in order to use Conv2D
                # keep first dimension of kernel_size and stride as 1 in order to operate along spectra only
                if iunit==0:

                    decode_x = tf.keras.layers.Dense(self.encode_x_shape[-2]*self.encode_x_shape[-1],
                                                     activation=self.activation,
                                                     kernel_regularizer=self.kernel_regularizer)(decode_x)

                    decode_x = tfkl.Reshape((self.params['n_timestep'], self.encode_x_shape[-2], self.encode_x_shape[-1]))(decode_x)


                decode_x = tf.keras.layers.Conv2DTranspose(nunit,
                                                           kernel_size=(1, self.params['kernel_size']),
                                                           strides=(1, self.params['stride']),
                                                           activation=self.activation,
                                                           padding='same',
                                                           kernel_regularizer=self.kernel_regularizer)(decode_x)

                    
            else:
                sys.exit('Layer type {:s} does not exist'.format(params['layer_type']))

#            if self.params['dropout']:
#                decode_x = tf.keras.layers.Dropout(self.params['dropout_rate'],
#                                                   noise_shape=[None, 1, None])(decode_x, training=self.training)
                
            if self.params['batchnorm']:
                decode_x = tfkl.BatchNormalization()(decode_x)

        if self.params['layer_type'].upper()=='CONVOLUTION':
            decode_outputs = tfkl.Reshape((self.params['n_timestep'], decode_x.shape[-2]*decode_x.shape[-1]))(decode_x)

        else:
            decode_outputs = tf.keras.layers.Dense(self.params['data_dim'],
                                                   kernel_regularizer=self.kernel_regularizer)(decode_x)
        if self.params['colorlaw_preset']:
            # use input colorlaw, SALT2 or Fitzpatrick99
            decode_colorlaw = tf.keras.layers.Dense(self.params['data_dim'],
                                                    kernel_initializer=self.colorlaw_init,
                                                    name='color_law',
                                                    use_bias=False,
                                                    trainable=False)(decode_color)
                
        else:
            decode_colorlaw = tf.keras.layers.Dense(self.params['data_dim'],
                                                    kernel_initializer=self.colorlaw_init,
                                                    name='color_law',
                                                    use_bias=False,
                                                    trainable=True,
                                                    kernel_constraint=tfk.constraints.NonNeg())(decode_color)
        if self.params['physical_latent']:
            # multiply by amplitude and colorlaw
            # -0.4 to account for magnitudes
            decode_outputs = ( decode_outputs
                              * 10 ** (-0.4 * (decode_colorlaw + decode_amplitude) ))

        return tfk.Model(inputs=[decode_inputs_latent, decode_inputs_cond, decode_inputs_mask], outputs=decode_outputs*decode_inputs_mask) 


    def encode(self, x, cond, mask):
        return self.encoder((x, cond, mask))

    def decode(self, z, cond, mask):
        return self.decoder((z, cond, mask))


      # predict amplitude/redshift from latent parameters
#     def build_amplitude_predictor(self): 
#         '''Encoder architecture'''

#         if self.params['physical_latent']:    
#             amplitude_input_dim = self.latent_dim-1
#         else:
#             amplitude_input_dim = self.latent_dim

#         amplitude_predictor_inputs = tfkl.Input(shape=(amplitude_input_dim))

#         # add layers, either convolutional or fully connected
#         for iunit, nunit in enumerate(self.amplitude_predictor_dims):

#             # fully connected network
#             if iunit==0:
#                 amplitude_x = tf.keras.layers.Dense(nunit, activation=self.activation, kernel_regularizer=self.kernel_regularizer)(amplitude_predictor_inputs) 
#             else:
#                 amplitude_x = tf.keras.layers.Dense(nunit, activation=self.activation, kernel_regularizer=self.kernel_regularizer)(amplitude_x) 

#         # dense then take mean of time axis
#         amplitude_predictor_outputs = tf.keras.layers.Dense(1, kernel_regularizer=self.kernel_regularizer)(amplitude_x)
#     #                                      use_bias=False,)(encode_x)

#         return tfk.Model(inputs=amplitude_predictor_inputs, outputs=amplitude_predictor_outputs)

#    def amplitude_predictor(self, z):
#        return self.amplitude_predictor((z))
