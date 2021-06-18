import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow.keras as tfk
import numpy as np
import random as rn
from pathlib import Path
import os

class AutoEncoder(tf.keras.Model):
    '''Autoencoder with option for fixed amplitude paramater and colorlaw of decoder 
        Modified to take in conditional vatiables (time)'''
    def __init__(self, params): 
        super(AutoEncoder, self).__init__()
        # network dimensions and layers
        self.params = params

        # activation functions
        self.activation = tf.nn.relu
#        self.activation = tf.nn.swish

        # training 
        self.training = True

        if Path(params['colorlaw_file']).is_file():
            # load preset colorlaw
            w, CL, CL_deriv = np.loadtxt(params['colorlaw_file'], unpack=True)
            self.colorlaw = CL
            self.colorlaw_init = tf.constant_initializer(CL)
            # self.colorlaw = tf.convert_to_tensor(CL, dtype=tf.float32)

            self.colorlaw_deriv = CL_deriv
            self.colorlaw_deriv_init = tf.constant_initializer(CL_deriv)


        else:
            print('colorlaw file {:s} does not exist'.format(params['colorlaw_file']))
            self.colorlaw_preset = False

        if self.params['kernel_regularizer']:
            self.kernel_regularizer = tfk.regularizers.l2(params['kernel_regularizer_val'])
        else:
            self.kernel_regularizer = None


        # set random seeds
        os.environ['PYTHONHASHSEED']=str(params['seed'])
        tf.random.set_seed(params['seed'])
        np.random.seed(params['seed'])
        rn.seed(params['seed'])
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
#         self.amplitude_predictor = self.build_amplitude_predictor()

    def build_encoder(self): 
        '''Encoder architecture'''
    
        encode_inputs_data = tfkl.Input(shape=(self.params['n_timestep'], self.params['data_dim']))
        encode_inputs_cond = tfkl.Input(shape=(self.params['n_timestep'], self.params['cond_dim']))
        encode_inputs_mask = tfkl.Input(shape=(self.params['n_timestep'], 1))

        # Dropout on wavelength bins 
    #     if self.dropout:
    #         # dropout on data, not on the conditional time parameter
    #         encode_inputs_data_drop = tf.keras.layers.Dropout(self.dropout_rate, noise_shape=[None, None, 1])(encode_inputs_data, training=True)
    #         encode_x = tfkl.concatenate([encode_inputs_data_drop, encode_inputs_cond])

    #     else:
    #     # Dropout on whole time samples
    #     encode_x = tf.keras.layers.Dropout(self.dropout_rate, noise_shape=[None, None, 1])(encode_x, training=True)

        # add dense layer for conditional paramater only
    #     encode_inputs_cond_x = tf.keras.layers.Dense(self.cond_dim, activation=None, kernel_regularizer=self.kernel_regularizer)(encode_inputs_cond) 
    #     encode_x = tfkl.concatenate([encode_inputs_data, encode_inputs_cond_x])

        encode_x = tfkl.concatenate([encode_inputs_data, encode_inputs_cond])

        # add layers, either convolutional or fully connected
        for iunit, nunit in enumerate(self.params['encode_dims']):

            # fully connected network
            encode_x = tf.keras.layers.Dense(nunit, activation=self.activation, kernel_regularizer=self.kernel_regularizer)(encode_x) 

            if self.params['dropout'] and iunit>0: # and iunit==0:
                # dropout along features dimension, keeping sropout along time dimension consistent
                encode_x = tf.keras.layers.Dropout(self.params['dropout_rate'], noise_shape=[None, 1, None])(encode_x, training=self.training)        

            if self.params['batchnorm']:
                encode_x = tfkl.BatchNormalization()(encode_x)



        # dense then take mean of time axis
        encode_x = tf.keras.layers.Dense(self.params['latent_dim'], kernel_regularizer=self.kernel_regularizer)(encode_x)

        
        # need to mask time samples that do not exist. 
        # Take mean of non masked latent variables
        encode_outputs = tf.reduce_sum(encode_x*encode_inputs_mask, axis=-2)/tf.math.maximum(tf.reduce_sum(encode_inputs_mask, axis=-2), 1) # max(1) to deal with all-masked SN

        if self.params['physical_latent']:
            #encode_color = tf.nn.relu(encode_outputs[..., -1:]) # ensure >= 0
            #encode_color   = tf.math.log( tf.math.abs(encode_outputs[..., -1:]) ) # ensure >= 0
            encode_color   = encode_outputs[..., -1:]
            encode_outputs = encode_outputs[..., :-1]    
#            encode_color = tf.math.abs(encode_color)                

            if self.params['use_amplitude']:
                # amplitude multiplication of output
                encode_amplitude = encode_outputs[..., 0:1] # keep last dim shape
                encode_outputs   = encode_outputs[..., 1:]    
                
                # make amplitude positive by default
                encode_amplitude = tf.math.abs(encode_amplitude)                
                encode_outputs = tfkl.concatenate([encode_amplitude, encode_outputs, encode_color])

            else:
                encode_outputs = tfkl.concatenate([encode_outputs, encode_color])
            
        return tfk.Model(inputs=[encode_inputs_data, encode_inputs_cond, encode_inputs_mask], outputs=encode_outputs)


    def build_decoder(self):
        '''Decoder architecture''' 

        decode_inputs_latent = tfkl.Input(shape=(self.params['latent_dim'],), name='latent_params')
        decode_inputs_cond   = tfkl.Input(shape=(self.params['n_timestep'], self.params['cond_dim']), name='conditional_params')

        decode_latent = tfkl.RepeatVector(self.params['n_timestep'])(decode_inputs_latent)

        if self.params['physical_latent']:
            decode_color       = tf.exp(decode_latent[..., -1:]) # ensure latent parameter>=0
#            decode_color       = decode_latent[..., -1:]
#             decode_color       = decode_latent[..., -2]
#             decode_color_deriv = decode_latent[..., -1]

            # make first latent variable amplitude multiplication == "Luminosity"

            if self.params['use_amplitude']:
                decode_amplitude = decode_latent[..., 0:1]
                
                decode_x = tfkl.concatenate([decode_latent[..., 1:-1], decode_inputs_cond])
#                 decode_x = tfkl.concatenate([decode_latent[..., 1:-2], decode_inputs_cond])
        
            else:
                decode_x = tfkl.concatenate([decode_latent[..., :-1], decode_inputs_cond])
#                 decode_x = tfkl.concatenate([decode_latent[..., :-2], decode_inputs_cond])

        else:
            decode_x = tfkl.concatenate([decode_latent, decode_inputs_cond])

    #     # add dense layer for conditional paramater only
    #     decode_inputs_cond_x = tf.keras.layers.Dense(self.cond_dim, activation=None, kernel_regularizer=self.kernel_regularizer)(decode_inputs_cond) 
    #     decode_x = tfkl.concatenate([decode_x, decode_inputs_cond_x])

    #     # Dropout on whole time samples
    #     decode_x = tf.keras.layers.Dropout(self.dropout_rate, noise_shape=[None, None, 1])(decode_x, training=True)

        # get input into proper shape
        # add layers, either convolutional or fully connected
        for iunit, nunit in enumerate(self.params['decode_dims']):

            # fully connected network
            decode_x = tf.keras.layers.Dense(nunit,
                                             activation=self.activation,
                                             kernel_regularizer=self.kernel_regularizer)(decode_x)

            if self.params['dropout'] and iunit>0:# and iunit==len(self.decode_dims)-1:
                decode_x = tf.keras.layers.Dropout(self.params['dropout_rate'],
                                                   noise_shape=[None, 1, None])(decode_x, training=self.training)        

            if self.params['batchnorm']:
                decode_x = tfkl.BatchNormalization()(decode_x)

        decode_outputs = tf.keras.layers.Dense(self.params['data_dim'],
#                                               activation=tf.nn.relu,
                                               kernel_regularizer=self.kernel_regularizer)(decode_x) # relu activation to ensure outputs positive
        #                                            , use_bias=False)(decode_x)
        decode_outputs = tf.math.abs(decode_outputs)
        if self.params['physical_latent']:
            # get "color law", akin to multiplying by 10^(latent_var*CL(lambda))

            if self.params['colorlaw_preset']:
                # use input colorlaw, SALT2 or Fitzpatrick99
                decode_colorlaw = tf.keras.layers.Dense(self.params['data_dim'],
                                                        kernel_initializer=self.colorlaw_init,
                                                        name='color_law',
                                                        use_bias=False,
                                                        trainable=False)(decode_color)
#                 decode_colorlaw_deriv = tf.keras.layers.Dense(self.data_dim, kernel_initializer=self.colorlaw_deriv_init, name='color_law_deriv', use_bias=False, trainable=False)(decode_color_deriv)
                
            else:
                # train 'colorlaw' = time independent component
#                 decode_colorlaw = tf.keras.layers.Dense(self.data_dim, kernel_initializer=self.colorlaw_init, name='color_law', use_bias=False, trainable=True, kernel_constraint=tfk.constraints.NonNeg())(decode_color)
                decode_colorlaw = tf.keras.layers.Dense(self.params['data_dim'],
                                                        name='color_law',
                                                        use_bias=False,
                                                        trainable=True,
                                                        kernel_constraint=tfk.constraints.NonNeg())(decode_color)

            # -0.4 to account for magnitudes
            if self.params['use_amplitude']:
                # multiply by amplitude and colorlaw
                decode_outputs = (decode_outputs
                                  * decode_amplitude
                                  * 10**(-0.4*decode_colorlaw))                
#                 decode_outputs = decode_outputs * decode_amplitude[..., tf.newaxis] * 10**(-0.4*(decode_colorlaw + decode_colorlaw_deriv))

            else:
                # multiply by colorlaw
                decode_outputs = (decode_outputs
                                  * 10**(-0.4*decode_colorlaw))
#                 print(decode_colorlaw.shape, decode_colorlaw_deriv.shape)
#                 decode_outputs = decode_outputs * 10**(-0.4*(decode_colorlaw + decode_colorlaw_deriv))

        return tfk.Model(inputs=[decode_inputs_latent, decode_inputs_cond], outputs=decode_outputs)


    def encode(self, x, cond, mask):
        return self.encoder((x, cond, mask))

    def decode(self, z, cond):
        return self.decoder((z, cond))


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
