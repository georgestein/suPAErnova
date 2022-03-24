import tensorflow as tf
tfk  = tf.keras
tfkl = tf.keras.layers

import tensorflow_probability as tfp
tfb  = tfp.bijectors
tfd  = tfp.distributions

import numpy as np

#https://emcee.readthedocs.io/en/stable/tutorials/line/
class LogPosterior(tfk.Model):
    '''Performs posterior analysis on a Probabalistic AutoEncoder (PAE: https://arxiv.org/abs/2006.05479) 
    trained on supernovae (SN) spectral timeseries. 

    Parameters:
    ----------
    PAE: class
        PAE model which incluses encoder, decoder, and flow
    data: dictionary
        data to use. requires arrays of 'spectra', 'times', 'redshift', 'sigma', 'mask'
        
    Encoder takes in (N_sn, n_timesamples, data_dim), and the conditional time parameter (N_sn, n_timesamples,)

    where:
    N_sn is the number of supernovae
    n_timesamples is the number of spectra observed from the SN, 
            -1 padded to the maximum number of SN in a training/test sample
    data_dim is the number of wavelength bins observed (288)
    and the conditional time paramater is the time of the observation, usually scaled to (0,1)
    '''
    def __init__(self, PAE, params, data,
                 sigma_time_bin_cent, sigma_time_grid):
        
        super(LogPosterior, self).__init__()
        self.params = params
        
        self.encoder = PAE.encoder # encoder network
        self.decoder = PAE.decoder # decoder network
        self.flow    = PAE.flow    # normalizing network

        self.x   = data['spectra']   # data
        self.x_c = data['times']     # conditional paramater
        self.z_c = data['redshift']  # redshift
        self.sigma_x = data['sigma'] # sigma noise
        self.mask_x  = data['mask']  # some spectra may be missing, or wavelength bins masked 

        self.nsamples      = self.x.shape[0]
        self.n_timesamples = self.x.shape[1]
        self.data_dim      = self.x.shape[2]

        # if whole spectrum is masked then discard, else use, even if partial is masked
        self.use_spectra = tf.reduce_min(self.mask_x, axis=-1)

        # number of spectra that exist in observation
        n_spectra_each = [np.shape(np.where(self.x[i, :, 0] > -1))[1] for i in range(self.x.shape[0])] 
        self.n_spectra = tf.convert_to_tensor(n_spectra_each, dtype=tf.float32) 
        
#         self.drop_top = int(0.2 * self.n_spectra + 0.5) # ten percent        
#         print('dropping worst {:d} spectra in fit'.format(self.drop_top))
        
        # global ae noise 
        # ae noise as a function of observation time   
        self.sigma_time_bin_cent = sigma_time_bin_cent
        self.sigma_time_grid     = sigma_time_grid
 
        self.latent_dim = self.encoder((
            self.x,
            self.x_c,
            self.mask_x,
        )).shape[1] # latent space dimensionality

        self.latent_dim_u = self.latent_dim

        # Don't use time shift or amplitude in normalizing flow
        # Amplitude represents uncorrelated shift from peculiar velocity and/or gray instrumental effects
        # And this is the parameter we want to fit to get "cosmological distances", thus we don't want a prior on it
        self.istart_map = 2
        self.latent_dim_u -= 2

        self.MAPtrue = self.flow.bijector.inverse(
            self.encoder(
                (self.x,
                 self.x_c,
                 self.mask_x),
            )[:, -self.latent_dim_u:],
        )
        
        # Initializations. Save initial values as seperate variables, as variables will be updated
        if self.params['rMAPini']:
            print('Random initial MAPini')
#            tf.random.set_seed(self.params['seed'])
            self.MAPu_ini       = self.get_latent_prior().sample(self.nsamples)
            self.MAPz_ini      = self.flow.bijector.forward(self.MAPu_ini) 

            self.amplitude_ini = self.get_amplitude_prior().sample(self.nsamples)
            self.MAPz_ini = tf.concat([self.amplitude_ini, self.MAPz_ini], axis=1)
                
            self.dtime_ini     = self.get_dtime_prior().sample(self.nsamples)
            
        else:

            self.MAPz_ini      = self.encoder((self.x, self.x_c, self.mask_x))
            self.MAPu_ini      = self.flow.bijector.inverse(self.MAPz_ini[:, self.istart_map:])

            self.amplitude_ini = self.MAPz_ini[:, 1]  # amplitude shift, applied to all spectra from the SN
            self.dtime_ini     = self.MAPz_ini[:, 0] # time shift, applied to all spectra from the SN (assumes telescope time correct, but possible overall offset)    

            
        self.bias_ini = tf.zeros([self.nsamples]) # bias shift, applied to all spectra from the SN
        self.bias = tf.Variable(self.bias_ini)
        self.bias.assign(self.bias_ini)

        self.amplitude = tf.Variable(self.amplitude_ini)
        self.amplitude.assign(self.amplitude_ini)

        self.dtime = tf.Variable(self.dtime_ini)
        self.dtime.assign(self.dtime_ini)

    def call(self, input_params):

        inds_start_uparam = 0
        if self.params['train_amplitude']:
            if self.params['train_dtime']:
                self.amplitude = input_params[:, 1]
            else:
                self.amplitude = input_params[:, 0]
            inds_start_uparam += 1

        if self.params['train_dtime']:
            self.dtime = input_params[:, 0]/self.params['dtime_norm']
            inds_start_uparam += 1

        self.MAPu = input_params[:, inds_start_uparam:]

        likelihood     = self.get_likelihood()
        latent_prior   = self.get_latent_prior()
        #         dtime_prior    = self.get_dtime_prior()

        # mask spectra that move outside of time range used for fitting
        #dtime_mask = self.get_dtime_mask()
        #n_spectra = tf.reduce_sum(dtime_mask[..., 0], axis=1)
        #n_spectra = self.n_spectra#tf.reduce_sum(dtime_mask[..., 0], axis=1)
        #print(MAP[:, -1])
        #print(n_spectra)
        #print(self.n_spectra/n_spectra)
        # use all spectra to get total log_prob 
#        log_posterior  = (latent_prior.log_prob(self.MAP)
#                          + tf.reduce_sum(likelihood.log_prob(self.x*self.mask_x)*self.mask_x[..., 0]*dtime_mask[..., 0], axis=1)/self.n_spectra) #+ dtime_prior.log_prob(self.dtime)

        log_posterior  = (latent_prior.log_prob(self.MAPu)
                          + tf.reduce_sum(likelihood.log_prob(self.x*self.mask_x)*self.use_spectra, axis=1)/self.n_spectra) #+ dtime_prior.log_prob(self.dtime)

#             # get log_prob for each spectra, so can ignore ones with worst fit
#             l_spec = likelihood.log_prob(x[:,:self.n_spectra])/self.n_spectra

#             # get worst fitting spectra
#             top_val, self.top_ind = tf.math.top_k(-l_spec, k=self.drop_top)
#             threshold        = tf.greater(l_spec, -top_val[0,-1])

#             # set worst fits to 0
#             l_spec = l_spec * tf.cast(threshold, dtype=tf.float32)

#             log_posterior = latent_prior.log_prob(self.MAP) + tf.reduce_sum(l_spec)


        return log_posterior

    def fwd_pass(self):  
        z_    = self.get_z()
        
        if self.params['use_amplitude']:
            # overall amplitude factor learned in Autoencoder
            # first latent variable is amplitude
            spec_ = self.decoder((z_, self.x_c, self.mask_x))

        else:
            # overall amplitude is external free paramater
            spec_ = self.amplitude[..., None, None] * self.decoder((z_, self.x_c, self.mask_x))

        spec_ = spec_ + self.bias[..., None, None]

        return spec_

    def get_z(self):    
        '''transform from latent space of normalizing flow (u) to latent space of autoencoder (z)'''
        
        self.MAPz = tf.concat([self.dtime[..., None],
                               self.amplitude[..., None],
                               self.flow.bijector.forward(self.MAPu)],
                              axis=1) 
            
        return self.MAPz
    
    def get_likelihood(self):
        spec_   = self.fwd_pass()

        # Measured average AE reconstruction error at current times
        sigmai_ = tf.transpose(tfp.math.interp_regular_1d_grid(
                        x=tf.transpose(self.x_c[...,0] + self.dtime[..., None]), 
                        x_ref_min=self.sigma_time_bin_cent[0], 
                        x_ref_max=self.sigma_time_bin_cent[-1], 
                        y_ref=self.sigma_time_grid))# fill_value='extrapolate')

#        sigma_     = tf.sqrt(sigmai_**2 + self.sigma_x**2)
        sigma_     = tf.sqrt( (spec_*sigmai_)**2 + self.sigma_x**2)
#        sigma_     = sigmai_

        # set missing values to 1 for all times
        sigma_ = sigma_*self.mask_x + (1-self.mask_x)

        # use all spectra to get total log_prob 
#         likelihood = tfd.Independent(tfd.MultivariateNormalDiag(loc=spec_, scale_diag=sigma_))

        # get log_prob for each spectra, so can ignore ones with worst fit
        likelihood = tfd.Independent(tfd.MultivariateNormalDiag(loc=spec_*self.mask_x,
                                                                scale_diag=sigma_),
                                     reinterpreted_batch_ndims=0)
        '''
        cond  = self.mask_x == 1.

        tf.print('ratio', spec_, self.x)
        tf.print('Mean', tf.where(cond, tf.math.log((tf.math.abs(spec_/self.x))), 0.))
        tf.print('sigma', tf.reduce_min(sigma_), tf.reduce_max(sigma_))
        likelihood = tfd.Independent(tfd.MultivariateNormalDiag(loc=tf.where(cond, tf.math.log(tf.math.abs(spec_/self.x)), 0.),
                                                                scale_diag=sigma_),
                                     reinterpreted_batch_ndims=0)
        '''
        return likelihood

    def get_latent_prior(self):
        '''automatically calulates along new batch dimension [..., latent_dim]'''
        return tfd.MultivariateNormalDiag(loc=tf.zeros(self.latent_dim_u),
                                          scale_diag=tf.ones(self.latent_dim_u), name='prior')

    def get_dtime_prior(self):
        dtime_mean = 0.0
        dtime_std  = 0.01 #0.01x

        return tfd.Normal(loc=dtime_mean, scale=dtime_std)

    def get_dtime_mask(self):
        # mask spectra if they fall outside fitted time range
        # such that they do not affect fits
        timei = self.x_c + self.dtime[..., None, None]

        dm = (timei >= 0) & (timei <= 1)

        dtime_mask = tf.where(dm, 1., 0.)

        return dtime_mask
    
    def get_amplitude_prior(self):
        amplitude_mean = 0.0
        amplitude_std  = 0.1

        return tfd.Normal(loc=amplitude_mean, scale=amplitude_std)


def get_hessian(model, map_parameters, trainable_params):
    """Get Hessian of model parameters for each SN"""
    nhess = map_parameters.shape[0]

    with tf.GradientTape(persistent=True) as tape:
        y = -model(map_parameters)#/10000
        print('y = ', y)

        grads = tape.gradient(y, [map_parameters])
        print('GRADS', grads)

    hessians = tape.batch_jacobian(grads[0], map_parameters, experimental_use_pfor=False)
#    hessians = tape.jacobian(grads, map_parameters, experimental_use_pfor=False)
#    print('hessians = ', hessians)
#    flattened_grads = tf.concat([tf.reshape(grad, [nhess, -1]) for grad in grads], axis=1)
#    print('flattened_grads  = ', flattened_grads)

    print('hessians = ', hessians)
    # split hessians by sample
    hess_final = np.zeros([nhess, tf.shape(flattened_grads)[1], tf.shape(flattened_grads)[1]]).astype(np.float32)
    for ihess in range(nhess):
        hess_i = [hess[ihess,:,ihess:ihess+1] for hess in hessians]
        hess_final[ihess] = tf.concat([tf.reshape(hess, [hess.shape[0], -1]) for hess in hess_i], 1)

    return tf.convert_to_tensor(hess_final)
    '''
    return hessians
    '''
def get_hessian_paramlist(model, map_parameters, trainable_params):
    """Get Hessian of model parameters for each SN"""
    print(map_parameters.shape)
    nhess = map_parameters.shape[0]
    with tf.GradientTape(persistent=True) as tape:
        y = -model(map_parameters)

        grads = tape.gradient(y, trainable_params)
        tf.print(tf.shape(grads))
        flattened_grads = tf.concat([tf.reshape(grad, [nhess, -1]) for grad in grads], axis=1)

    hessians = tape.jacobian(flattened_grads, trainable_params, experimental_use_pfor=False)

    # split hessians by sample
    hess_final = np.zeros([nhess, tf.shape(flattened_grads)[1], tf.shape(flattened_grads)[1]]).astype(np.float32)
    for ihess in range(nhess):
        hess_i = [hess[ihess,:,ihess:ihess+1] for hess in hessians]
        hess_final[ihess] = tf.concat([tf.reshape(hess, [hess.shape[0], -1]) for hess in hess_i], 1)

    return tf.convert_to_tensor(hess_final)


# get samples in both u and z around MAP
# matrix not always positive definitate. Re-finding MAP usually helps
def sample_from_hessian(model, hess, trainable_params, iplt, nsamp=10000):
    '''get samples in both z_latent and u_latent around MAP
    matrix not always positive definitate. Re-finding MAP usually helps
    '''
    
    covi       = np.linalg.inv(hess)
    
    samples = []
    for jj in range(nsamp):
        samples.append(np.dot(np.linalg.cholesky(covi), np.random.randn(hess.shape[-1])))

    samples = np.asarray(samples).astype(np.float32)

    lu = model.MAPu.shape[1]

    shift   = np.concatenate([i[iplt].numpy().flatten() for i in trainable_params])
    samples += shift
    
    samples_z = samples.copy()
    samples_z[:,:lu] = model.flow.bijector.forward(samples[:, :lu]).numpy()

    return samples, samples_z


def setup_trainable_parameters(model, params):        
    trainable_params       = []
    trainable_params_label = []
    
    if params['train_MAP']:
        trainable_params.append(model.MAPu)
        trainable_params_label.append('MAP')
    if params['train_amplitude']:
        trainable_params.append(model.amplitude)
        trainable_params_label.append('amplitude')
    if params['train_dtime']:
        trainable_params.append(model.dtime)
        trainable_params_label.append('dtime')
    if params['train_bias']:
        trainable_params.append(model.bias)
        trainable_params_label.append('bias')   

    return trainable_params, trainable_params_label
