import numpy as np

def compute_sigma_ae_time(spec_true, spec_pred, sigma, time, mask, weighted=False, outlier_cut=100, relative=True):
    """Calculate std of true and reconstructed spectra as a function of time

    Parameters
    ----------
    spec_true: array (N_sn, n_timesamples, data_dim) 
       measured spectra
    spec_pred: array (N_sn, n_timesamples, data_dim) 
       model spectra
    sigma: array (N_sn, n_timesamples, data_dim) 
       measurement uncertainty
    time: array (N_sn, n_timesamples) 
       observation time
    """

    relative = True
    dm = mask[:, :, 0] == 1.
    print('DEBUG ', dm.shape, spec_true.shape)
    ntbins = 11
    t_bin_edge = np.linspace(0, 1, ntbins+1)
    t_bin_cent = (t_bin_edge[:-1] + t_bin_edge[1:])/2

    s0 = spec_true[dm].copy()
    s1 = spec_pred[dm].copy()  
    sig = sigma[dm].copy()

    t   = time[dm][:,0]
    
    s0 = np.reshape(s0, (-1, 288))
    s1 = np.reshape(s1, (-1, 288))
    sig = np.reshape(sig, (-1, 288))

    mse = (s0 - s1)
    if relative:
        mse = np.abs((s0 - s1))/s1

    #mse = np.log(np.abs(s1/s0))
    #mse = np.abs(s1/(s0+1e-9)) - 1

    if weighted:
        mse /= sig**2
        
    # sometimes there are huge outliers,
    # so set errors larger than some percentile to the value corresponding to that percentile
    sigma_t = np.zeros((mse.shape[1], len(t_bin_cent)))
    bins = np.digitize(t, t_bin_edge) - 1
    for ibin in np.unique(bins):
        dm   = bins == ibin
        msei = mse[dm].copy()
        maxval = np.percentile(msei, outlier_cut, axis=0)
        maxval_tile = np.tile(maxval, (msei.shape[0], 1))

        indgt = np.greater(msei, maxval)

        # mask values greater than outlier_cut
        mask = np.ones(msei.shape)
        mask[indgt] = 0. 

        if not relative:
            minval = np.percentile(msei, 100-outlier_cut, axis=0)
            minval_tile = np.tile(minval, (msei.shape[0], 1))
        
            indlt = np.less(msei, minval)
            mask[indlt] = 0. 
            
            # take std of non masked
            sigma_t[:, ibin] = np.sum(  (msei - np.sum(msei*mask, axis=0)/np.sum(mask, axis=0))**2*mask, axis=0)/np.sum(mask, axis=0)

        if relative:
            # take mean of non masked
            sigma_t[:, ibin] = np.sum(msei * mask, axis=0)/np.sum(mask, axis=0)

    if not relative:
        sigma_t = np.sqrt(sigma_t)
    
    return sigma_t.astype(np.float32), t_bin_edge.astype(np.float32), t_bin_cent.astype(np.float32)

