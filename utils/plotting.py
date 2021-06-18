import matplotlib.pyplot as plt
import seaborn as sns

def plot_spectra_map(model, params, observations, data_fit, title='Training', ispec=0, train=True):
    
    dm = observations['mask'][ispec, :, 0] != 0

    x_obs = observations['spectra'][ispec, dm].copy()
    x_ae  = data_fit[ispec, 0, dm].copy() # AE reconstructed spectra
    x_map = data_fit[ispec, -1, dm].copy() # MAP spectra

    sigma  =  observations['sigma'][ispec, dm].copy()
    times  =  observations['times'][ispec, dm].copy()

    wavelengths = observations['wavelengths']
    red = observations['redshifts'][ispec]
    
    n_spectra = x_obs.shape[0]
    print(n_spectra)
    # Plot aesthetics
    plttypes = ['normal', 'residual']
    aoffsets = [0.4, 0.15]
    
    cmap = plt.cm.coolwarm
    colors = cmap(np.linspace(0, 1, len(x_obs)))

    lwt = 2
    lw  = 3
    
    c0  = 'orangered'
    c1  = 'C0'
    
    alpha=0.8

    wave_txt =wavelengths[-1]+100

    # Model uncertainty
    sigma_ae = tf.transpose(tfp.math.interp_regular_1d_grid(
                x=tf.transpose(times[:, 0] + model.dtime[ispec]), x_ref_min=model.sigma_time_bin_cent[0], x_ref_max=model.sigma_time_bin_cent[-1], y_ref=model.sigma_time_grid))# fill_value='extrapolate')

    # Data uncertainty
    sigmai = tf.sqrt(sigma_ae**2 + sigma**2)
    

    for iplt, plttype in enumerate(plttypes):
        fig, ((ax1 )) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(12,10))#, gridspec_kw={'height_ratios': [3, 1]})

        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(1.5)

        ax1.tick_params('both', length=6, width=1.5, which='major', color='k',direction='in')                                         
        ax1.tick_params('both', length=3, width=1.5, which='minor', color='k',direction='in')

        offset = np.arange(len(times)) * -1 * aoffsets[iplt]
        ymin = 100
        ymax = 0
        for i in range(n_spectra):
            labp = None
            labs = None
            labo = None
            ls= '-'

            if i==0:
                labo = 'Observed spectra'
                labs = 'Encoder'
                labp = 'MAP'
                
            if plttype == 'normal':
                xi = x_obs[i] + offset[i]
                xaei = x_ae[i] + offset[i]
                xmapi = x_map[i] + offset[i]

                ax1.plot(wavelengths, xi, '-', c='k', lw=lwt, alpha=0.8, label=labo)#colors[i])
                ax1.fill_between(wavelengths, xi-sigma[i],  xi+sigma[i], color='gray', alpha=0.8, lw=1)
                ax1.plot(wavelengths, xaei, '-', c=c1, lw=lw, alpha=alpha, label=labs)
                ax1.plot(wavelengths, xmapi, ls=ls, c=c0, lw=lw, alpha=alpha, label=labp)

                ax1.text(wave_txt, offset[i], '{:.2f} days'.format(times[i][0]), fontsize=16)

                ymin = min(ymin, min(xi), min(xaei), min(xmapi) )
                ymax = max(ymax, max(xi), max(xaei), max(xmapi) )
                    
                    
            if plttype == 'residual':
                xi = x_obs[i]
                xaei = x_ae[i]
                xmapi = x_map[i]

                ax1.plot(wavelengths, wavelengths*0+offset[i], '-', c='k', lw=lwt, alpha=0.8, label=labo)#colors[i])
                ax1.fill_between(wavelengths, offset[i]-sigma[i],  offset[i]+sigma[i], color='gray', alpha=0.8, lw=1)
                ax1.plot(wavelengths, xaei - xi + offset[i], '-', c=c1, lw=lw, alpha=alpha, label=labs)
                ax1.plot(wavelengths, xmapi - xi + offset[i], ls=ls, c=c0, lw=lw, alpha=alpha, label=labp)

                ax1.text(wave_txt, offset[i], '{:.2f} days'.format(times[i][0]), fontsize=16)

                ymin = min(ymin, min(xaei - xi + offset[i]), min(xmapi - xi + offset[i]) )
                ymax = max(ymax, max(xaei - xi + offset[i]), max(xmapi - xi + offset[i]) )

  
        ax1.set_ylabel('Restframe Flux')
        ax1.set_xlabel('Wavelength [$\AA$]')
        ax1.set_ylabel('Normalized Restframe Flux(t, $\lambda$)')
        
        if plttype == 'normal':      
            ax1.set_title(title+', z={:.4f}'.format(red))
            ax1.set_ylim(ymin-0.1, ymax)
            
        if plttype == 'residual':        
            ax1.set_title(title+' residual, z={:.4f}'.format(red))
            ax1.set_ylim(ymin-0.1, ymax+0.1)
            
        ax1.set_frame_on(False)
        ax1.get_xaxis().tick_bottom()
        ax1.axes.get_yaxis().set_visible(False)
        xmin, xmax = ax1.get_xaxis().get_view_interval()
        ymin, ymax = ax1.get_yaxis().get_view_interval()
        ax1.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))

        ax1.legend(loc='upper right', frameon=False, ncol=3)
        
        

        ax1.set_xlim(wavelengths[0], wavelengths[-1])

        
def plot_samples_hessian(params, observations, samples, ispec=0):
    """seaborn plot of Hessian samples"""

    nspec = np.sum(observations['mask'][ispec, :, 0] != 0)
    redshift = observations['redshifts'][ispec]
    
    # Put samples in dictionary
    df = {}
    df['time']      = samples_u[:, -1]
    df['amplitude'] = samples_u[:, -2] 

    # plot
    g = sns.jointplot(x='time', y='amplitude', data=df, kind='kde')
    g.set_axis_labels("$\Delta$time [days] (Nspec={:d})\n{:.3f}$\pm${:.3f}".format(int(nspec), df['time'].mean(),
                                                                    df['time'].std()),
                      "Amplitude (z={:.3f})\n{:.3f}$\pm${:.4f}".format(redshift, df['amplitude'].mean(),
                                                                df['amplitude'].std()),
                     fontsize=18);
    
