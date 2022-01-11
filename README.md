# suPAErnova
Codes for *A Probabilistic Autoencoder for Type Ia Supernovae Spectral Time Series*. 

Coded in tensorflow 2 and tensorflow probability.

### Training a new model on your dataset consists of 4 main steps:
1.) Reshaping SNe spectral timeseries to dimensions (N\_SN, N\_timesteps, N\_wavelengths), with a corresponding mask array to denote any missing spectra. See suPAErnova/make_datasets/make\_train\_test\_data.py for an example of how to do so.


2.) scripts/train\_ae.py:
    Trains the autoencoder based on setup detailed in training configuration file, config/train.yaml.
    Models are saved to outputs/tensorflow_models/
    
2.) scripts/train\_flow.py:
    Trains the flow based on setup detailed in training configuration file, config/train.yaml 
    Models are saved to outputs/tensorflow_models/

3.) scripts/run\_posterior\_analysis.py:
    Runs posterior analysis based on setup detailed in training configuration file, config/posterior_analysis.yaml
    Outputs are saved to outputs/

### Codebase:

suPAErnova/models/: contains machine learning models, loading functions, and training loss updates

	autoencoder.py:
		Autoencoder model
	autoencoder_training.py:
		Training functions for autoencoder model
	flows.py:
		Flow model
	flow_training.py:
		Training functions for flow model
	posterior.py:
		posterior analysis setup
	posterior_analysis.py:
		Functions to run posterior analysis
	losses.py:
		Various losses for autoencoder training
	loader.py:
		Load in models

suPAErnova/utils/: contains functionality to load in data and perform a few calculations


	
