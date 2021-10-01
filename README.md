# suPAErnova
A probabilistic autoencoder for type Ia supernovae. Coded in tensorflow.

## Workflow is in three main codes:
1.) train_ae.py:
    Trains the autoencoder based on setup detailed in training configuration file, config/train.yaml.
    Models are saved to tensorflow_models/
    
2.) train_flow.py:
    Trains the flow based on setup detailed in training configuration file, config/train.yaml 
    Models are saved to tensorflow_models/

3.) posterior_analysis.py:
    Runs posterior analysis based on setup detailed in training configuration file, config/posterior_analysis.yaml
    Outputs are saved to data/

## Codebase:

models/: contains machine learning models, loading functions, and training loss updates

	autoencoder.py:
		autoencoder model
	flow.py:
		flow model
	posterior.py:
		posterior analysis model
	losses.py:
		various losses for autoencoder training
	loader.py:
		load in models

utils/: contains functionality to load in data and perform a few calculations


	
