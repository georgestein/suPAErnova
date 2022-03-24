# suPAErnova
This repository contains the codes required to the train models and perform analyses for *A Probabilistic Autoencoder for Type Ia Supernovae Spectral Time Series*. Constructed in TensorFlow 2 and TensorFlow Probability.

![alt text](figures/network_illustration.png)

## Installation
Install the package reqirements with conda

`conda env create -f environment.yml`

Activate conda environment, and install suPAErnova package in your python environment

`python setup.py`


## Training a new PAE model on your dataset
This requires 3 steps:

1.) **suPAErnova/make_datasets/make\_train\_test\_data.py:**
Individual spectra from each supernova first need to be reshaped along the time dimension, as the PAE model requires training data of dimensionality (N\_SN, N\_timesteps, N\_wavelengths), with a corresponding mask array to denote any missing spectra.

2.) **scripts/train\_ae.py:**
    Trains the autoencoder based on setup detailed in training configuration file, *config/train.yaml*.
    Models are saved to outputs/tensorflow_models/
    
2.) **scripts/train\_flow.py:**
    Trains the flow based on setup detailed in training configuration file, *config/train.yaml* 
    Models are saved to *outputs/tensorflow_models/*

## Performing inference with a trained model
**scripts/run\_posterior\_analysis.py:**
    Runs posterior analysis based on setup detailed in training configuration file, *config/posterior_analysis.yaml*. Outputs are saved to *outputs/*

## Codebase:

suPAErnova/models/: contains custom machine learning models, loading functions, and training loss updates

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


	
