# Protein nanobarcodes enable single-step multiplexed fluorescence imaging

This repository holds the custom codes for deep-learning-based recognition of "protein nanobarcodes" from confocal micrographs.

Please use the Jupyter Notebook file ```./nanobarcode_classifier_net.ipynb``` to run the code for the provided examples. 

The notebook contains the code necessary for setting up the deep neural network as well as the dataset manager that are used in the manuscript for producing the reported results.

In the current setup of the code, which does not repeat the training but uses the weights from an already trained version, the whole notebook should run within a minute time, and produce plots of prediction accuracies, false positive/negatives (similar to Figure 2c of the manuscript) as well as false-color output of micrographs with nanobarcodes highlighted (similar to Extended Data Figure 6 of the manuscript).

Current setup of the code loads hyperparameters and network weights from the ```./data directory```. The flags in the code can easily be changed to retrain and save the network.

This notebook depends on local installations of these libraries

 - **PyTorch** with CUDA enabled for network training/prediction on the GPU. The code is compatible with PyTorch version > 1.9 with cudatoolkit version > 11.1. See <a href="https://pytorch.org/get-started/locally">https://pytorch.org/get-started/locally</a> for instructions.
 - **Jupyter** with Python version 3.7 or newer.
 - **numpy** for arrays and numerical computations
 - **scipy** for signal processing functions
 - **matplotlib** for plotting
 - **scikit-image** for image processing
 - **tifffile** for image IO
 - **tqdm** for progress tracking

This notebook additionally relies on the included custom library ```./dnn_classifier``` for network and data handler components.

Running this code requires a desktop PC with a graphics card containing an NVIDIA GPU.

The local installation of all the required components should take < 10 min, if no problems with hardware compatibility and driver availability arise.

We recommend making a conda environment for installing all the necessary components and running the notebook. See <a href="https://docs.conda.io/en/latest/miniconda.html">https://docs.conda.io/en/latest/miniconda.html</a> for instructions.

## Reference
[1] de Jong-Bolm, Sadeghi, Bao, Klaehn, Hoff, Mittelmeier, Basmanav, Opazo, Noé, Rizzoli, “<a href="https://doi.org/10.1101/2022.06.03.494744">Protein nanobarcodes enable single-step multiplexed fluorescence imaging</a>”, bioRxiv (2022) 494744.

