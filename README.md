# Deep-Nanobarcode

This repository contains the software Deep-Nanobarcode (in the form of a Python package accompanied by jupyter notebooks)
 for deep-learning-based identification of "protein nanobarcodes" from confocal fluorescence microscopy images.

The software relies on a PyTorch backend. To benefit from the GPU-accelerated deep learning provided by Pytorch, 
you should run the software on a desktop PC equipped with a graphics card with an NVIDIA GPU.

When you use the software, please cite the following preprint:

```
@article
{
	author = {{De Jong-Bolm}, Dani{\"{e}}lle and Sadeghi, Mohsen and
	 Bao, Guobin and Klaehn, Gabriele and Hoff, Merle and Mittelmeier, Lucas and
	  {Buket Basmanav}, F and Opazo, Felipe and No{\'{e}}, Frank and Rizzoli, Silvio O},
	doi = {10.1101/2022.06.03.494744},
	journal = {bioRxiv},
	pages = {2022.06.03.494744},
	publisher = {Cold Spring Harbor Laboratory},
	title = {{Protein nanobarcodes enable single-step multiplexed fluorescence imaging}},
	url = {https://www.biorxiv.org/content/10.1101/2022.06.03.494744v1},
	doi = {10.1101/2022.06.03.494744}
	year = {2022}
}

```



Installation
---

We recommend using a conda environment for installing all the necessary components and running the notebooks.
See https://docs.conda.io/en/latest/miniconda for instructions on how to set up a local ```miniconda``` installation.

With miniconda installed on your machine, you can create a dedicated conda environment for the software and all its dependencies.
To do so, run the following command in terminal (substitute ```your_environment_name``` with a name of your choosing),

```
conda create -n your_environment_name python
```

Switch to the newly created environment:

```
conda activate your_environment_name
```

**Hint**: Further steps of the installation uses ```pip``` package manager.
It is good practice to check beforehand if the ```pip``` instance is invoked from the correct environment by
looking at the output of ```which pip``` and making sure that it points to the folder made for your new conda environment.

Having the conda environment up and running, you can easily proceed with the software installation in two steps:

1. Clone a local copy of the repository from **GitHub**. Navigate to a folder you intend to download the software in and run the following command:

```
git clone https://github.com/noegroup/deep_nanobarcode.git
```

2. Navigate to the folder containing the cloned repository (e.g. ```cd deep_nanobarcode```) and install the software package with the following command,

```
pip install --upgrade .
```

if you are interested in further development of the software, you can install the package in the development (aka editable) mode,

```
pip install --upgrade --editable .
```

If everything goes according to plan, you should not need to do anything else, and you can start using the software.
The local installation of all the required components should take < 5 min, if no problems with hardware compatibility and driver availability arise.

The most common problem would be with GPU drivers and their compatibility with PyTorch.
Make sure you have the latest NVIDIA drivers already installed on your machine.
You can check your current driver version via the ```nvidia-smi``` command.
NVIDIA drivers can be downloaded from https://www.nvidia.com/download/index.aspx.


Application
---

The two Jupyter Notebook files

- ```./notebooks/nanobarcode_classifier_net.ipynb```
- ```./notebooks/tomographic_reconstruction.ipynb```

contain the scripts necessary for processing confocal images with Deep-Nanobarcode
To access these notebooks, navigate to the ```notebooks``` folder and run,

```
jupyter notebook
```

This should open a browser window with a tree view of the two notebooks.

The first notebook, ```./notebooks/nanobarcode_classifier_net.ipynb```, contains all the code necessary for application of the already trained deep neural network.
The notebook contains the code necessary for setting up the deep neural network as well as the dataset manager that are used in the manuscript for producing the reported results.
In the current setup of the code, which does not repeat the training but uses the weights from an already trained version, the whole notebook should run in a few minutes, and produce reports of prediction accuracy, precision, recall, F1-score, and false positive/negatives on the test dataset (similar to Figure 2c of the manuscript) as well as false-color output of micrographs with nanobarcodes highlighted (similar to Extended Data Figure 6 of the manuscript).

Some example images are included in the ```./examples``` folder. The rest of the images used in training and evaluation of the network are publicly available from <a href="http://dx.doi.org/10.17169/refubium-39512">this 
Refubium repository</a>.

If you are interested in tomographic reconstruction of nanobarcodes identificed from a z-stack, you can use the second notebook, ```./notebooks/tomographic_reconstruction.ipynb```.

Deep network training
---

Using this software does not require tweaking/training the deep network.
We have already obtained optimal network hyperparameters and have also fully trained the network using the protocol described in the preprint.
Network parameters are already included in the repository in the ```./network_params``` folder.
When you run the ```./notebooks/nanobarcode_classifier_net.ipynb``` notebook for the first time,
copies of saved network weights are automatically downloaded to the same folder.

In case you think your application can benefit from retraining the network on a new dataset,
 you can use notebook ```./notebooks/nanobarcode_classifier_net.ipynb```, but you have to change the ```train_net``` flag. 
 Some information about the training procedure is given in the notebook. For more detailes, please contact the authors.

Dependencies
---

 - **PyTorch** with CUDA enabled for network training/prediction on the GPU. The code is compatible with PyTorch version > 1.9 with cudatoolkit version > 11.1. See <a href="https://pytorch.org/get-started/locally">https://pytorch.org/get-started/locally</a> for instructions.
 - **Jupyter** with Python version 3.7 or newer.
 - **numpy** for arrays and numerical computations
 - **scipy** for signal processing functions
 - **matplotlib** for plotting
 - **scikit-image** for image processing
 - **tifffile** for image IO
 - **tqdm** for progress tracking

### Copyright

Copyright (c) 2022-2023, Mohsen Sadeghi (mohsen.sadeghi@fu-berlin),
Artificial Intelligence for the Sciences Group (AI4Science),
Freie Universität Berlin, Germany.

