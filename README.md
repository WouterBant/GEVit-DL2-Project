# E(2) Equivariant Attention Models for Image Classification 

### Authors: Wouter Bant, Colin Bot, Jasper Eppink, Clio Feng, Floris Six Dijkstra


## Introduction
Inspired by the group equivariant attention model presented in [Group Equivariant Vision Transformer](https://openreview.net/forum?id=uVG_7x41bN) (GE-ViT), we conduct experiments to validate the performance of the presented model. We provide many visualizations for a better understanding of GE-ViTs and other presented methods. Furthermore, we present and evaluate several ways of making non-equivariant models equivariant by combining the latent embeddings or probabilities of different transformed inputs. We also speed up the experiments with GE-ViT by first projecting the image to an artificial image with smaller spatial resolution. 

<table align="center">
  <tr align="center">
      <td><img src="figures/Rotation.gif" width=400></td>
  </tr>
</table>

For the full analysis see [our blogpost](Blogpost.md), but to give a little preview:

- 👓 We visualize many layers of the [Group Equivariant Vision Transformer](https://openreview.net/forum?id=uVG_7x41bN) (GE-ViT)
<table align="center">
  <tr align="center">
      <td><img src="figures/GEVIT_latent_representations_2.gif" width=560></td>
  </tr>
</table>

- 🎯 Evaluate and propose novel ways of making any image classification model globally E(2) equivariant and beat previous image classification benchmarks:

<table align="center">
  <tr align="center">
      <td><img src="figures/posthocaggregation.png" width="600"></td>
      <td><img src="figures/meanagg_vs_mostprobable.png" width="657"></td>
  </tr>
</table>


- ⚡ Speed up GE-ViTs by projecting the image to an artificial image with lower spatial resolution for less attention computations:
<table align="center">
  <tr align="center">
      <td><img src="figures/modern_eq_vit.png" width=600></td>
  </tr>
</table>
<table align="center">
  <tr align="center">
      <td><img src="figures/hybrid.png" width=600></td>
  </tr>
</table>

## Reproducing results

### Installation

#### Getting the code
Clone the repository:

```bash
git clone https://github.com/WouterBant/GEVit-DL2-Project.git
```

And go inside the directory:
```bash
cd GEVit-DL2-Project
```

#### Getting the environment
Unfortunately we had to use two different environments. For running the GE-ViT you can create the environment with:

```bash
conda env create -f gevit_conda_env.yml 
```

```bash
conda activate gevit
```

For running the post hoc experiments and training of the equivariant modern ViT:

```bash
conda env create -f posthoc_conda_env.yml
```

```bash
conda activate posthoc
```

### Demos
In the [demos](demos) folder we provide notebooks for visualizing the artifacts for non 90 degree rotations and creating the video comparing normal ViT to equivariant models for rotated inputs.

### Running experiments

For reproducing the results for GE-ViT, change directory to the [src](src) folder and execute the commands from the [README](src/README.md).

For reproducing the results of the post hoc experiments, change directory to [src/post_hoc_equivariant](src/post_hoc_equivariance/) and follow the instructions from the [README](src/post_hoc_equivariance/README.md). Also, in this folder [checkpoints](src/post_hoc_equivariance/checkpoints) and [results](src/post_hoc_equivariance/results) are saved for various models.

For reproducing the results of the modern equivariant ViT, change directory to [src/modern_eq_vit](src/modern_eq_vit/) and refer to the [README](src/modern_eq_vit/README.md) for instructions to run the training.

## Important files

- The blogpost can be found in [Blogpost.md](Blogpost.md)
- Jupyter notebooks with small experiments can be found in [demos/](demos)
- All the other code can be found in [src/](src)
  * [src/README.md](src/README.md): instructions for running GE-ViT experiments
  * [src/modern_eq_vit](src/modern_eq_vit/): code for down sampling images for faster GE-ViT experiments
    * [src/modern_eq_vit/README.md](src/modern_eq_vit/README.md): instructions to reproduce these experiments
    * [src/modern_eq_vit/eq_modern_vit.py](src/modern_eq_vit/eq_modern_vit.py): implementation for this part
  * [src/post_hoc_equivariance](src/post_hoc_equivariance/): post hoc equivariant methods
    * [src/post_hoc_equivariance/README.md](src/post_hoc_equivariance/README.md): instructions to reproduce these experiments
    * [src/post_hoc_equivariance/post_hoc_equivariant.py](src/post_hoc_equivariance/post_hoc_equivariant.py): implementation of the different methods
  * [src/models/](src/models/): various models
    * [src/models/gcnn.py](src/models/gcnn.py): implementation of the Group Convolutional Neural Network

## Acknowledgements
This repository contains the source code accompanying the paper: [Group Equivariant Vision Transformer](https://openreview.net/forum?id=uVG_7x41bN),  UAI 2023.

The original code, containing a small error, is from the [GSA-Nets](https://openreview.net/forum?id=JkfYjnOEo6M) paper by David W. Romero and Jean-Baptiste Cordonnier.
