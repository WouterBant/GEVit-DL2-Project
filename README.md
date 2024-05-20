# E(2) (Post-Hoc) Equivariant (Attention) Models for Image Classification 

### Wouter Bant, Colin Bot, Jasper Eppink, Clio Feng, Floris Six Dijkstra

Inspired by the equivariant attention model present in [Group Equivariant Vision Transformer](https://openreview.net/forum?id=uVG_7x41bN),  UAI 2023, we conduct experiments to validate the performance of the presented model. Furthermore, we present and evaluate several ways of making non-equivariant models equivariant by test time augmentations. Building on the GE-ViT, we make the modern ViT equivariant. Also, we provide many visualizations for a better understanding of GE-ViTs and other presented methods.

For the full analysis see [our blogpost](Blogpost.md), but to give a little preview:

- 👓 We visualize many layers of the [Group Equivariant Vision Transformer](https://openreview.net/forum?id=uVG_7x41bN) (GE-ViT)
<table align="center">
  <tr align="center">
      <td><img src="figures/GEVIT_latent_representations_2.gif" width=600></td>
  </tr>
</table>

- 🎯 Evaluate and propose novel ways of making any image classification model globally E(2) equivariant and beat previous image classification benchmarks:

<table align="center">
  <tr align="center">
      <td><img src="figures/posthocaggregation.png" width=600></td>
  </tr>
</table>

<table align="center">
  <tr align="center">
      <td><img src="figures/meanagg_vs_mostprobable.png" width=600></td>
  </tr>
</table>

- ⚡ Speed up GE-ViTs by projecting the image to an artificial image with lower spatial resolution for less attention computations:
<table align="center">
  <tr align="center">
      <td><img src="figures/modern_eq_vit.png" width=600></td>
  </tr>
</table>


### Reproducing results

#### Installation

##### Getting the code
Clone the repository:

```bash
git clone https://github.com/WouterBant/GEVit-DL2-Project.git
```

And go inside the directory:
```bash
cd GEVit-DL2-Project
```

##### Getting the environment
Unfortunately we had to use two different environments. For running the GE-ViT you can create the environment with:

```bash
conda create -f gevit_conda_env.yml
```

```bash
conda activate gevit
```

For running the post hoc experiments and training of the equivariant modern ViT:

```bash
conda create -f posthoc_conda_env.yml
```

```bash
conda activate posthoc
```

##### Demos
In the [demos](demos) folder we provide notebooks for visualizing the artifacts for non 90 degree rotations and creating the video comparing normal ViT to equivariant models for rotated inputs.

##### Reproducing results

For reproducing the results for GE-ViT, change directory to the [src](src) folder and execute the commands from the [README](src/README.md).

For reproducing the results of the post hoc experiments, change directory to [src/post_hoc_equivariant](src/post_hoc_equivariance/) and follow the instructions from the [README](src/post_hoc_equivariance/README.md). Also, in this folder [checkpoints](src/post_hoc_equivariance/checkpoints) and [results](src/post_hoc_equivariance/results) are saved for various models.

### Acknowledgements
This repository contains the source code accompanying the paper: [Group Equivariant Vision Transformer](https://openreview.net/forum?id=uVG_7x41bN),  UAI 2023.

The original code, containing a small error, is from the [GSA-Nets](https://openreview.net/forum?id=JkfYjnOEo6M) paper by David W. Romero and Jean-Baptiste Cordonnier.
