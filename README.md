## Group Equivariant Vision Transformer

This repository contains the source code accompanying the paper:

 [Group Equivariant Vision Transformer](https://openreview.net/forum?id=uVG_7x41bN),  UAI 2023.
 
 Code Author: [Kaifan Yang](https://github.com/ZJUCDSYangKaifan/) & [Ke Liu](https://github.com/zjuKeLiu)

#### TODO's
- Make interpretability plots for [our demo](demo/demo_only_3_and_8.ipynb), make sure the code can easily be applied to other experiments. For inspiration see the original ViT paper
- Decide what other experiments are interesting
- Make an extension

### Reproducing experimental results

#### Command for running rot-MNIST
```bash
python run_experiment.py --config.dataset rotMNIST --config.model p4sa --config.norm_type LayerNorm --config.attention_type Local --config.activation_function Swish --config.patch_size 9 --config.dropout_att 0.1 --config.dropout_values 0.1 --config.whitening_scale 1.41421356 --config.epochs 300 --config.optimizer Adam --config.lr 0.001 --config.optimizer_momentum 0.9 --config.scheduler constants --config.sched_decay_steps='(1000,)' --config.sched_decay_factor 1.0 --config.weight_decay 0.0001 --config.batch_size 8 --config.device cuda --config.seed 0 --config.comment ''
```
#### running rot-MNIST with less datafraction
```bash
python run_experiment.py --config.dataset rotMNIST --config.model p4sa --config.norm_type LayerNorm --config.attention_type Local --config.activation_function Swish --config.patch_size 9 --config.dropout_att 0.1 --config.dropout_values 0.1 --config.whitening_scale 1.41421356 --config.epochs 300 --config.optimizer Adam --config.lr 0.001 --config.optimizer_momentum 0.9 --config.scheduler constants --config.sched_decay_steps='(1000,)' --config.sched_decay_factor 1.0 --config.weight_decay 0.0001 --config.batch_size 8 --config.device cuda --config.seed 0 --config.comment '' --config.data_fraction 0.1
```
#### running rot-MNIST with only the 3 and 8 images
```bash
python run_experiment.py --config.dataset rotMNIST --config.model p4msa --config.norm_type LayerNorm --config.attention_type Local --config.activation_function Swish --config.patch_size 9 --config.dropout_att 0.1 --config.dropout_values 0.1 --config.whitening_scale 1.41421356 --config.epochs 300 --config.optimizer Adam --config.lr 0.001 --config.optimizer_momentum 0.9 --config.scheduler constants --config.sched_decay_steps='(1000,)' --config.sched_decay_factor 1.0 --config.weight_decay 0.0001 --config.batch_size 8 --config.device cuda --config.seed 0 --config.comment '' --config.only_3_and_8 True
```
#### Command for running  CIFAR-10
```bash
python run_experiment.py --config.dataset CIFAR10 --config.model mz2sa --config.norm_type LayerNorm --config.attention_type Local --config.activation_function Swish --config.patch_size 5 --config.dropout_att 0.1 --config.dropout_values 0.0 --config.whitening_scale 1.41421356 --config.epochs 350 --config.optimizer SGD --config.lr=0.01 --config.optimizer_momentum 0.9 --config.scheduler linear_warmup_cosine --config.sched_decay_steps='(1000,)' --config.sched_decay_factor 1.0 --config.weight_decay 0.0001 --config.batch_size 24 --config.device cuda --config.seed 0 --config.comment ""
```

#### Command for running PatchCamelyon
```bash
python run_experiment.py --config.dataset PCam --config.model p4sa --config.norm_type LayerNorm --config.attention_type Local --config.activation_function Swish --config.patch_size 5 --config.dropout_att 0.1 --config.dropout_values 0.1 --config.whitening_scale 1.41421356 --config.epochs 100 --config.optimizer SGD --config.lr 0.01 --config.optimizer_momentum 0.9 --config.scheduler linear_warmup_cosine --config.sched_decay_steps='(1000,)' --config.sched_decay_factor 1.0 --config.weight_decay 0.0001 --config.batch_size 16 --config.device cuda --config.seed 0 --config.comment ""
```
# running PatchCamelyon with less datafraction
```bash
python run_experiment.py --config.dataset PCam --config.model p4sa --config.norm_type LayerNorm --config.attention_type Local --config.activation_function Swish --config.patch_size 5 --config.dropout_att 0.1 --config.dropout_values 0.1 --config.whitening_scale 1.41421356 --config.epochs 100 --config.optimizer SGD --config.lr 0.01 --config.optimizer_momentum 0.9 --config.scheduler linear_warmup_cosine --config.sched_decay_steps='(1000,)' --config.sched_decay_factor 1.0 --config.weight_decay 0.0001 --config.batch_size 16 --config.device cuda --config.seed 0 --config.comment "" --config.data_fraction 0.001
```

### Note
Our code was modified based on the code presented in paper A. We mainly modified the “construct_relative_positions” function of the g_selfatt/groups/SE2.py and g_selfatt/g_selfatt/groups/E2.py module in [GSA-Nets](https://github.com/dwromero/g_selfatt) which corresponds to the part of position encoding. 

From the experimental results, there are differences between the results in our paper and those in [GSA-Nets](https://openreview.net/forum?id=JkfYjnOEo6M) and we suspect that this is caused by differences in the experimental environment. The paper of [GSA-Nets](https://openreview.net/forum?id=JkfYjnOEo6M) uses NVIDIA TITAN RTX, while we used NVIDIA Tesla A100. To ensure a fair comparison, we re-ran the code of [GSA-Nets](https://openreview.net/forum?id=JkfYjnOEo6M) on our hardware. 

The experimental results of rot-MNIST and PatchCamelyon are similar to those presented in the paper, but the results of CIFAR-10 differ significantly from the paper. It is worth mentioning that in the [GSA-Nets](https://openreview.net/forum?id=JkfYjnOEo6M), the authors mentioned that they did not use automatic mixed precision when conducting experiments on the CIFAR-10 datasets. However, when we tried to run the experiments without using automatic mixed precision, we found that at the beginning of the training, the loss would become 'nan', and not converge. When we used automatic mixed precision, the loss converged to a smaller value and the model achieved high accuracy in prediction. The results presented in our paper were obtained using automatic mixed precision. Therefore, the experimental results presented in the table may not be consistent with those reported in the original [GSA-Nets](https://openreview.net/forum?id=JkfYjnOEo6M) paper. The experimental logs can be found in the folder "CIFAR10_EXP_LOG".

### Contributions of each author to the paper

[Kaifan Yang](https://github.com/ZJUCDSYangKaifan) & [Ke Liu](https://github.com/zjuKeLiu) led the project and made significant contributions, including proposing the ideas, designing the model architecture, and conducting the experiments.
### Acknowledgements
*We gratefully acknowledge the authors of GSA-Nets paper David W. Romero and Jean-Baptiste Cordonnier.  They patiently answered and elaborated on the experimental details of the paper [GSA-Nets](https://openreview.net/forum?id=JkfYjnOEo6M).*
