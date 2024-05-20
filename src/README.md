### Reproducing experimental results for GE-ViT

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

##### Command for running PatchCamelyon with less data
```bash
python run_experiment.py --config.dataset PCam --config.model p4sa --config.norm_type LayerNorm --config.attention_type Local --config.activation_function Swish --config.patch_size 5 --config.dropout_att 0.1 --config.dropout_values 0.1 --config.whitening_scale 1.41421356 --config.epochs 100 --config.optimizer SGD --config.lr 0.01 --config.optimizer_momentum 0.9 --config.scheduler linear_warmup_cosine --config.sched_decay_steps='(1000,)' --config.sched_decay_factor 1.0 --config.weight_decay 0.0001 --config.batch_size 16 --config.device cuda --config.seed 0 --config.comment "" --config.data_fraction 0.001
```