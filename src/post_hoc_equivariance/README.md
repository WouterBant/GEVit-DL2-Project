## Reproducing results for the post-hoc experiments

> Download patch Camelyon here:
https://drive.google.com/file/d/1THSEUCO3zg74NKf_eb3ysKiiq2182iMH/view?usp=sharing

---

- All our results are stored in the [results](results) folder
- We provide checkpoints for various pretrained models in the [checkpoints](checkpoints) folder

### Training base model
To train on Rotation Mnist or Patch Camelyon simply call:
```bash
python train_vit.py [--rotmnist]
```

- --rotmnist: if you want to train on the rotation MNIST dataset else you train on patch Camelyon

To train a model on normal MNIST:
```bash
python mnist.py
```

---

### Experiments for pretrained resnet
To run the experiments for the pretrained resnet, simply run (note that you need to install the patch Camelyon dataset yourself):
```bash
python resnet.py [--finetune]
```

- --finetune: will finetune the last layer (experiments showed that finetuning the entire model is not beneficial)

---

### Other experiments general
All other experiments can be reproduced with:
```bash
python post_hoc_experiments.py --model_path path_to_model [--n_rotations] [--flips] [--finetune_model] [--finetune_mlp_head] [--pcam]
```

- --n_rotations: the number of rotations to take for the test time augmentations of the input. For rotation MNIST experiments we found that 16 works best and 4 for patch Camelyon

- --flips: if passed in the command the different rotations will be flipped too, note that this is not desired for rotation MNIST, but it is used for patch Camelyon.

- --finetune_model: passing this argument will allow you to finetune the entire 

- --finetune_mlp_head: passing this argument will allow you to finetune the final layer

---

### Other experiments commands
- less_data_rotmnist: Here we train on 10% of the training set of rotation mnist and evaluate on the entire test set
```bash
python post_hoc_experiments.py --model_path checkpoints/less_data_rotmnist/normal_vit.pt --n_rotations 16 --less_data [--finetune_model] [--finetune_mlp_head] 
```
- rotmnist: Here we use store the results we got for a model trained on the entire training set of rotation MNIST
```bash
python post_hoc_experiments.py --model_path checkpoints/rotmnist/normal_vit.pt --n_rotations 16 [--finetune_model] [--finetune_mlp_head]
```
- train_mnist: Here we trained on normal MNIST and evaluate on rotation MNIST
```bash
python post_hoc_experiments.py --model_path checkpoints/train_mnist/normal_vit.pt --n_rotations 16 [--finetune_model] [--finetune_mlp_head]
```