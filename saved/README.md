Details for the files that don't have stats in the name:

- mnist_2555.pt: here we trained a normal vit on mnist without data augmentation. Just adam with lr 0.001 no scheduler. VisionTransformer(embed_dim=64, hidden_dim=512, num_heads=4, num_layers=6, patch_size=4, num_channels=1, num_patches=49, num_classes=10, dropout=0.1) https://wandb.ai/ge_vit_DL2/pretraining-mnist-our-vit?nw=nwuserwouterbant 

- model.pt: no idea think some old rotmnist non equivariant vit. Best guess is the below but also with flips data augmentation which was dumb

- model2.pt: our vit trained on rotmnist. Just adam with lr 0.001 no scheduler. Data augmentation is random rotation (no flips). with VisionTransformer(embed_dim=64, hidden_dim=512, num_heads=4, num_layers=6, patch_size=4, num_channels=1, num_patches=49, num_classes=10, dropout=0.1). https://wandb.ai/ge_vit_DL2/our_non_equivariant_se_vit?nw=nwuserwouterbant

- modelpcamgood.pt: this is our vit. Doesnt perform well on test set. Data augmentation is random k*90 degree rotation and random flips. https://wandb.ai/ge_vit_DL2/non-equivariant-vit-pcam/runs/nzc3ib95?nw=nwuserwouterbant

