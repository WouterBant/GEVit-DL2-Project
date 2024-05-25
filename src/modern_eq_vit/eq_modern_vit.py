import sys
sys.path.append("..")
import models
from g_selfatt import utils
import g_selfatt.groups as groups
from datasets import MNIST_rot, PCam

import torch
import torch.nn as nn
import torchvision.transforms as tvtf
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import StepLR,LambdaLR

import os
import copy
import math
import wandb
import random
import argparse
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomRotation(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return tvtf.functional.rotate(img, angle)


def get_transforms(images, n_rotations=4, flips=True):
    """ Returns all transformations of the input images """

    B, C, H, W = images.shape
    T = 2*n_rotations if flips else n_rotations  # number of transformations

    # initialize empty transforms tensor
    transforms = torch.empty(size=(B, T, C, H, W)) #.to(device)
    transforms[:, 0,...] = images
    idx = 1

    # remember all orientations that need to be flipped
    orientations = [images] if flips else []

    # rotations
    for i in range(1, n_rotations):
        angle = i * (360 / n_rotations)
        rotated_images = TF.rotate(images, angle)  # B, C, H, W
        transforms[:, idx,...] = rotated_images
        idx += 1

        if flips:
            orientations.append(rotated_images)

    # flips
    for transform in orientations:
        flipped_image = TF.hflip(transform)
        transforms[:, idx, ...] = flipped_image
        idx += 1

    return transforms.to(device)  # B, T, C, H, W


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Args:
        x: Tensor representing the image of shape [B, C, H, W]
        patch_size: Number of pixels per dimension of the patches (integer)
        flatten_channels: If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using CUDA
    torch.backends.cudnn.deterministic = True  # if using CUDA
    torch.backends.cudnn.benchmark = False  # if using CUDA, may improve performance but can lead to non-reproducible results

class EquivariantViT(nn.Module):
    def __init__(self, patch_size=7, num_patches=16, num_channels=1, n_rotations=4, flips=False, n_embd=64, att_patch_size=None, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels  
        self.n_rotations = n_rotations
        self.flips = flips
        self.n_embd = n_embd
        self.num_patches_x = int(math.sqrt(num_patches))
        # below can be more intricate, but for now we just use a linear layer
        self.project = nn.Linear(num_channels*patch_size**2, n_embd).to(device)  # to project the patches to their embedding space
        self.gevit =  models.GroupTransformer(
            group=groups.SE2(num_elements=4),
            in_channels=n_embd,
            num_channels=20,
            block_sizes=[2, 3],
            expansion_per_block=1,
            crop_per_layer=[1, 0, 0, 0, 0],
            image_size=self.num_patches_x,
            num_classes=num_classes,
            dropout_rate_after_maxpooling=0.0,
            maxpool_after_last_block=False,
            normalize_between_layers=True,
            patch_size=att_patch_size,
            num_heads=9,
            norm_type="LayerNorm",
            activation_function="Swish",
            attention_dropout_rate=0.1,
            value_dropout_rate=0.1,
            whitening_scale=1.41421356,
        ).to(device)
        self.layernorm = torch.nn.LayerNorm(n_embd)

    def forward(self, x):
        # get the patches
        x = img_to_patch(x, self.patch_size, flatten_channels=False)  # B, num_patches, C, patch_size, patch_size
        B, num_patches, C, patch_size, _ = x.shape

        # get all transformations for the patches
        x = x.view(B*num_patches, C, patch_size, patch_size)
        x = get_transforms(x, n_rotations=self.n_rotations, flips=self.flips)

        T = x.shape[1]  # number of transformations

        # flatten and project all patches
        x = x.view(B*num_patches*T, C*patch_size*patch_size)

        x = self.project(x)

        # combine the transformations for the patches to make it invariant
        x = x.view(B, num_patches, T, self.n_embd)

        x = x.mean(dim=2)
        
        x = self.layernorm(x)
        
        # reshape to image grid
        x = x.view(B, self.num_patches_x, self.num_patches_x, self.n_embd).permute(0, 3, 1, 2)

        # pass through the GEViT to get predictions
        x = self.gevit(x)
        return x

def main(args):
    os.environ["WANDB_API_KEY"] = "26de9d19e20ea7e7f7352e5b36f139df8d145bc8"
    set_seed(42)

    data_mean = (0.701, 0.538, 0.692)
    data_stddev = (0.235, 0.277, 0.213)
    transform_train = tvtf.Compose([
        CustomRotation([0, 90, 180, 270]),
        tvtf.RandomHorizontalFlip(),  # Random horizontal flip with a probability of 0.5
        tvtf.RandomVerticalFlip(),
        tvtf.ToTensor(),
        tvtf.Normalize(data_mean, data_stddev)
    ])
    
    transform_test = tvtf.Compose(
        [
            tvtf.ToTensor(),
            tvtf.Normalize(data_mean, data_stddev),
        ]
    )

    train_set = PCam(root="../data", train=True, download=True, transform=transform_train, data_fraction=0.01)
    validation_set = PCam(root="../data", train=False, valid=True, download=True, transform=transform_test, data_fraction=0.1)
    test_set = PCam(root="../data", train=False, download=True, transform=transform_test)

    batch_size = 64 if (args.modern_vit) else 16
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )

    wandb.init(
        project="wouters_eq_vit",
        group="rotmnist",
        entity="ge_vit_DL2",
    )

    if args.modern_vit:
        patch_size = args.patch_size
        num_patches = (96 // patch_size)**2
        wandb.log({"patch_size": patch_size})
        model = EquivariantViT(
                patch_size=patch_size, 
                num_patches=num_patches,
                num_classes=2, 
                num_channels=3, 
                n_rotations=4, 
                flips=True, 
                n_embd=128, 
                att_patch_size=3).to(device)
    elif args.modern_vit_w_cnn:
        gcnn = models.get_gcnn(order=4,
            in_channels=3,
            out_channels=32,
            kernel_size=5,
            num_hidden=17,
            hidden_channels=32)
        group_transformer = models.GroupTransformer(
                group=groups.SE2(num_elements=4),
                in_channels=gcnn.out_channels,
                num_channels=20,
                block_sizes=[2, 3],
                expansion_per_block=1,
                crop_per_layer=[2, 0, 2, 1, 1],
                image_size=gcnn.output_dimensionality,
                num_classes=2,
                dropout_rate_after_maxpooling=0.0,
                maxpool_after_last_block=True,
                normalize_between_layers=True,
                patch_size=5,
                num_heads=9,
                norm_type="LayerNorm",
                activation_function="Swish",
                attention_dropout_rate=0.1,
                value_dropout_rate=0.1,
                whitening_scale=1.41421356,
            )
        model = models.Hybrid(gcnn, group_transformer).to(device)
    else:
        model = models.GroupTransformer(
            group=groups.SE2(num_elements=4),
            in_channels=3,
            num_channels=12,
            block_sizes=[0, 1, 2, 1],
            expansion_per_block=[1, 2, 2, 2],
            crop_per_layer=[0, 2, 1, 1],
            image_size=96,
            num_classes=2,
            dropout_rate_after_maxpooling=0.0,
            maxpool_after_last_block=True,
            normalize_between_layers=True,
            patch_size=5,
            num_heads=9,
            norm_type="LayerNorm",
            activation_function="Swish",
            attention_dropout_rate=0.1,
            value_dropout_rate=0.01,
            whitening_scale=1.41421356,
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), 0.001)  # 0.001 works well here for floris model
    max_steps = epochs = 50
    max_steps *= len(train_loader.dataset) // batch_size
    lr_scheduler = utils.schedulers.linear_warmup_cosine_lr_scheduler(
        optimizer, 10.0 / epochs, T_max=max_steps  # Perform linear warmup for 10 epochs.
    )
    scaler = GradScaler()
    criterion = torch.nn.CrossEntropyLoss()
    best_model = copy.deepcopy(model.state_dict())
    best_val_acc = 0

    # epsilon = torch.tensor(0.1)
    # smoothing = [epsilon, torch.tensor(1.0) - epsilon]

    for epoch in tqdm(range(epochs)):
        
        model.train()
        losses = []
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device
            # smoothed_labels = labels * smoothing[1] + (1 - labels) * smoothing[0]

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                with autocast():  # Sets autocast in the main thread. It handles mixed precision in the forward pass.
                    outputs = model(inputs)
                    # print(model.gcnn)
                    # print(outputs.shape)
                    print(outputs[0:5,:])
                    loss = criterion(outputs, labels)

                if loss.item() != loss.item():
                    continue
                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                scaler.step(optimizer)
                # Updates the scale for next iteration.
                scaler.update()
                print(loss.item())
                # print(model.project.weight.grad)
                lr_scheduler.step()

            losses.append(loss.item())

        wandb.log({"loss_train":sum(losses)/len(losses)}, step=epoch+1)

        # Validate on the validation set
        # Initialize counters for TP, TN, FP, FN
        positive = negative = total = correct = TP = TN = FP = FN = 0

        with torch.no_grad():  # Disable gradient calculation during inference
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                # Update total and correct counters
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                negative += (labels == 0).sum().item()
                positive += (labels == 1).sum().item()

                # Update TP, TN, FP, FN counters
                TP += ((predicted == 1) & (labels == 1)).sum().item()
                TN += ((predicted == 0) & (labels == 0)).sum().item()
                FP += ((predicted == 1) & (labels == 0)).sum().item()
                FN += ((predicted == 0) & (labels == 1)).sum().item()

        # Calculate accuracy
        accuracy = 100 * correct / total
        if accuracy > best_val_acc:
            best_model = copy.deepcopy(model.state_dict())
            best_val_acc = accuracy

        TP_percentage = 100 * TP / positive
        TN_percentage = 100 * TN / negative
        FP_percentage = 100 * FP / positive
        FN_percentage = 100 * FN / negative

        # Calculate precision and recall
        precision = 100 * TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = 100 * TP / (TP + FN) if (TP + FN) > 0 else 0

        # Log the metrics to wandb
        wandb.log({
            "validation_accuracy": accuracy,
            "TP_percentage": TP_percentage,
            "TN_percentage": TN_percentage,
            "FP_percentage": FP_percentage,
            "FN_percentage": FN_percentage,
            "precision": precision,
            "recall": recall
        }, step=epoch + 1)

        # Log the accuracy to wandb
        wandb.log({"validation_accuracy": accuracy}, step=epoch + 1)

        # Optionally log TP, TN, FP, FN to wandb

    wandb.run.summary["best_validation_accuracy"] = best_val_acc

    # Test on the test set
    model.eval()  # Set the model to evaluation mode
    correct = 0
    # total = 0
    # with torch.no_grad():  # Disable gradient calculation during inference
    #     for inputs, labels in test_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device
    #         outputs = model(inputs)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    # test_acc = 100 * correct / total
    
    # wandb.run.summary["test_acc"] = test_acc

    # save model and log it
    model.load_state_dict(best_model)
    torch.save(model.state_dict(), "saved/modern_eq_vit.pt")
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "modern_eq_vit.pt"))


def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser for Model Configuration")
    parser.add_argument("--modern_vit", action="store_true", help="Training for modern vit")
    parser.add_argument("--modern_vit_w_cnn", action="store_true", help="Training for modern vit with cnn")
    parser.add_argument("--patch_size", default=6, type=int, help="Patch size")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)