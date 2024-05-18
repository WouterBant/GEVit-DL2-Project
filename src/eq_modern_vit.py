import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvtf
import torchvision.transforms.functional as TF
import wandb
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from datasets import MNIST_rot
from train_vit import VisionTransformer
import os
import models
import g_selfatt.groups as groups
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_mean = (0.1307,)
data_stddev = (0.3081,)

transform_train = tvtf.Compose([
    tvtf.RandomRotation(degrees=(-180, 180)),  # random rotation
    tvtf.ToTensor(),
    tvtf.Normalize(data_mean, data_stddev)
])
transform_test = tvtf.Compose(
    [
        tvtf.ToTensor(),
        tvtf.Normalize(data_mean, data_stddev),
    ]
)

train_set = MNIST_rot(root="../data", stage="train", download=True, transform=transform_train, data_fraction=1, only_3_and_8=False)
validation_set = MNIST_rot(root="../data", stage="validation", download=True, transform=transform_test, data_fraction=1, only_3_and_8=False)
test_set = MNIST_rot(root="../data", stage="test", download=True, transform=transform_test, data_fraction=1, only_3_and_8=False)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=16,
    shuffle=True,
    num_workers=4,
)
val_loader = torch.utils.data.DataLoader(
    validation_set,
    batch_size=128,
    shuffle=True,
    num_workers=4,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=128,
    shuffle=False,
    num_workers=4,
)
img_loader = torch.utils.data.DataLoader(  # single element for visualization purposes
    test_set,
    batch_size=1,
    shuffle=False,
    num_workers=4,
)

def get_transforms(images, n_rotations=4, flips=True):
    """ Returns all transformations of the input images """

    B, C, H, W = images.shape
    T = 2*n_rotations if flips else n_rotations  # number of transformations

    # initialize empty transforms tensor
    transforms = torch.empty(size=(B, T, C, H, W))
    transforms[:, 0,...] = images
    idx = 1

    # remember all orientations that need to be flipped
    orientations = [images] if flips else []

    # rotations
    for i in range(1, n_rotations):
        angle = i * (360 / n_rotations)
        rotated_images = TF.rotate(images, angle, interpolation=InterpolationMode.BILINEAR)  # B, C, H, W
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


class EquivariantViT(nn.Module):
    def __init__(self, patch_size=7, num_patches=16, num_channels=1, n_rotations=4, flips=True, n_embd=1):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels  
        self.n_rotations = n_rotations
        self.flips = flips
        self.n_embd = n_embd
        self.num_patches_x = int(math.sqrt(num_patches))
        # below can be more intricate, but for now we just use a linear layer
        self.project = nn.Linear(num_channels*patch_size**2, n_embd)  # to project the patches to their embedding space
        self.gevit = models.GroupTransformer(
            group=groups.SE2(num_elements=8),
            in_channels=1,
            num_channels=20,
            block_sizes=[2, 3],
            expansion_per_block=1,
            crop_per_layer=[0, 0, 0, 0, 0],
            image_size=self.num_patches_x,
            num_classes=10,
            dropout_rate_after_maxpooling=0.0,
            maxpool_after_last_block=False,
            normalize_between_layers=False,
            patch_size=None,
            num_heads=9,
            norm_type="LayerNorm",
            activation_function="Swish",
            attention_dropout_rate=0.0,
            value_dropout_rate=0.01,
            whitening_scale=1.41421356,
        )

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
        x = x.view(B, num_patches, T, self.n_embd)  # TODO check this
        x = x.mean(dim=2)
        
        # reshape to image grid
        x = x.view(B, self.num_patches_x, self.num_patches_x, self.n_embd).permute(0, 3, 1, 2)

        # print(x.shape)
        # pass through the GEViT to get predictions
        x = self.gevit(x)

        return x


def main():
    os.environ["WANDB_API_KEY"] = "691777d26bb25439a75be52632da71d865d3a671"  # TODO change this if we are doing serious runs
    wandb.init(
        project="non-equivariant-vit",
        entity="equivatt_team",
    )

    model = EquivariantViT()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.001)

    best_model = copy.deepcopy(model.state_dict())
    best_val_acc = 0

    for epoch in range(500):
        model.train()
        losses = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # Update weights
            losses.append(loss.item())
        wandb.log({"loss_train":sum(losses)/len(losses)}, step=epoch+1)

        # Validate on the validation set
        if epoch % 10 == 0:
            model.eval()  # Set the model to evaluation mode
            correct = 0
            total = 0
            with torch.no_grad():  # Disable gradient calculation during inference
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            if accuracy > best_val_acc:
                best_model = copy.deepcopy(model.state_dict())
                best_val_acc = accuracy
            
            wandb.log({"validation_accuracy":accuracy}, step=epoch+1)

    wandb.run.summary["best_validation_accuracy"] = best_val_acc

    # Test on the test set
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation during inference
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total
    
    wandb.run.summary["test_acc"] = test_acc

    # save model and log it
    model.load_state_dict(best_model)
    torch.save(model.state_dict(), "saved/modern_eq_vit.pt")
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "modern_eq_vit.pt"))


if __name__ == "__main__":
    main()