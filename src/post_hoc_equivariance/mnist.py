import sys

sys.path.append('..')
import copy
import os
import random
from collections import Counter

import torch
import torch.nn as nn
import wandb
from torchvision import transforms
from torchvision.datasets import MNIST
from utils import CustomRotation, img_to_patch, linear_warmup_cosine_lr_scheduler

from datasets import MNIST_rot, PCam
from g_selfatt.utils import num_params


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """Attention Block.

        Args:
            embed_dim: Dimensionality of input and attention feature vectors
            hidden_dim: Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads: Number of heads to use in the Multi-Head Attention block
            dropout: Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        num_classes,
        patch_size,
        num_patches,
        dropout=0.0,
    ):
        """Vision Transformer.

        Args:
            embed_dim: Dimensionality of the input feature vectors to the Transformer
            hidden_dim: Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels: Number of channels of the input (3 for RGB)
            num_heads: Number of heads to use in the Multi-Head Attention block
            num_layers: Number of layers to use in the Transformer
            num_classes: Number of classes to predict
            patch_size: Number of pixels that the patches have per dimension
            num_patches: Maximum number of patches an image can have
            dropout: Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x, output_cls=False):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        if output_cls:
            return cls

        out = self.mlp_head(cls)
        return out


def main():
    dataset = "mnist" # "rotmnist" "pcam"

    # os.environ["WANDB_API_KEY"] = ""  # TODO insert your wandb key here
    
    # wandb.init(
    #     project="pretraining-mnist-our-vit",
    #     group="pcam",
    #     entity="ge_vit_DL2",
    # )

    if dataset == "rotmnist" or dataset == "mnist":
        data_mean = (0.1307,)
        data_stddev = (0.3081,)
    elif dataset == "pcam":
        data_mean = (0.701, 0.538, 0.692)
        data_stddev = (0.235, 0.277, 0.213)
    else:
        raise ValueError("Invalid dataset") 
    
    if dataset == "pcam":
        # random 90 degree rotation and flips
        transform_train = transforms.Compose([
            CustomRotation([0, 90, 180, 270]),
            transforms.RandomHorizontalFlip(),  # Random horizontal flip with a probability of 0.5
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_stddev)
        ])
    elif dataset == "rotmnist":
        # random rotation
        transform_train = transforms.Compose([
            transforms.RandomRotation(degrees=(-180, 180)),  # Random rotation
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_stddev)
        ])
    elif dataset == "mnist":
        # no data augmentation
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_stddev)
        ])

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_stddev),
        ]
    )

    if dataset == "rotmnist":
        train_set = MNIST_rot(root="../data", stage="train", download=True, transform=transform_train, data_fraction=1, only_3_and_8=False)
        validation_set = MNIST_rot(root="../data", stage="validation", download=True, transform=transform_test, data_fraction=1, only_3_and_8=False)
        test_set = MNIST_rot(root="../data", stage="test", download=True, transform=transform_test, data_fraction=1, only_3_and_8=False)
    elif dataset == "pcam":
        train_set = PCam(root="../data", train=True, download=True, transform=transform_train)
        validation_set = PCam(root="../data", train=False, valid=True, download=True, transform=transform_test)
        test_set = PCam(root="../data", train=False, download=True, transform=transform_test)
    elif dataset == "mnist":
        train_set = MNIST(root="../data/mnistreal", train=True, download=True, transform=transform_train)
        test_set = MNIST(root="../data/mnistreal", train=False, download=True, transform=transform_test)
        # Define the size of the validation set
        validation_size = int(0.2 * len(test_set))  # Adjust as needed

        # Define indices for the validation set and the remaining for testing
        torch.manual_seed(42)
        indices = torch.randperm(len(test_set)).tolist()
        validation_indices = indices[:validation_size]
        test_indices = indices[validation_size:]

        # Create subsets for validation and testing
        validation_set = torch.utils.data.Subset(test_set, validation_indices)
        test_set = torch.utils.data.Subset(test_set, test_indices)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=1024,
        shuffle=True,
        num_workers=4,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dataset == "rotmnist" or dataset == "mnist":
        model = VisionTransformer(embed_dim=64,
                                hidden_dim=512,
                                num_heads=4,
                                num_layers=6,
                                patch_size=4,
                                num_channels=1,
                                num_patches=49,
                                num_classes=10,
                                dropout=0.1).to(device)
    elif dataset == "pcam":
        model = VisionTransformer(embed_dim=64,
                            hidden_dim=512,
                            num_heads=4,
                            num_layers=6,
                            patch_size=6,
                            num_channels=3,
                            num_patches=256,
                            num_classes=2,
                            dropout=0.1).to(device)
                            
    print(f"Number of parameters in the model: {num_params(model)}")
    criterion = torch.nn.CrossEntropyLoss()

    if dataset == "pcam":
        lr = 0.01
        # wandb.log({"lr":lr})
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        # scheduler = StepLR(optimizer, step_size=3, gamma=0.5)  # only used on pcam
        max_steps = 50
        max_steps *= len(train_loader.dataset) // 1024
        lr_scheduler = linear_warmup_cosine_lr_scheduler(
                optimizer, 10.0 / 50, T_max=50  # Perform linear warmup for 10 epochs.
            )
    elif dataset == "rotmnist" or dataset == "mnist":
        lr = 0.001
        # wandb.log({"lr":lr})
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        lr_scheduler = None

    best_model = copy.deepcopy(model.state_dict())
    best_val_acc = 0

    n_epochs = 500 if (dataset == "rotmnist" or dataset == "mnist") else 50
    for epoch in range(n_epochs):
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
            if lr_scheduler != None:
                lr_scheduler.step()

        # wandb.log({"loss_train":sum(losses)/len(losses)}, step=epoch+1)

        # Validate on the validation set
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
        # wandb.log({"validation_accuracy":accuracy}, step=epoch+1)
        # scheduler.step()

    # wandb.run.summary["best_validation_accuracy"] = best_val_acc

    model.load_state_dict(best_model)
    
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
    
    # wandb.run.summary["test_acc"] = test_acc

    # save model and log it
    torch.save(model.state_dict(), f"saved/{dataset}_{random.randint(0, 10000)}.pt")
    # torch.save(model.state_dict(), os.path.join(wandb.run.dir, f"{dataset}_{random.randint(0,10000)}.pt"))


if __name__ == "__main__":
    main()