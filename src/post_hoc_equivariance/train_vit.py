import sys

sys.path.append('..')
import argparse
import copy

import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
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

def main(args):

    if args.rotmnist:
        data_mean = (0.1307,)
        data_stddev = (0.3081,)
        transform_train = transforms.Compose([
            transforms.RandomRotation(degrees=(-180, 180)),  # Random rotation
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_stddev)
        ])
    else:
        data_mean = (0.701, 0.538, 0.692)
        data_stddev = (0.235, 0.277, 0.213)
        transform_train = transforms.Compose([
            CustomRotation([0, 90, 180, 270]),
            transforms.RandomHorizontalFlip(),  # Random horizontal flip with a probability of 0.5
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_stddev)
        ])
    
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_stddev),
        ]
    )

    if args.rotmnist:
        train_set = MNIST_rot(root="../data", stage="train", download=True, transform=transform_train, data_fraction=0.1, only_3_and_8=False)
        validation_set = MNIST_rot(root="../data", stage="validation", download=True, transform=transform_test, data_fraction=1, only_3_and_8=False)
        test_set = MNIST_rot(root="../data", stage="test", download=True, transform=transform_test, data_fraction=1, only_3_and_8=False)
    else:
        train_set = PCam(root="../data", train=True, download=True, transform=transform_train)
        validation_set = PCam(root="../data", train=False, valid=True, download=True, transform=transform_test)
        test_set = PCam(root="../data", train=False, download=True, transform=transform_test)

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
    if args.rotmnist:
        model = VisionTransformer(embed_dim=64,
                                hidden_dim=512,
                                num_heads=4,
                                num_layers=6,
                                patch_size=4,
                                num_channels=1,
                                num_patches=49,
                                num_classes=10,
                                dropout=0.1).to(device)
    else:
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

    if args.rotmnist:
        optimizer = torch.optim.Adam(model.parameters(), 0.001)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        max_steps = 50
        max_steps *= len(train_loader.dataset) // 1024
        lr_scheduler = linear_warmup_cosine_lr_scheduler(
                optimizer, 10.0 / 50, T_max=50  # Perform linear warmup for 10 epochs.
            )

    best_model = copy.deepcopy(model.state_dict())
    best_val_acc = 0

    n_epochs = 500 if args.rotmnist else 50  # pcam large
    for epoch in tqdm(range(n_epochs)):
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
            if not args.rotmnist: lr_scheduler.step()

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
    
    print(test_acc)

    # save model and log it
    if args.rotmnist:
        torch.save(model.state_dict(), "saved/model_rotmnist.pt")
    else:
        torch.save(model.state_dict(), "saved/model_pcam.pt")

def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser for Model Configuration")

    parser.add_argument("--rotmnist", action="store_true", help="Training for rotmnist")
    parser.add_argument("--only_3_and_8", action="store_true", help="Only use classes 3 and 8")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
