import sys

sys.path.append('..')
import copy
import csv

import timm
import torch
import torch.nn as nn
import torchvision.transforms as tvtf
from post_hoc_equivariant import *
from tqdm import tqdm
from utils import CustomRotation, evaluate, set_seed, test

from datasets import PCam


class PretrainedResnet50(nn.Module):

    def __init__(self):
        super().__init__()
        # https://huggingface.co/1aurent/resnet50.tiatoolbox-pcam/
        model = timm.create_model(model_name="hf-hub:1aurent/densenet161.tiatoolbox-pcam", pretrained=True)
        self.layers_before_last_linear = nn.Sequential(*list(model.children())[:-1])
        self.mlp_head = nn.Sequential(list(model.children())[-1])

    def forward(self, x, output_cls=False):
        cls = self.layers_before_last_linear(x)
        if output_cls:
            return cls
        logits = self.mlp_head(cls)
        return logits

def train(model, train_loader, val_loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), n_epochs=5):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = evaluate(model, val_loader)
    print(type(model).__name__)
    print(f"Starting validaitons accuracy: {best_val_acc}")
    best_model_state = None

    for epoch in tqdm(range(n_epochs)):
        model.train(True)
        epoch_losses = []
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # validate and store best model state
        val_acc = evaluate(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

        # log epoch loss
        print(f"Epoch {epoch+1}: loss {sum(epoch_losses)/len(epoch_losses):.4f}, validation accuracy {val_acc}")

    # Load best model state into the original model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model.to(device)

def main(finetune):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_mean = [0., 0., 0.]  # Your mean values
    data_stddev = [1., 1., 1.]  # Your standard deviation values


    transform_train = tvtf.Compose([
        tvtf.Resize(96, interpolation=tvtf.InterpolationMode.BICUBIC),  # Resize the image to 96x96 using bicubic interpolation
        tvtf.CenterCrop(96),  # Center crop the image to 96x96
        tvtf.ToTensor(),  # Convert the image to a PyTorch tensor
        CustomRotation([0, 90, 180, 270]),
        tvtf.RandomHorizontalFlip(),  # Random horizontal flip with a probability of 0.5
        tvtf.RandomVerticalFlip(),
        tvtf.Normalize(mean=data_mean, std=data_stddev)  # Normalize the tensor
    ])

    transform_test = tvtf.Compose([
        tvtf.Resize(96, interpolation=tvtf.InterpolationMode.BICUBIC),  # Resize the image to 96x96 using bicubic interpolation
        tvtf.CenterCrop(96),  # Center crop the image to 96x96
        tvtf.ToTensor(),  # Convert the image to a PyTorch tensor
        tvtf.Normalize(mean=data_mean, std=data_stddev)  # Normalize the tensor
    ])

    train_set = PCam(root="../data", train=True, valid=False, download=True, transform=transform_train)
    validation_set = PCam(root="../data", train=False, valid=True, download=True, transform=transform_test)
    test_set = PCam(root="../data", train=False, download=True, transform=transform_test, data_fraction=1.0)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=64,
        shuffle=True,
        num_workers=4,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=128,
        shuffle=False,
        num_workers=4,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=128,
        shuffle=False,
        num_workers=4,
    )
    accuracies = list()

    model = PretrainedResnet50()
    model.eval()
    val_acc = evaluate(model, val_loader, device)
    test_acc = test(model, test_loader, device)
    accuracies.append((type(model).__name__, val_acc, test_acc))

    # mean pooling
    set_seed(42)
    model = PretrainedResnet50()
    model = model.to(device)
    eq_model_mean = PostHocEquivariantMean(model, n_rotations=4, flips=True, finetune_mlp_head=finetune)
    if finetune:
        eq_model_mean = train(eq_model_mean, train_loader, val_loader, device, n_epochs=3)
    val_acc = evaluate(eq_model_mean, val_loader, device)
    test_acc = test(eq_model_mean, test_loader, device)
    print(test_acc)
    accuracies.append((type(eq_model_mean).__name__, val_acc, test_acc))
    print(accuracies)

    # max pooling
    set_seed(42)
    model = PretrainedResnet50()
    model = model.to(device)
    eq_model_max = PostHocEquivariantMax(model, n_rotations=4, flips=True, finetune_mlp_head=finetune)
    if finetune:
        eq_model_max = train(eq_model_max, train_loader, val_loader, device, n_epochs=3)
    val_acc = evaluate(eq_model_max, val_loader, device)
    test_acc = test(eq_model_max, test_loader, device)
    accuracies.append((type(eq_model_max).__name__, val_acc, test_acc))
    print(accuracies)

    # summing latent dimensions
    set_seed(42)
    model = PretrainedResnet50()
    model = model.to(device)
    eq_model_sum = PostHocEquivariantSum(model, n_rotations=4, flips=True, finetune_mlp_head=finetune)
    if finetune:
        eq_model_sum = train(eq_model_sum, train_loader, val_loader, device, n_epochs=3)
    val_acc = evaluate(eq_model_sum, val_loader, device)
    test_acc = test(eq_model_sum, test_loader, device)
    accuracies.append((type(eq_model_sum).__name__, val_acc, test_acc))
    print(accuracies)

    # product of class probabilities
    set_seed(42)
    model = PretrainedResnet50()
    model = model.to(device)
    eq_model_most_probable = PostHocEquivariantMostProbable(model, n_rotations=4, flips=True, finetune_mlp_head=finetune)
    if finetune:
        eq_model_most_probable = train(eq_model_most_probable, train_loader, val_loader, device, n_epochs=3)
    val_acc = evaluate(eq_model_most_probable, val_loader, device)
    test_acc = test(eq_model_most_probable, test_loader, device)
    accuracies.append((type(eq_model_most_probable).__name__, val_acc, test_acc))
    print(accuracies)

    # take transformation with highest certainty for class
    set_seed(42)
    model = PretrainedResnet50()
    model = model.to(device)
    eq_model_most_certain = PostHocMostCertain(model, n_rotations=4, flips=True, finetune_mlp_head=finetune)
    if finetune:
        eq_model_most_certain = train(eq_model_most_certain, train_loader, val_loader, device, n_epochs=3)
    val_acc = evaluate(eq_model_most_certain, val_loader, device)
    test_acc = test(eq_model_most_certain, test_loader, device)
    accuracies.append((type(eq_model_most_certain).__name__, val_acc, test_acc))
    print(accuracies)

    with open("resnet_finetune_model.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Validation Accuracy', 'Test Accuracy'])
        for model, val_acc, test_acc in accuracies:
            writer.writerow([model, val_acc, test_acc])

if __name__ == "__main__":
    # check for --finetune flag
    finetune = False if len(sys.argv) == 1 else sys.argv[1] == "--finetune"
    main(finetune)
    