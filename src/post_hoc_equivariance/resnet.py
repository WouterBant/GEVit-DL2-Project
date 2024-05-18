import torch
import torchvision.transforms as tvtf
import sys
sys.path.append('..')
from datasets import PCam
import timm
import torch.nn as nn
from tqdm import tqdm
from post_hoc_equivariant import *
import csv
import copy
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_mean = [0., 0., 0.]  # Your mean values
data_stddev = [1., 1., 1.]  # Your standard deviation values

class CustomRotation(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return tvtf.functional.rotate(img, angle)

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

train_set = PCam(root="../data", train=True, valid=False, download=True, transform=transform_test)
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

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using CUDA
    torch.backends.cudnn.deterministic = True  # if using CUDA
    torch.backends.cudnn.benchmark = False  # if using CUDA, may improve performance but can lead to non-reproducible results

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

def evaluate(model):
    model = model.to(device)
    model.eval()
    correct = total = 0
    with torch.no_grad():  # disable gradient calculation during inference
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # move inputs and labels to device
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total
    return val_acc

def test(model):
    model = model.to(device)
    model.eval()
    correct = total = 0
    with torch.no_grad():  # disable gradient calculation during inference
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # move inputs and labels to device
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total
    return test_acc

def train(model, n_epochs=5):
    model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.00000001, )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = evaluate(model)
    print(type(model).__name__)
    print(f"Starting validaitons accuracy: {best_val_acc}")
    best_model_state = None

    for epoch in tqdm(range(5)):
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
        val_acc = evaluate(model)
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
    accuracies = list()

    model = PretrainedResnet50()
    model.eval()
    val_acc = evaluate(model)
    test_acc = test(model)
    accuracies.append((type(model).__name__, val_acc, test_acc))

    # mean pooling
    set_seed(42)
    model = PretrainedResnet50()
    model = model.to(device)
    eq_model_mean = PostHocEquivariantMean(model, n_rotations=4, flips=True, finetune_mlp_head=finetune)
    if finetune:
        eq_model_mean = train(eq_model_mean, n_epochs=3)
    # val_acc = evaluate(eq_model_mean)
    test_acc = test(eq_model_mean)
    print(test_acc)
    # accuracies.append((type(eq_model_mean).__name__, val_acc, test_acc))
    # print(accuracies)

    # max pooling
    set_seed(42)
    model = PretrainedResnet50()
    model = model.to(device)
    eq_model_max = PostHocEquivariantMax(model, n_rotations=4, flips=True, finetune_mlp_head=finetune)
    if finetune:
        eq_model_max = train(eq_model_max, n_epochs=3)
    val_acc = evaluate(eq_model_max)
    test_acc = test(eq_model_max)
    accuracies.append((type(eq_model_max).__name__, val_acc, test_acc))
    print(accuracies)

    # summing latent dimensions
    set_seed(42)
    model = PretrainedResnet50()
    model = model.to(device)
    eq_model_sum = PostHocEquivariantSum(model, n_rotations=4, flips=True, finetune_mlp_head=finetune)
    if finetune:
        eq_model_sum = train(eq_model_sum, n_epochs=3)
    val_acc = evaluate(eq_model_sum)
    test_acc = test(eq_model_sum)
    accuracies.append((type(eq_model_sum).__name__, val_acc, test_acc))
    print(accuracies)

    # product of class probabilities
    set_seed(42)
    model = PretrainedResnet50()
    model = model.to(device)
    eq_model_most_probable = PostHocEquivariantMostProbable(model, n_rotations=4, flips=True, finetune_mlp_head=finetune)
    if finetune:
        eq_model_most_probable = train(eq_model_most_probable, n_epochs=3)
    val_acc = evaluate(eq_model_most_probable)
    test_acc = test(eq_model_most_probable)
    accuracies.append((type(eq_model_most_probable).__name__, val_acc, test_acc))
    print(accuracies)

    # take transformation with highest certainty for class
    set_seed(42)
    model = PretrainedResnet50()
    model = model.to(device)
    eq_model_most_certain = PostHocMostCertain(model, n_rotations=4, flips=True, finetune_mlp_head=finetune)
    if finetune:
        eq_model_most_certain = train(eq_model_most_certain, n_epochs=3)
    val_acc = evaluate(eq_model_most_certain)
    test_acc = test(eq_model_most_certain)
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
    