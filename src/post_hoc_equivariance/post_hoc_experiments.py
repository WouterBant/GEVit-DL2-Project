import sys
sys.path.append('..')
from datasets import MNIST_rot, PCam
from train_vit import VisionTransformer
from post_hoc_equivariant import *
from sub_models import ScoringModel, Transformer

import torch
import torch.nn as nn
import torchvision.transforms as tvtf

import os 
import csv
import copy
import random
import argparse
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomRotation(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return tvtf.functional.rotate(img, angle)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using CUDA
    torch.backends.cudnn.deterministic = True  # if using CUDA
    torch.backends.cudnn.benchmark = False  # if using CUDA, may improve performance but can lead to non-reproducible results

def get_non_equivariant_vit(model_path, device):
    if not args.pcam:
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
                                num_classes=10 if args.only_3_and_8 else 2,
                                dropout=0.1).to(device)
    print(model.load_state_dict(torch.load(model_path, map_location=device), strict=False))
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser for Model Configuration")

    # required arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")

    # optional arguments
    parser.add_argument("--n_rotations", type=int, default=4, help="Number of rotations")
    parser.add_argument("--flips", action="store_true", help="Enable flips")
    parser.add_argument("--finetune_model", action="store_true", help="Enable fine-tuning of the entire model")
    parser.add_argument("--finetune_mlp_head", action="store_true", help="Enable fine-tuning of the linear layer")
    parser.add_argument("--pcam", action="store_true", help="Experiment for pcam")
    parser.add_argument("--only_3_and_8", action="store_true", help="Only use classes 3 and 8")
    parser.add_argument("--less_data", action="store_true", help="Use less data for training")

    args = parser.parse_args()
    return args

def train(model, n_epochs, train_loader, val_loader):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
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

def evaluate(model, val_loader):
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

def test(model, test_loader):
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

def run_experiments(args):
    if not args.pcam:
        data_mean = (0.1307,)
        data_stddev = (0.3081,)
    else:
        data_mean = (0.701, 0.538, 0.692)
        data_stddev = (0.235, 0.277, 0.213)

    if not args.pcam:
        transform_train = tvtf.Compose([
            tvtf.RandomRotation(degrees=(-180, 180)),  # random rotation
            tvtf.ToTensor(),
            tvtf.Normalize(data_mean, data_stddev)
        ])
    else:
        transform_train = tvtf.Compose([
                CustomRotation([0, 90, 180, 270]),
                # transforms.RandomRotation(degrees=(-180, 180)),  # Random rotation
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

    if not args.pcam:
        train_set = MNIST_rot(root="../data", stage="train", download=True, transform=transform_train, data_fraction=0.1 if args.less_data else 1, only_3_and_8=args.only_3_and_8)
        validation_set = MNIST_rot(root="../data", stage="validation", download=True, transform=transform_test, data_fraction=1, only_3_and_8=args.only_3_and_8)
        test_set = MNIST_rot(root="../data", stage="test", download=True, transform=transform_test, data_fraction=1, only_3_and_8=args.only_3_and_8)
    else:
        train_set = PCam(root="../data", train=True, download=True, transform=transform_train)
        validation_set = PCam(root="../data", train=False, valid=True, download=True, transform=transform_test)
        test_set = PCam(root="../data", train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=128,
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

    accuracies = list()

    finetune = args.finetune_mlp_head or args.finetune_model

    # baseline model
    set_seed(42)
    model = get_non_equivariant_vit(args.model_path, device)
    val_acc = evaluate(model, val_loader)
    test_acc = test(model, test_loader)
    accuracies.append((type(model).__name__, val_acc, test_acc))
    print(accuracies)

    # mean pooling
    set_seed(42)
    model = get_non_equivariant_vit(args.model_path, device)
    eq_model_mean = PostHocEquivariantMean(model, n_rotations=args.n_rotations, flips=args.flips, finetune_mlp_head=args.finetune_mlp_head, finetune_model=args.finetune_model)
    if finetune:
        eq_model_mean = train(eq_model_mean, 25, train_loader, val_loader)
    val_acc = evaluate(eq_model_mean, val_loader)
    test_acc = test(eq_model_mean, test_loader)
    accuracies.append((type(eq_model_mean).__name__, val_acc, test_acc))
    print(accuracies)

    # max pooling
    set_seed(42)
    model = get_non_equivariant_vit(args.model_path, device)
    eq_model_max = PostHocEquivariantMax(model, n_rotations=args.n_rotations, flips=args.flips, finetune_mlp_head=args.finetune_mlp_head, finetune_model=args.finetune_model)
    if finetune:
        eq_model_max = train(eq_model_max, 25, train_loader, val_loader)
    val_acc = evaluate(eq_model_max, val_loader)
    test_acc = test(eq_model_max, test_loader)
    accuracies.append((type(eq_model_max).__name__, val_acc, test_acc))
    print(accuracies)

    # summing latent dimensions
    set_seed(42)
    model = get_non_equivariant_vit(args.model_path, device)
    eq_model_sum = PostHocEquivariantSum(model, n_rotations=args.n_rotations, flips=args.flips, finetune_mlp_head=args.finetune_mlp_head, finetune_model=args.finetune_model)
    if finetune:
        eq_model_sum = train(eq_model_sum, 25, train_loader, val_loader)
    val_acc = evaluate(eq_model_sum, val_loader)
    test_acc = test(eq_model_sum, test_loader)
    accuracies.append((type(eq_model_sum).__name__, val_acc, test_acc))
    print(accuracies)

    # product of class probabilities
    set_seed(42)
    model = get_non_equivariant_vit(args.model_path, device)
    eq_model_most_probable = PostHocEquivariantMostProbable(model, n_rotations=args.n_rotations, flips=args.flips, finetune_mlp_head=args.finetune_mlp_head, finetune_model=args.finetune_model)
    if finetune:
        eq_model_most_probable = train(eq_model_most_probable, 25, train_loader, val_loader)
    val_acc = evaluate(eq_model_most_probable, val_loader)
    test_acc = test(eq_model_most_probable, test_loader)
    accuracies.append((type(eq_model_most_probable).__name__, val_acc, test_acc))
    print(accuracies)

    # take transformation with highest certainty for class
    set_seed(42)
    model = get_non_equivariant_vit(args.model_path, device)
    eq_model_most_certain = PostHocMostCertain(model, n_rotations=args.n_rotations, flips=args.flips, finetune_mlp_head=args.finetune_mlp_head, finetune_model=args.finetune_model)
    if finetune:
        eq_model_most_certain = train(eq_model_most_certain, 25, train_loader, val_loader)
    val_acc = evaluate(eq_model_most_certain, val_loader)
    test_acc = test(eq_model_most_certain, test_loader)
    accuracies.append((type(eq_model_most_certain).__name__, val_acc, test_acc))
    print(accuracies)

    # learn to score latent dimensions
    set_seed(42)
    model = get_non_equivariant_vit(args.model_path, device)
    scoring_model = ScoringModel()
    eq_model_learned_score_aggregation = PostHocLearnedScoreAggregation(model=model, scoring_model=scoring_model, finetune_mlp_head=args.finetune_mlp_head, finetune_model=args.finetune_model)
    eq_model_learned_score_aggregation = train(eq_model_learned_score_aggregation, 25, train_loader, val_loader)
    val_acc = evaluate(eq_model_learned_score_aggregation, val_loader)
    test_acc = test(eq_model_learned_score_aggregation, test_loader)
    accuracies.append((type(eq_model_learned_score_aggregation).__name__, val_acc, test_acc))
    print(accuracies)

    # learn to (invariantly) aggregate the latent dimensions
    set_seed(42)
    model = get_non_equivariant_vit(args.model_path, device)
    aggregation_model = Transformer(embed_dim=64, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1)
    eq_model_learned_aggregation = PostHocLearnedAggregation(model=model, aggregation_model=aggregation_model, finetune_mlp_head=args.finetune_mlp_head, finetune_model=args.finetune_model)
    eq_model_learned_aggregation = train(eq_model_learned_aggregation, 60, train_loader, val_loader)
    val_acc = evaluate(eq_model_learned_aggregation, val_loader)
    test_acc = test(eq_model_learned_aggregation, test_loader)
    accuracies.append((type(eq_model_learned_aggregation).__name__, val_acc, test_acc))
    return accuracies

def generate_filename(args):
    filename = f"model_{os.path.basename(args.model_path)}_rot_{args.n_rotations}_flips_{args.flips}_finetune_model_{args.finetune_model}_finetune_mlp_{args.finetune_mlp_head}.csv"
    return filename

def write_to_csv(filename, val_accs):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Validation Accuracy', 'Test Accuracy'])
        for model, val_acc, test_acc in val_accs:
            writer.writerow([model, val_acc, test_acc])

def main(args):
    accuracies = run_experiments(args)
    filename = generate_filename(args)
    write_to_csv(filename, accuracies)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
