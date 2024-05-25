from typing import Dict

import ml_collections
import torch
import torchvision
from collections import Counter
import numpy as np

from datasets import MNIST_rot, PCam


def get_dataset(config: ml_collections.ConfigDict, 
                num_workers: int = 4, 
                data_fraction = 1, 
                data_root: str = "./data",
                ) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create dataloaders for the chosen datasets
    :return: {'train': training_loader, 'validation': validation_loader, 'test': test_loader}
    """
    dataset = {
        "cifar10": torchvision.datasets.CIFAR10,
        "mnist": torchvision.datasets.MNIST,
        "rotmnist": MNIST_rot,
        "pcam": PCam,
    }[config["dataset"].lower()]

    if "cifar" in config.dataset.lower():
        data_mean = (0.4914, 0.4822, 0.4465)
        data_stddev = (0.2023, 0.1994, 0.2010)
        # Augment the data if specified
        if config.augment:
            transform_train = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomCrop(32, padding=4),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(data_mean, data_stddev),
                ]
            )
        else: # Else only normalize
            transform_train = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(data_mean, data_stddev),
                ]
            )
    elif "mnist" in config.dataset.lower():
        data_mean = (0.1307,)
        data_stddev = (0.3081,)
        transform_train = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(data_mean, data_stddev),
            ]
        )
    elif "pcam" in config.dataset.lower():
        data_mean = (0.701, 0.538, 0.692)
        data_stddev = (0.235, 0.277, 0.213)
        transform_train = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(data_mean, data_stddev),
            ]
        )
    else:
        raise ValueError(f"Unkown preprocessing for datasets '{config.dataset}'")

    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ]
    )

    if "rotmnist" in config.dataset.lower():
        training_set = dataset(root=data_root, stage="train", download=True, transform=transform_train, data_fraction=data_fraction, only_3_and_8=config.only_3_and_8)
        validation_set = MNIST_rot(root=data_root, stage="validation", download=True, transform=transform_test, data_fraction=data_fraction, only_3_and_8=config.only_3_and_8)  # Use test transform
        test_set = dataset(root=data_root, stage="test", download=True, transform=transform_test, data_fraction=data_fraction, only_3_and_8=config.only_3_and_8)
    elif "mnist" in config.dataset.lower():
        training_set = get_MNIST(root=data_root, train=True, transform=transform_train, data_fraction=data_fraction, only_3_and_8=config.only_3_and_8)
        validation_set = get_MNIST(root=data_root, train=True, transform=transform_test, data_fraction=data_fraction, only_3_and_8=config.only_3_and_8)
        test_set = get_MNIST(root=data_root, train=True, transform=transform_test, data_fraction=data_fraction, only_3_and_8=config.only_3_and_8)
    elif "both" in config.evaluate.lower():
        mnist_set = MNIST_rot(root=data_root, stage="validation", download=True, transform=transform_test, data_fraction=data_fraction, only_3_and_8=config.only_3_and_8)  # Use test transform
        rotmnist_set = get_MNIST(root=data_root, train=True, transform=transform_test, data_fraction=data_fraction, only_3_and_8=config.only_3_and_8)
        validation_set = torch.utils.data.ConcatDataset([mnist_set, rotmnist_set])
    else:   
        training_set = dataset(root=data_root, train=True, download=True, transform=transform_train, data_fraction=data_fraction)
        test_set = dataset(root=data_root, train=False, download=True, transform=transform_test, data_fraction=data_fraction)

    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    dataloaders = {"train": training_loader, "test": test_loader}
    if "pcam" in config.dataset.lower():
        validation_set = dataset(
            root=data_root, train=False, valid=True, download=False, transform=transform_test
        )
        val_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        dataloaders["validation"] = val_loader
    elif "mnist" in config.dataset.lower():
        # The test loader is the same as the validation loader
        dataloaders["validation"] = torch.utils.data.DataLoader(
            validation_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    else:
        dataloaders["validation"] = test_loader  # TODO: this will finetune on the test dataset (already fixed this for mnist)


    # Calculate label distribution in training set
    train_labels = [label for _, label in training_loader.dataset]
    train_label_counts = Counter(train_labels)

    # Calculate label distribution in validation set
    val_labels = [label for _, label in dataloaders['validation'].dataset]
    val_label_counts = Counter(val_labels)

    # Calculate label distribution in test set
    test_labels = [label for _, label in test_loader.dataset]
    test_label_counts = Counter(test_labels)


        

    print("\n"+"-"*30)
    print(f"Running experiment on {config.dataset.lower()} dataset")
    print(f"Number of train samples: {len(training_loader.dataset)}")
    print(f"Number of validation samples: {len(dataloaders['validation'].dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")

    print("\nLabel distribution in training set:")
    for label, count in sorted(train_label_counts.items()):
        print(f"Label {label}: {count} samples")

    print("\nLabel distribution in validation set:")
    for label, count in sorted(val_label_counts.items()):
        print(f"Label {label}: {count} samples")

    print("\nLabel distribution in test set:")
    for label, count in sorted(test_label_counts.items()):
        print(f"Label {label}: {count} samples")
    print("\n"+"-"*30)

    return dataloaders


def get_MNIST(root, train, transform, data_fraction, only_3_and_8):
        mnist_full = torchvision.datasets.MNIST(root=root, train=train, download=True, transform=transform)

        num_samples = len(mnist_full)
        if only_3_and_8:
            # Filter indices for samples with labels 3 and 8
            indices = []
            for idx, (image, label) in enumerate(mnist_full):
                if label == 3 or label == 8:
                    indices.append(idx)
                    
                    if label == 3:
                        mnist_full.targets[idx] = 0
                    else:
                        mnist_full.targets[idx] = 1

            num_samples = len(indices)


        # Randomly select a fraction of filtered indices
        num_selected_samples = int(data_fraction * num_samples)
        selected_indices = np.random.choice(indices, num_selected_samples, replace=False)

        # Create the training set with a fraction of the data
        training_set = torch.utils.data.Subset(mnist_full, selected_indices)

        return training_set