import pathlib
import random

from torchvision.datasets import ImageFolder


class PCam(ImageFolder):
    """
    PCam dataset.

    Download the dataset from https://drive.google.com/file/d/1THSEUCO3zg74NKf_eb3ysKiiq2182iMH/view?usp=sharing

    For more information, please refer to the README.md of the repository.
    """

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False, valid=False, data_fraction=0.001
    ):
        if train and valid:
            raise ValueError("PCam 'valid' split available only when train=False.")

        root = pathlib.Path(root) / "PCam"
        split = "train" if train else ("valid" if valid else "test")
        directory = root / split
        if not (root.exists() and directory.exists()):
            raise FileNotFoundError(
                "Please download the PCam dataset. How to download it can be found in 'README.md'"
            )

        super().__init__(root=directory, transform=transform, target_transform=target_transform)

        # Reduce the dataset size if specified by data_fraction sample a fraction of the data
        if data_fraction < 1:

            print(f"Total length of the dataset: {len(self)}")

            # Get list of all data indices
            all_indices = list(range(len(self)))

            # Calculate the number of samples to keep
            num_samples_to_keep = int(len(all_indices) * data_fraction)

            # Randomly select subset of indices
            selected_indices = random.sample(all_indices, num_samples_to_keep)

            # Filter out the selected samples
            self.samples = [self.samples[i] for i in selected_indices]

            print(f"Reduced length of the dataset: {len(self)}")
