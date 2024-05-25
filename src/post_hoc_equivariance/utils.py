import torch
import torchvision.transforms.functional as TF

import time
from functools import wraps


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

    return transforms  # B, T, C, H, W


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' took {end_time - start_time:.4f} seconds")
        return result
    return wrapper