import math
import random
import time
from functools import wraps

import torch
import torchvision.transforms as tvtf
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import LambdaLR, StepLR
from tqdm import tqdm


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

def evaluate(model, val_loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
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

def test(model, test_loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
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

def linear_warmup_cosine_lr_scheduler(
    optimizer,
    warmup_time_ratio: float,
    T_max: int,
) -> torch.optim.lr_scheduler:
    """
    Creates a cosine learning rate scheduler with a linear warmup time determined by warmup_time_ratio.
    The warm_up increases linearly the learning rate from zero up to the defined learning rate.

    Args:
        warmup_time_ratio: Ratio in normalized percentage, e.g., 10% = 0.1, of the total number of iterations (T_max)
        T_max: Number of iterations
    """
    T_warmup = int(T_max * warmup_time_ratio)

    def lr_lambda(epoch):
        # linear warm up
        if epoch < T_warmup:
            return epoch / T_warmup
        else:
            progress_0_1 = (epoch - T_warmup) / (T_max - T_warmup)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress_0_1))
            return cosine_decay

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

# https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html
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