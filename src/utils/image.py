"""
Utility module for images.
"""

import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def normalize(image):
    """Normalize the image with ImageNet statistics."""
    out = image / 255
    out = (out - IMAGENET_MEAN) / IMAGENET_STD
    return np.float32(out)


def reverse_normalize(image):
    """Reverse the ImageNet normalization of the image."""
    out = image * IMAGENET_STD + IMAGENET_MEAN
    out = out * 255
    return np.uint8(out)