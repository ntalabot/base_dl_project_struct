"""
Main module for data loading and preparation, when
working with images.
"""

import os
import imageio as io

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from src.utils.image import normalize


class ImagesDataset(Dataset):
    """
    PyTorch Dataset that loads images from disk with each getitem.
    
    It creates synthetic targets by taking the negative of the images.
    """

    def __init__(self, imagedir=None):
        """
        Initialize the Dataset with a folder of images.

        Parameters
        ----------
        imagedir: str
            Path to the directory of images, without subfolders.
        """
        super().__init__()

        self.imagedir = imagedir
        self.filenames = sorted(os.listdir(self.imagedir))

    
    def __len__(self):
        return len(self.filenames)

    
    def __getitem__(self, idx):
        image = io.imread(os.path.join(self.imagedir, self.filenames[idx]))
        target = 255 - image

        image = normalize(image)
        target = normalize(target)
        if image.ndim == 3:
            # HWC to CHW
            image = image.transpose(2, 0, 1)
            target = target.transpose(2, 0, 1)

        return image, target


def pad_collate_fn(batch):
    """
    Collate function that zero-pads input images to the same size.
    
    Parameters
    ----------
    batch : list of array
        List of image numpy array.
    
    Returns
    -------
    pad_batch : tensor
        Tensor of padded images of the same size.
    """
    # Find largest shape
    shapes = [item[0].shape for item in batch]
    height = max([shape[-2] for shape in shapes])
    width = max([shape[-1] for shape in shapes])
    
    # Pad images to the largest shape 
    pad_batch = []
    for shape, item in zip(shapes, batch):
        pad_height = (height - shape[-2]) / 2
        pad_width = (width - shape[-1]) / 2
        padding = [(int(np.floor(pad_height)), int(np.ceil(pad_height))), 
                   (int(np.floor(pad_width)), int(np.ceil(pad_width)))]
        if item[0].ndim == 3:
            padding = [(0, 0)] + padding

        pad_batch.append((np.pad(item[0], padding, 'constant'),
                          np.pad(item[1], padding, 'constant')))
    return default_collate(pad_batch)