"""
General utility module.
"""

import random

import numpy as np
import torch

def set_seed(seed):
    """Set the manual seed for python, numpy, and pytorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)