"""
Module for model definition, saving, and loading.
"""

import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        out = self.conv(x)
        return out