"""
Module used for defining losses.
"""

import torch.nn as nn


def get_MSELoss_fn(reduction='mean'):
    """
    Return the Mean Squared Error loss function.

    Parameters
    ----------
    reduction : str (default = 'mean')
        Reduction scheme for the loss, see PyTorch documentation for details.
        'mean' return the average of the batch, 'sum' return the sum of the
        batch, and 'none' return a batch of losses.
    
    Returns
    -------
    loss_fn : callable
        Loss function taking as input predictions and targets.
    """
    return nn.MSELoss(reduction=reduction)