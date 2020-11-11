"""
Module implementing metrics for performance evaluations.
"""

import torch.nn as nn


def get_metric_fn(reduction='mean'):
    """
    Return the Mean Squared Error function.

    Parameters
    ----------
    reduction : str (default = 'mean')
        Reduction scheme for the loss, see PyTorch documentation for details.
        'mean' return the average of the batch, 'sum' return the sum of the
        batch, and 'none' return a batch of losses.
    
    Returns
    -------
    metric_fn : callable
        Metric function taking as input predictions and targets.
    """
    return nn.MSELoss(reduction=reduction)