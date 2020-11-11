"""
Module used for defining optimizers and schedulers.
"""

import torch.optim as optim


def get_optimizer(model, lr=0.001):
    """
    Return the Adam optimizer.

    Parameters
    ----------
    model: pytorch model
        The model which parameters will be optimized.
    lr: float
        Learning rate for the optimizer.
    
    Returns
    -------
    optimizer: pytorch optimizer
        An optimizer over the model parameters.
    """
    return optim.Adam(model.parameters(), lr=lr)


def get_scheduler(optimizer, step_size, decay_factor=0.1):
    """
    Return a step decay scheduler for the optimizer.

    Parameters
    ----------
    optimizer : pytorch optimizer
        The optimizer for which the schedule will be applied
    step_size: int
        Number of epochs after which the learning rate will decay.
    decay_factor: float
        The learning rate is multiplied by this factor at each decay.
    
    Returns
    -------
    scheduler: pytorch scheduler
        A learning rate scheduler for the optimizer.
    """
    return optim.lr_scheduler.StepLR(optimizer, step_size, decay_factor)
