"""
Module for testing models (evaluation, predictions).
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision


def predict_dataloader(model, dataloader, discard_target=True):
    """
    Return predictions for the given dataloader and model.
    
    Parameters
    ----------
    model : pytorch model
        The pytorch model to predict with.
    dataloader : pytorch dataloader
        Dataloader returning batches of inputs. Shuffle should be False if
        order is important in the predictions
    discard_target : bool (default = True)
        If True, onyl the first element of the batch is kept. This is useful if
        the dataloader returns batches as (inputs, targets, ...). If it only
        returns inputs directly as a tensor, set this to False.
    
    Returns
    -------
    predictions : list of tensors
        List of predicted batch.
    """
    model_device = next(model.parameters()).device
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if discard_target:
                batch = batch[0]
                
            batch = batch.to(model_device)
            predictions.append(model(batch))
    return predictions
    
def predict(model, inputs, batch_size=None):
    """
    Output predictions for the given input and model.
    
    Parameters
    ----------
    model : pytorch model
        The pytorch model to predict with.
    inputs : tensor
        Input tensors.
    batch_size : int (optional)
        Number of images to send to the network at once. Useful if inputs is 
        too large for the GPU. If not given, inputs is sent whole.
    
    Returns
    -------
    predictions : tensor
        Tensor of predictions on the same device as the input.
    """
    model_device = next(model.parameters()).device
    predictions = []
    model.eval()
    with torch.no_grad():
        if batch_size is None:
            batch_size = len(inputs)
        n_batches = int(np.ceil(len(inputs) / batch_size))
        for i in range(n_batches):
            batch = inputs[i * batch_size: (i + 1) * batch_size]
            batch = batch.to(model_device)

            preds = model(batch).to(inputs.device)
            predictions.append(preds)
    predictions = torch.cat(predictions)
    return predictions


def evaluate_dataloader(model, dataloader, metrics):
    """
    Return the average metric values for the given dataloader and model.
    
    Parameters
    ----------
    model : pytorch model
        The pytorch model to predict with.
    dataloader : pytorch dataloader
        Dataloader returning batches of inputs.
    metrics : dict
        Dictionary where keys are metric names (str), and values are metric 
        function (callable, taking (predictions, targets) as inputs).
    
    Returns
    -------
    values : dict
        Dictionary where keys are metric names (str), and values are metric 
        average values.
    """
    values = {}
    for key in metrics.keys():
        values[key] = 0
    
    model_device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(model_device)
            targets = batch[1].to(model_device)
            
            preds = model(inputs)

            for key, metric in metrics.items():
                values[key] += metric(preds, targets).item() * inputs.shape[0]
            
    for key in values.keys():
        values[key] /= len(dataloader.dataset)
    return values

def evaluate(model, inputs, targets, metrics, batch_size=None):
    """
    Return the average metric values for the given inputs and model.
    
    Parameters
    ----------
    model : pytorch model
        The pytorch model to predict with.
    inputs : tensor
        Input tensors.
    targets : tensor
        Target tensors.
    metrics : dict
        Dictionary where keys are metric names (str), and values are metric 
        function (callable, taking (predictions, targets) as inputs).
    batch_size : int (optional)
        Number of images to send to the network at once. Useful if inputs is 
        too large for the GPU. If not given, inputs is sent whole.
    
    Returns
    -------
    values : dict
        Dictionary where keys are metric names (str), and values are metric 
        average values.
    """        
    # Make predictions
    predictions = predict(model, inputs, batch_size=batch_size)
    
    # Compute metrics
    values = {}
    for key, metric in metrics.items():
        values[key] = metric(predictions, targets).item()
    return values