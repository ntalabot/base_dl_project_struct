"""
Module for training models with PyTorch.
"""

import time

import torch


def train(model, dataloaders, loss_fn, optimizer, n_epochs, scheduler=None, metric_fns={},
          valid_every=1, verbose=True):
    """
    Train the model and return the training history.
    
    Parameters
    ----------
    model: PyTorch model
        The model to train.
    dataloaders: dict of dataloaders
        Contains the train and validation DataLoaders with respective keys 
        "train" and "valid".
    loss_fn: callable
        A callable that takes (predictions, targets) as input.
    optimizer: PyTorch optimizer
        Optimzer for the SGD algorithm.
    n_epochs: int
        Number of training epochs (pass over the whole data).
    scheduler: pytorch scheduler (optional)
        A learning rate scheduler for the optimizer.
    metric_fns: dict
        Dictionary where keys are metric names (str), and values are metric 
        function (callable, taking (predictions, targets) as inputs).
    valid_every: int (default=1)
        Validation will be performed every valid_every epochs. By default (=1), it is done
        every epoch which might significantly slow down the training.
        First and last epochs are always validated.
    verbose: int (default=True)
        Enable or disable the verbosity of the training.
    
    Returns
    -------
    history : dict
        Dictionary containing the training history. It contains one dict for training
        and one for validation.
    """
    if verbose:
        start_time = time.time()
    
    model_device = next(model.parameters()).device
    history = {}
    for split in ['train', 'valid']:
        history[split] = {"loss": [], "epoch": []}
        for key in metric_fns.keys():
            history[split][key] = []
    history['train']["lr"] = []
    
    for epoch in range(n_epochs):
        # Apply schedule if applicable
        if scheduler is not None:
            scheduler.step(epoch)
            history["train"]["lr"].append(scheduler.get_lr()[0])
        else:
            history["train"]["lr"].append(optimizer.state_dict()["param_groups"][0]["lr"])
        if verbose and epoch > 0 and \
           history["train"]["lr"][-2] != history["train"]["lr"][-1]:
            print("Learning rate updated from", history["train"]["lr"][-2], 
                  "to", history["train"]["lr"][-1])

        if verbose:
            print(f"Epoch {epoch}/{n_epochs - 1}:", end=" ", flush=True)
        
        for phase in ["train", "valid"]:
            if phase == 'train':
                model.train()
            else:
                if not(epoch == 0 or (epoch+1) % valid_every == 0 or (epoch+1) == n_epochs):
                    continue
                model.eval()
            history[phase]["epoch"].append(epoch)

            running_loss = 0
            running_metrics = {}
            for key in metric_fns.keys():
                running_metrics[key] = 0

            # Iterate over the data
            for batch in dataloaders[phase]:
                inputs = batch[0].to(model_device)
                targets = batch[1].to(model_device)
                
                with torch.set_grad_enabled(phase == "train"):
                    # Forward pass
                    preds = model(inputs)
                    
                    # Loss
                    loss = loss_fn(preds, targets)
                    running_loss += loss.item() * len(inputs)
                    
                    # Metrics
                    for key, metric_fn in metric_fns.items():
                        running_metrics[key] += metric_fn(preds, targets).item() * len(inputs)
                        
                    if phase == "train":
                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
            
            # Save statistics
            history[phase]["loss"].append(running_loss / len(dataloaders[phase].dataset))
            for key in metric_fns.keys():
                history[phase][key].append(running_metrics[key] / len(dataloaders[phase].dataset))
            
            if verbose:
                phase_msg = f"{phase.capitalize()} loss={history[phase]['loss'][-1]:.6f}"
                for key in metric_fns.keys():
                    phase_msg += f" - {key}={history[phase][key][-1]:.6f}"
                print(phase_msg, end=" ", flush=True)
        print()

    if verbose:
        duration = time.time() - start_time
        duration_msg = "{:.0f}h {:02.0f}min {:02.0f}s".format(duration // 3600, (duration // 60) % 60, duration % 60)
        print("Training took %s." % duration_msg)    
    return history