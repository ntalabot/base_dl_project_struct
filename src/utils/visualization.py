"""
Utility module for visualization of data, results, etc.
"""

import matplotlib.pyplot as plt


def plot_train(history):
    """
    Return a figure of the training history.
    
    Parameters
    ----------
    history : dict
        Dictionary containing the training history. It contains one dict for
        training and one for validation.
    
    Returns
    -------
    fig : matplotlib.pyplot figure
        The pyplot figure object. Can be showed or saved by the user.
    """
    # Find the metrics
    metrics = []
    for key in history['train'].keys():
        if key not in ['epoch', 'lr', 'loss']:
            metrics.append(key)
    n_metrics = len(metrics)

    # Plot the figure
    fig, ax = plt.subplots(1, 1+n_metrics, figsize=(12,6))
    if n_metrics > 0:
        axs = {'loss': ax[0]}
        axs.update({metrics[i]: ax[i+1] for i in range(n_metrics)})
    else:
        axs = {'loss': ax} 

    for name in ['loss'] + metrics:
        for split, color in zip(['train', 'valid'], ['C0', 'C1']):
            axs[name].plot(history[split]['epoch'], history[split][name], 
                           color=color)

        axs[name].set_title(name.capitalize())
        axs[name].set_xlabel('Epoch')
        axs[name].set_ylabel(name.capitalize())
        axs[name].legend(["train", "valid"])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig