import os
import logging

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
PLOT_DIR = 'figures'

if not os.path.exists(PLOT_DIR):
    os.mkdir(PLOT_DIR)


def batch(x, y, batch_size=1, shuffle=True):
    assert len(x) == len(
        y), "Input and target data must contain same number of elements"
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()

    n = len(x)

    if shuffle:
        rand_perm = torch.randperm(n)
        x = x[rand_perm]
        y = y[rand_perm]

    batches = []
    for i in range(n // batch_size):
        x_b = x[i * batch_size: (i + 1) * batch_size]
        y_b = y[i * batch_size: (i + 1) * batch_size]

        batches.append((x_b, y_b))
    return batches





def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def seq_map_values(A, range_end):
    # Calculate the total number of elements to distribute
    total_elements = range_end + 1
    num_values = len(A)
    
    # Calculate the number of elements to assign to each value in A
    elements_per_value = total_elements // num_values

    # Handle any remaining elements that don't fit evenly
    remaining_elements = total_elements % num_values

    # Initialize the mapping dictionary
    mapping = {}
    current_index = 0

    for value in A:
        # Calculate the number of elements for this value
        num_elements = elements_per_value + (1 if remaining_elements > 0 else 0)
        remaining_elements -= 1

        # Map each element in the range to the current value
        for i in range(current_index, current_index + num_elements):
            mapping[i] = value

        # Update the index for the next value
        current_index += num_elements

    return mapping


def show_model_layers(model, logger=None):
    for k, v in model.named_parameters():
        if logger is not None:
            logger.info(f"layer: {k}, Shape:{v.shape} No. Params: {v.numel()}")
        else:
            # logging.info(f"layer: {k}, has number of params: {v.numel()}")
            logging.info(f"layer: {k}, Shape:{v.shape} No. Params: {v.numel()}")





















