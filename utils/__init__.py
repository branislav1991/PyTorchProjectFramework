import json
import math
import numpy as np
import os
from pathlib import Path
import torch
from torch.optim import lr_scheduler


def transfer_to_device(x, device):
    """Transfers pytorch tensors or lists of tensors to GPU. This
        function is recursive to be able to deal with lists of lists.
    """
    if isinstance(x, list):
        for i in range(len(x)):
            x[i] = transfer_to_device(x[i], device)
    else:
        x = x.to(device)
    return x


def parse_configuration(config_file):
    """Loads config file if a string was passed
        and returns the input if a dictionary was passed.
    """
    if isinstance(config_file, str):
        with open(config_file) as json_file:
            return json.load(json_file)
    else:
        return config_file


def get_scheduler(optimizer, configuration, last_epoch=-1):
    """Return a learning rate scheduler.
    """
    if configuration['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=configuration['lr_decay_iters'], gamma=0.3, last_epoch=last_epoch)
    else:
        return NotImplementedError('learning rate policy [{0}] is not implemented'.format(configuration['lr_policy']))
    return scheduler


def stack_all(list, dim=0):
    """Stack all iterables of torch tensors in a list (i.e. [[(tensor), (tensor)], [(tensor), (tensor)]])
    """
    return [torch.stack(s, dim) for s in list]