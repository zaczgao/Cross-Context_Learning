import torch
from optimizers.lars import LARS


def get_optimizer(type, optim_params, lr, momentum, weight_decay, nesterov=False):
    if type == 'lars':
        sgd = torch.optim.SGD(optim_params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        optimizer = LARS(sgd)
    elif type == 'sgd':
        optimizer = torch.optim.SGD(optim_params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    elif type == "adam":
        optimizer = torch.optim.Adam(optim_params, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer
