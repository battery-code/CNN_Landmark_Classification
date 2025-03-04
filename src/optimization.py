import torch
import torch.nn as nn
import torch.optim


def get_loss(use_cuda: bool = False):
    """
    Get an instance of the CrossEntropyLoss (useful for classification),
    optionally moving it to the GPU if use_cuda is set to True
    """
    loss  = nn.CrossEntropyLoss() 

    # Using GPU if use_cuda is True
    if use_cuda and torch.cuda.is_available():
        loss = loss.cuda()  # Moves the loss function to GPU if available

    return loss


def get_optimizer(
    model: nn.Module,
    optimizer: str = "SGD",
    learning_rate: float = 0.01,
    momentum: float = 0.95,
    weight_decay: float = 0,
):
    """
    Returns an optimizer instance

    :param model: the model to optimize
    :param optimizer: one of 'SGD' or 'Adam'
    :param learning_rate: the learning rate
    :param momentum: the momentum (if the optimizer uses it)
    :param weight_decay: regularization coefficient
    """
    if optimizer.lower() == "sgd":
        # Instantiate SGD optimizer
        opt = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum, 
            weight_decay=weight_decay  # Weight decay for L2 regularization
        )

    elif optimizer.lower() == "adam":
        # Instantiate Adam optimizer 
        opt = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay  # Weight decay for L2 regularization
        )
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    return opt
