"""
Optimizer utilities and factory functions.

"""

import logging
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


def get_optimizer(
    parameters: Iterable["nn.Parameter"],
    optimizer_name: str = "adamw",
    lr: float = 0.001,
    weight_decay: float = 0.01,
    **kwargs,
) -> "Optimizer":
    """
    Factory function to create optimizers.

    Args:
        parameters: Model parameters to optimize
        optimizer_name: Name of optimizer ('adam', 'adamw', 'sgd')
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Optimizer instance

    Example:
        >>> model = MyModel()
        >>> optimizer = get_optimizer(
        ...     model.parameters(),
        ...     optimizer_name='adamw',
        ...     lr=0.001,
        ...     weight_decay=0.01
        ... )

    References:
        - Loshchilov & Hutter (2019). "Decoupled Weight Decay Regularization"
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adamw":
        optimizer = AdamW(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name == "adam":
        optimizer = Adam(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name == "sgd":
        optimizer = SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=kwargs.get("momentum", 0.9), **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    logger.info(f"Created optimizer: {optimizer_name}, " f"lr={lr}, weight_decay={weight_decay}")

    return optimizer


# Recommended settings
OPTIMIZER_CONFIGS = {
    "adamw_default": {"optimizer_name": "adamw", "lr": 0.001, "weight_decay": 0.01, "betas": (0.9, 0.999)},
    "adam_legacy": {"optimizer_name": "adam", "lr": 0.001, "weight_decay": 0.0},  # Adam doesn't handle weight decay well
    "sgd_momentum": {"optimizer_name": "sgd", "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0001},
}


def init_weights_he_normal(module: "nn.Module") -> None:
    """
    Initialize weights using He Normal initialization.

    Best for ReLU activations. Maintains variance across layers.

    Args:
        module: PyTorch module to initialize

    Example:
        >>> model = MyModel()
        >>> model.apply(init_weights_he_normal)

    References:
        - He et al. (2015). "Delving Deep into Rectifiers"
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.LSTM, nn.GRU)):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)


def init_weights_xavier_uniform(module: "nn.Module") -> None:
    """
    Initialize weights using Xavier Uniform initialization.

    Best for tanh/sigmoid activations.

    Args:
        module: PyTorch module to initialize
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
