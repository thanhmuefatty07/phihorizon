"""Utility functions for training, data processing, and optimization."""


# Lazy imports to avoid PyTorch DLL loading issues during test collection
def __getattr__(name):
    if name in ["clip_grad_norm", "get_grad_norm"]:
        try:
            from src.utils.training_utils import clip_grad_norm, get_grad_norm

            return locals()[name]
        except ImportError:
            # Graceful fallback if PyTorch unavailable
            def dummy_clip_grad_norm(*args, **kwargs):
                return None

            def dummy_get_grad_norm(*args, **kwargs):
                return 0.0

            return dummy_clip_grad_norm if name == "clip_grad_norm" else dummy_get_grad_norm

    elif name in ["kelly_criterion_scalar", "kelly_criterion_batch"]:
        try:
            from src.utils.fast_math import kelly_criterion_batch, kelly_criterion_scalar

            return locals()[name]
        except ImportError:
            # Graceful fallback for fast math functions
            def dummy_kelly(*args, **kwargs):
                raise ImportError("Fast math dependencies not available")

            return dummy_kelly

    elif name in ["get_optimizer", "OPTIMIZER_CONFIGS", "init_weights_he_normal", "init_weights_xavier_uniform"]:
        try:
            from src.utils.optimizer_utils import (
                OPTIMIZER_CONFIGS,
                get_optimizer,
                init_weights_he_normal,
                init_weights_xavier_uniform,
            )

            return locals()[name]
        except ImportError:
            # Graceful fallback for optimizer utilities
            def dummy_get_optimizer(*args, **kwargs):
                raise ImportError("PyTorch dependencies not available")

            dummy_configs = {}

            def dummy_init_weights(*args, **kwargs):
                pass

            return {
                "get_optimizer": dummy_get_optimizer,
                "OPTIMIZER_CONFIGS": dummy_configs,
                "init_weights_he_normal": dummy_init_weights,
                "init_weights_xavier_uniform": dummy_init_weights,
            }.get(name, lambda *args, **kwargs: None)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "clip_grad_norm",
    "get_grad_norm",
    "get_optimizer",
    "OPTIMIZER_CONFIGS",
    "init_weights_he_normal",
    "init_weights_xavier_uniform",
    "kelly_criterion_scalar",
    "kelly_criterion_batch",
]
